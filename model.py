import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils import *
from allennlp.nn.util import masked_softmax
from metrics import *
import sys
# sys.path.append("./virtualhome/simulation/")
# sys.path.append("./virtualhome/dataset_utils/")
# import add_preconds
# import evolving_graph.check_programs as check_programs
import multiprocessing
from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN_with_history(nn.Module):
    def __init__(self, hidden_size, rnn_type, rnn_numlayers, bidirectional_rnn, data):
        super(EncoderRNN_with_history, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding_size = 300

        word_embedding_matrix = np.load("./disjoint_data/word_embedding_matrix.npy")
        
        self.embedding = nn.Embedding(word_embedding_matrix.shape[0], 300)
        self.embedding.weight.data.copy_(torch.from_numpy(word_embedding_matrix))
        self.embedding.weight.requires_grad = True

        self.rnn_type = rnn_type
        self.rnn_numlayers = rnn_numlayers
        self.bidirectional_rnn = bidirectional_rnn

        if rnn_type=="lstm":
            self.rnn = nn.LSTM(input_size=300, 
                                hidden_size=hidden_size, 
                                num_layers = self.rnn_numlayers, 
                                bidirectional=self.bidirectional_rnn,
                                batch_first = True)
        else:
            self.rnn = nn.GRU(input_size=300, 
                                hidden_size=hidden_size, 
                                num_layers = self.rnn_numlayers, 
                                bidirectional=self.bidirectional_rnn,
                                batch_first = True)

        self.dp1 = nn.Dropout(0.3)
        self.dp2 = nn.Dropout(0.3)


        # weights for self attention
        self.W_attn1 = nn.Linear(self.hidden_size, int(self.hidden_size /2) )
        self.W_attn2 = nn.Linear(int(self.hidden_size /2) , 1)


    def forward(self, input, seq_lengths,hidden_cell):
        batch_size = input.size()[0] # batch-first
        max_unpacked_length = input.size()[1]
        seq_lengths = seq_lengths.view(batch_size)
        embedded = self.embedding(input).view(batch_size, input.size()[1], -1)
        packed_input = pack_padded_sequence(embedded, seq_lengths, batch_first=True)
        
        
        self.rnn.flatten_parameters()
        
        hidden = hidden_cell[0]
        cell = hidden_cell[1]
        hidden = hidden.transpose(0,1)
        cell = cell.transpose(0,1)

        hidden = hidden.contiguous()
        cell = cell.contiguous()
        if self.rnn_type=="lstm":
            output, hidden_cell = self.rnn(packed_input, (hidden,cell))
            hidden = hidden_cell[0].transpose(0,1)
            cell = hidden_cell[1].transpose(0,1)
            output, seq_lengths =  pad_packed_sequence(output, batch_first=True, total_length=max_unpacked_length)
        else:
            output, hidden = self.rnn(packed_input, hidden)
            hidden = hidden.transpose(0,1)
            cell = cell.transpose(0,1)
            output, seq_lengths =  pad_packed_sequence(output, batch_first=True, total_length=max_unpacked_length)
        
       
        seq_lengths = seq_lengths.cuda()

        
        t = output.contiguous().view(output.size(0) * output.size(1), self.hidden_size)
        t = self.W_attn1(t) # (B * M) x (H/2)
        t = torch.tanh(t)
        t = self.W_attn2(t)
        t = t.view(batch_size, -1)
        alpha = torch.softmax(t, dim=1)
        
        alpha = alpha.view(alpha.size()[0], alpha.size()[1], 1)
        alpha = alpha.repeat(1,1,self.hidden_size)
        self_attended_output = torch.sum (alpha * output , dim = 1)

        

        return output, (hidden, cell), seq_lengths.view(batch_size, -1), self_attended_output

    def initHidden(self,batch_size):
        if not self.bidirectional_rnn :
            return torch.zeros(1, batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(2, batch_size, self.hidden_size, device=device)

class AttnDecoderRNN_with_history(nn.Module):
    def __init__(self, hidden_size, rnn_type, data, config, dropout_p=0.1, encoder_hidden_size=None):
        super(AttnDecoderRNN_with_history, self).__init__()
        self.config = config
        self.rnn_type = rnn_type
        self.embedding_size = config.embedding_dim_decoder
        self.hidden_size = hidden_size
        if encoder_hidden_size != None:
            self.encoder_hidden_size = encoder_hidden_size
        else:
            self.encoder_hidden_size = hidden_size
        
        self.embedding_size_env = 300     
        
        if not self.config.np_context:
            self.input_size = self.embedding_size + self.hidden_size  # self.hidden size is added for dimension of x_att vector 
        else :
            self.input_size = 2 * self.embedding_size + self.hidden_size  # self.hidden size is added for dimension of x_att vector 
        self.input_size_env = self.embedding_size + self.embedding_size_env  # self.hidden size is added for dimension of x_att-env vector 
        
        self.action_object_pair2id = json.load(open("./disjoint_data/step2id.json", "r"))
        self.id2action_object_pair = json.load(open("./disjoint_data/id2step.json", "r"))
        self.id2action_object_pair = {int(k):str(v) for k,v in self.id2action_object_pair.items()}
        
        
        # env_embedding_matrix = np.load("./disjoint_data/all_object2id_graph_extended_embedding.npy")
        # # self.embedding_env = nn.Embedding(data.env_entity_emb.shape[0], data.embedding_dim)
        # self.embedding_env = nn.Embedding(env_embedding_matrix.shape[0],300)
        # self.embedding_env.weight.data.copy_(torch.from_numpy(env_embedding_matrix))
        # self.embedding_env.weight.requires_grad = True


        # self.env_np = torch.nn.Parameter(torch.from_numpy(env_embedding_matrix).float())
        # self.env_np.requires_grad = True

        self.Wv = nn.Linear(self.hidden_size + self.encoder_hidden_size, self.embedding_size)
        self.W_att = nn.Linear(self.hidden_size  + self.encoder_hidden_size, self.embedding_size)
        if self.config.np_context:
            self.W_att_np = nn.Linear(self.hidden_size  + self.embedding_size, self.embedding_size)

        self.Wv_env = nn.Linear(self.hidden_size + self.embedding_size_env, self.embedding_size)
        
        self.Wv_new = nn.Linear(self.hidden_size + self.hidden_size + self.embedding_size_env, self.embedding_size)
        self.W_att_env = nn.Linear(self.hidden_size  + self.embedding_size_env, self.embedding_size)

        word_embedding_matrix = np.load("./disjoint_data/word_embedding_matrix.npy")
        self.word_embedding_layer = nn.Embedding(word_embedding_matrix.shape[0], 300)
        self.word_embedding_layer.weight.data.copy_(torch.from_numpy(word_embedding_matrix))
        self.word_embedding_layer.requires_grad = True


        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first = True)
            self.rnn_env = nn.LSTM(self.input_size_env, self.hidden_size, batch_first = True)        
        else:
            self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first = True)
            self.rnn_env = nn.GRU(self.input_size_env, self.hidden_size, batch_first = True)
        
        self.dp1 = nn.Dropout(0.3)
        self.dp2 = nn.Dropout(0.3)
        
        # self.num_actions = len(data.action2id)
        # self.num_objects = len(data.object2id)
        v_np = np.load("./disjoint_data/step_embedding_matrix.npy")
        # self.v = torch.nn.Parameter(torch.from_numpy(data.v_np).float())
        self.v = torch.nn.Parameter(torch.from_numpy(v_np).float())
        self.v.requires_grad = True
        
        self.W_att_self1 = nn.Linear(self.config.embedding_dim_decoder, int(self.config.embedding_dim_decoder/2))
        self.W_att_self2 = nn.Linear( int(self.config.embedding_dim_decoder/2) , 1)

        self.W_prob_linear = nn.Linear(self.hidden_size + self.embedding_size + self.config.embedding_dim_decoder, 1)

        
        obj_emb_np = np.load("./disjoint_data/object_embedding_new.npy")
        self.obj_emb = torch.nn.Parameter(torch.from_numpy(obj_emb_np).float())
        self.obj_emb.requires_grad = False

        self.object2id = json.load(open("./disjoint_data/object2id.json","r"))
        self.id2object = json.load(open("./disjoint_data/id2object.json","r"))

        
    def self_attention_decoder_input(self, history, mask=None):
        # given history get the attented decoder input
        # decoder is a list of embedding of size B x 1 x E
        # first concat and then follow steps same as self attention for encoder.
        if isinstance(history, list):
            history = torch.cat(history, dim=1)
        
        concat = history
        embed_size = concat.size(2)
        l = concat.size(1)
        concat = concat.contiguous()
        concat = concat.view(-1, embed_size)
        concat = self.W_att_self1(concat)
        concat = torch.tanh(concat)
        concat = self.W_att_self2(concat)
        concat = concat.view(-1, l)

        alpha = masked_softmax(concat, mask, dim=1, memory_efficient=True)
        
        
        alpha = alpha.view(alpha.size()[0], alpha.size()[1], 1)
        alpha = alpha.repeat(1,1,embed_size)
        dec_inp = torch.sum (alpha * history , dim = 1)
        
        return dec_inp

    
    def get_context_vector(self, hi, inp, all_encoder_hidden, mask_tensor):
        # calculate attention
        all_encoder_timesteps = all_encoder_hidden.size()[1]
        hi_rep = hi.repeat(1,all_encoder_timesteps,1)
        
        _h = torch.cat((hi_rep, all_encoder_hidden) ,2 )
        _mul = self.W_att(_h)
        _mul = torch.bmm(_mul, inp.view (inp.size()[0], self.embedding_size, 1))
        _mul = _mul.view(_mul.size()[0], all_encoder_timesteps)
        
        alpha = masked_softmax(_mul, mask_tensor.bool(), dim=1, memory_efficient=True)
        
        alpha = alpha.view(alpha.size()[0], alpha.size()[1], 1)
        alpha = alpha.repeat(1,1,self.hidden_size)
        x_att = torch.sum (alpha * all_encoder_hidden , dim = 1)
        
        return x_att, alpha

    

    
    

    # define a small function to get softmax over states using the lstm output
    def get_logits(self, dec_out, x_att):
        batch_size = dec_out.size()[0]
        hi = dec_out
        
        hi2 = torch.cat((hi, x_att.view(x_att.size()[0], 1, self.hidden_size)), dim = 2 )
        # hi2 = hi2.matmul(self.Wv)
        hi2 = self.Wv(hi2)
        hi2 = hi2.view(batch_size, self.embedding_size)
        v_norm = self.v
        v_norm = F.normalize(v_norm, p=2, dim=1)
        v_norm = v_norm.t()
        p = hi2.matmul(v_norm)
        # print(p.size())
        return p

    
    
    def get_coverage_similarity_score(self, logits, noun_ids, bds, bnc):
        batch_size = logits.size(0)
        noun_embs = self.word_embedding_layer(noun_ids)
        logit_argmax = logits

        all_max_scores = []
        similarity_indicator = []
        distinct_flag = []
        new_noun_covered_flag = []
        distinct_set = bds.copy()
        nouns_covered = bnc.copy()

        for b in range(batch_size):
            max_score = -100
            target_max_score = -100
            step_id = logit_argmax[b]
            selected_noun = None
            if step_id == int(self.action_object_pair2id["<eop>"]):
                all_max_scores.append(0.0)
                similarity_indicator.append(0)
                distinct_flag.append(0)
                new_noun_covered_flag.append(0)
                continue
            
            step = self.id2action_object_pair[int(step_id)]
            step = step.split("||")
            
            if len(step) == 1 :
                max_score = 1.0
                distinct_flag.append(1)
                # new_noun_covered_flag.append(0)

            else:
                objs = step[1:]
                objs_key = "||".join(objs) 
                if objs_key not in distinct_set[b]:
                    distinct_flag.append(1)
                    distinct_set[b].add(objs_key)
                else:
                    distinct_flag.append(0)
                    # new_noun_covered_flag.append(0)
                new_noun_decision = False
                for obj in objs:
                    objid = int(self.object2id[obj])
                    obj_emb = self.obj_emb[objid]
                    y = obj_emb.unsqueeze(0)
                    x = noun_embs[b].view(-1,300)
                    
                    x_n = x.norm(dim=1)[:, None]
                    y_n = y.norm(dim=1)[:, None]
                    x_norm = x / torch.max(x_n, 1e-8 * torch.ones_like(x_n))
                    y_norm = y / torch.max(y_n, 1e-8 * torch.ones_like(y_n))
                    cos_sim = torch.mm(x_norm, y_norm.transpose(0,1)).view(noun_embs.size(1))
                    
                    obj_score = cos_sim.max()
                    noun_id_index = int(cos_sim.argmax().item())
                    
                    
                    if float(obj_score) > max_score:
                        max_score = float(obj_score)
                        selected_noun = int(noun_ids[b][noun_id_index])
                       

            all_max_scores.append(max_score)
            if max_score > 0.8:
                similarity_indicator.append(1)
                if selected_noun not in nouns_covered[b]:
                    new_noun_covered_flag.append(1)
                    nouns_covered[b].add(selected_noun)
                else:
                    new_noun_covered_flag.append(0)

            else :
                similarity_indicator.append(0)
                new_noun_covered_flag.append(0)

            

        
        assert len(new_noun_covered_flag) == batch_size

        return all_max_scores, similarity_indicator, distinct_flag, new_noun_covered_flag, distinct_set, nouns_covered

    def forward(self, input, 
                env_entities = None,
                beginning_room_entities = None,
                noun_ids = None,
                padded_noun_phrase = None,
                mask_noun_phrase = None,
                hidden_cell = None,
                hidden_cell_env = None, 
                all_encoder_hidden = None, 
                seq_lengths = None, 
                mask_tensor = None, 
                use_teacher_forcing_tensor = None,
                max_decoder_length = None,
                training_mode = True,
                beam_size = 5,
                get_sampled_sequence = False,
                input_word_ids = None,
                input_noun_vecs = None, 
                preconds_file_path = None,
                environment = None):  

        hidden_cell_env = None
        beginning_room_entities = None
        env_entities = None
        
        target_given = input
        
        use_teacher_forcing = False
        if use_teacher_forcing_tensor.all() > 0.0 :
            use_teacher_forcing = True
        # if using teacher forcing the output sequnces will be of varying length, thus we need to use pack_pad_seqence again
        # for decoding in inference or for decoding fixed steps using past timestep decoder output, there is no need to pack
        
        batch_size = all_encoder_hidden.size()[0]
        self.rnn.flatten_parameters()
        self.rnn_env.flatten_parameters()

        # reshape hidden and cell states received
        if self.rnn_type == "lstm":
            hidden = hidden_cell[0].transpose(0,1)
            cell = hidden_cell[1].transpose(0,1)
            hidden_cell = (hidden,cell)

            

        else:
            hidden = hidden_cell[0].transpose(0,1)
            # hidden_env = hidden_cell_env[0].transpose(0,1)
        
        if training_mode:
            if use_teacher_forcing :
                max_unpacked_length = input.size()[1]
                
                all_logit_outs = []
                
                prev_hidden = hidden_cell[0] if self.rnn_type=="lstm" else hidden
                
                prev_hidden = prev_hidden.squeeze(0).unsqueeze(1)

               
                for dec_idx in range (max_unpacked_length):


                    decoder_input = input[:,dec_idx,:]
                    

                    x_att, alpha = self.get_context_vector(prev_hidden, decoder_input, all_encoder_hidden, mask_tensor)
                    assert (self.config.np_context == False)
                    if self.config.np_context:
                        x_att_np, alpha_np = self.get_context_vector_noun_phrase(prev_hidden, decoder_input, padded_noun_phrase, mask_noun_phrase)
                   
                    
                    decoder_input_text = torch.cat((decoder_input,x_att), dim = 1)
                    
                    decoder_input_text = decoder_input_text.unsqueeze(1)
                    
                    if self.rnn_type == "lstm":
                        dec_out_text, hidden_cell = self.rnn(decoder_input_text, hidden_cell)
                    else :
                        dec_out_text, hidden = self.rnn(decoder_input_text, hidden)

                    hidden_cell = (hidden_cell[0], hidden_cell[1])
                   
                    prev_hidden = dec_out_text
                    
                    
                    p = self.get_logits(dec_out_text, x_att)
                    

                    all_logit_outs.append(p)
                
                all_logit_outs = torch.stack(all_logit_outs, dim=1)

                return all_logit_outs, None

            else :
                print("Not teacher forcing")
                batch_distinct_sets = {}
                batch_distinct_sets_target = {}
                batch_noun_ids_covered = {}
                batch_noun_ids_covered_target = {}
                for bid in range(batch_size):
                    batch_distinct_sets[bid] = set()
                    batch_distinct_sets_target[bid] = set()
                    batch_noun_ids_covered[bid] = set()
                    batch_noun_ids_covered_target[bid] = set()

                # decode max_decoder_length steps
                decoder_input = self.v[self.action_object_pair2id["<bop>"]].detach().view(1,1,self.embedding_size)
                decoder_input = decoder_input.repeat(batch_size,1,1)
                prev_hidden = hidden_cell[0] if self.rnn_type=="lstm" else hidden
                prev_hidden = prev_hidden.squeeze(0).unsqueeze(1)
                
                all_logit_outs = []
                all_step_ids = []
                all_coverage_similarity_scores = []
                all_similarity_indicators = []

                all_coverage_similarity_scores_target = []
                all_similarity_indicators_target = []

                all_distinct_flags = []
                all_distinct_flags_target = []

                all_new_noun_covered_flags = []
                all_new_noun_covered_flags_target = []
                print(max_decoder_length)
                for dec_idx in range (max_decoder_length):
                    x_att, alpha = self.get_context_vector(prev_hidden, decoder_input, all_encoder_hidden, mask_tensor)
                    
                    decoder_input_text = torch.cat((decoder_input.squeeze(1),x_att), dim = 1)
                    
                    decoder_input_text = decoder_input_text.unsqueeze(1)
                    

                    if self.rnn_type == "lstm":
                        dec_out_text, hidden_cell = self.rnn(decoder_input_text, hidden_cell)
                        
                    else :
                        dec_out_text, hidden = self.rnn(decoder_input_text, hidden)
                        
                    

                    
                    prev_hidden = dec_out_text
                    

                    p = self.get_logits(dec_out_text, x_att)
                    
                    p2 = p.clone()
                    all_logit_outs.append(p)
                    
                    p2 = p2.detach()
                    
                    p2 = F.softmax(p2,dim=1)

                    if get_sampled_sequence :
                        p2 = torch.multinomial(p2, 1) # Batchsize
                        p2 = p2.squeeze(1)
                        p2 = p2.detach().cpu().numpy()
                    else:
                        p2 = p2.cpu().numpy()
                        p2 = np.argmax(p2, axis=1)
                    
                    all_step_ids.append(torch.from_numpy(p2))
                    
                    vv = self.v.clone().detach()
                    


                  
                    decoder_input = vv[p2].detach()
                        
                    
                    coverage_similarity_scores, similarity_indicator, distinct_flags, \
                        new_noun_covered_flags, batch_distinct_sets, batch_noun_ids_covered \
                            = self.get_coverage_similarity_score(torch.from_numpy(p2).clone().detach(), noun_ids, batch_distinct_sets,batch_noun_ids_covered )
                    
                    target_steps = target_given[:, dec_idx]
                    coverage_similarity_scores_target, similarity_indicator_target, distinct_flags_target, \
                        new_noun_covered_flags_target, batch_distinct_sets_target, batch_noun_ids_covered_target \
                            = self.get_coverage_similarity_score(target_steps.clone().detach(), noun_ids, batch_distinct_sets_target,batch_noun_ids_covered_target )

                    all_coverage_similarity_scores.append(np.array(coverage_similarity_scores) )
                    all_similarity_indicators.append(np.array(similarity_indicator))
                    all_distinct_flags.append(np.array(distinct_flags))
                    all_new_noun_covered_flags.append(np.array(new_noun_covered_flags))
                    
                    all_coverage_similarity_scores_target.append(np.array(coverage_similarity_scores_target))
                    all_similarity_indicators_target.append(np.array(similarity_indicator_target))
                    all_distinct_flags_target.append(np.array(distinct_flags_target))
                    all_new_noun_covered_flags_target.append(np.array(new_noun_covered_flags_target))




                
                all_logit_outs = torch.stack(all_logit_outs, dim=1)
                
                all_step_ids = torch.stack(all_step_ids, dim=1).cuda().to(all_logit_outs.get_device())
                
                all_coverage_similarity_scores = torch.from_numpy(np.array(all_coverage_similarity_scores)).cuda().to(all_logit_outs.get_device())
                all_coverage_similarity_scores = all_coverage_similarity_scores.transpose(0,1)

                all_similarity_indicators = torch.from_numpy(np.array(all_similarity_indicators)).cuda().to(all_logit_outs.get_device())
                all_similarity_indicators = all_similarity_indicators.transpose(0,1)

                all_distinct_flags = torch.from_numpy(np.array(all_distinct_flags)).cuda().to(all_logit_outs.get_device())
                all_distinct_flags = all_distinct_flags.transpose(0,1)
                
                all_new_noun_covered_flags = torch.from_numpy(np.array(all_new_noun_covered_flags)).cuda().to(all_logit_outs.get_device())
                all_new_noun_covered_flags = all_new_noun_covered_flags.transpose(0,1)
                

                all_coverage_similarity_scores_target = torch.from_numpy(np.array(all_coverage_similarity_scores_target)).cuda().to(all_logit_outs.get_device())
                all_coverage_similarity_scores_target = all_coverage_similarity_scores_target.transpose(0,1)

                all_similarity_indicators_target = torch.from_numpy(np.array(all_similarity_indicators_target)).cuda().to(all_logit_outs.get_device())
                all_similarity_indicators_target = all_similarity_indicators_target.transpose(0,1)

                all_distinct_flags_target = torch.from_numpy(np.array(all_distinct_flags_target)).cuda().to(all_logit_outs.get_device())
                all_distinct_flags_target = all_distinct_flags_target.transpose(0,1)

                all_new_noun_covered_flags_target = torch.from_numpy(np.array(all_new_noun_covered_flags_target)).cuda().to(all_logit_outs.get_device())
                all_new_noun_covered_flags_target = all_new_noun_covered_flags_target.transpose(0,1)
                if self.config.training_procedure == "CE":
                    return all_logit_outs # shape :  batch_size x num_max_steps x num_classes
                elif self.config.training_procedure == "SCST":
                    if get_sampled_sequence:
                        print(all_step_ids.size())
                        return all_logit_outs, all_step_ids, \
                            all_coverage_similarity_scores, all_similarity_indicators, all_distinct_flags, all_new_noun_covered_flags, \
                                 all_coverage_similarity_scores_target, all_similarity_indicators_target, all_distinct_flags_target, all_new_noun_covered_flags_target
                    else :
                        return all_logit_outs, all_step_ids, all_coverage_similarity_scores, all_similarity_indicators, all_distinct_flags, all_new_noun_covered_flags

        else:
            
            # inference
            # output is not logits but the final sequence
            # all sequences are scored in a list [ [ [s1,s2,....], hidden_cell, hidden_cell_env, score ] , ...... ]
            # the not executable hyothesis are ruled out and not carried forward

            all_hypothesis = []
            all_hypothesis_indices = {}
            

            for b in range(batch_size):
                
                th = hidden_cell[0][:,b,:].unsqueeze(1)
                tc = hidden_cell[1][:,b,:].unsqueeze(1)
                
                thc = (th,tc)
                
                all_hypothesis.append(Hypothesis_env(id_list=[int(self.action_object_pair2id["<bop>"])],
                                                    last_hidden_cell = thc, 
                                                    last_hidden_cell_env = None, 
                                                    last_hidden_cell_combined = th,
                                                    attn_alpha_list = [] , 
                                                    score = 0, 
                                                    id_in_batch = b,
                                                    beginning_room_entities=None,
                                                    preconds_file_path = preconds_file_path[b],
                                                    environment = environment[b]))
                all_hypothesis_indices[b] = b

            batch_pos = [0] * batch_size

            early_completed_hypothesis = []

           

            for step_id in range(max_decoder_length):
                if len(all_hypothesis) == 0:
                    break
                batch_pos = [0] * batch_size

                temp_all_hypothesis = []
                # decoder_input_list = []
                prev_hidden_list = []
                prev_cell_list = []
                prev_hidden_list_env = []
                prev_cell_list_env = []
                all_encoder_hidden_list = []
                mask_tensor_list = []
                all_noun_phrase_list = []
                mask_noun_phrase_list = []
                last_id = None
                prev_env = None
                prev_mask = None

                
                prev_hidden_list_combined = []

                if step_id == 0:
                        decoder_input = self.v[int(self.action_object_pair2id["<bop>"])].detach().view(1,1,self.embedding_size)
                        decoder_input = decoder_input.repeat(len(all_hypothesis),1,1)  # for inference mostly batch size will be 1 due to gpu memeory contraints with iher beam size
                        
                else:
                    last_id = [(x.id_list)[-1] for x in all_hypothesis] 
                    
                    decoder_input  = self.v[last_id].cuda()
                    decoder_input = decoder_input.view(len(all_hypothesis), 1, self.config.embedding_dim_decoder)

                for hyp_idx in range(len(all_hypothesis)):
                    prev_hidden = all_hypothesis[hyp_idx].last_hidden_cell[0]
                    prev_cell = all_hypothesis[hyp_idx].last_hidden_cell[1]
                    prev_hidden = prev_hidden.squeeze(0).unsqueeze(1)
                    prev_cell = prev_cell.squeeze(0).unsqueeze(1)
                    prev_hidden_list.append(prev_hidden)
                    prev_cell_list.append(prev_cell)

                   
                    all_encoder_hidden_list.append(all_encoder_hidden[all_hypothesis[hyp_idx].id_in_batch])
                    mask_tensor_list.append(mask_tensor[all_hypothesis[hyp_idx].id_in_batch])

                    all_noun_phrase_list.append(padded_noun_phrase[all_hypothesis[hyp_idx].id_in_batch])
                    mask_noun_phrase_list.append(mask_noun_phrase[all_hypothesis[hyp_idx].id_in_batch])

                
                prev_hidden_list = torch.stack(prev_hidden_list, dim=1).squeeze(2)
                prev_cell_list = torch.stack(prev_cell_list, dim=1).squeeze(2)
                prev_hidden_list_t = prev_hidden_list.transpose(0,1)

                
                all_encoder_hidden_list = torch.stack(all_encoder_hidden_list, dim=0)
                mask_tensor_list = torch.stack(mask_tensor_list, dim=0)

                all_noun_phrase_list = torch.stack(all_noun_phrase_list, dim=0)
                mask_noun_phrase_list = torch.stack(mask_noun_phrase_list, dim=0)
                
                
               
                # else:
                decoder_input = decoder_input.squeeze(1)
                hidden_cell = (prev_hidden_list, prev_cell_list)
                

                x_att, alpha = self.get_context_vector(prev_hidden_list_t, decoder_input, all_encoder_hidden_list, mask_tensor_list)
                
                decoder_input_text = torch.cat((decoder_input.squeeze(1),x_att), dim = 1)
                
                decoder_input_text = decoder_input_text.unsqueeze(1)
                

                if self.rnn_type == "lstm":
                    dec_out_text, hidden_cell = self.rnn(decoder_input_text, hidden_cell)
                    
                    
                else :
                    dec_out_text, hidden = self.rnn(decoder_input_text, hidden_cell[0])
                    hidden_cell = (hidden, hidden_cell[1])
                    
            
                
                

                prev_hidden = dec_out_text
                p = self.get_logits(dec_out_text, x_att)
                

                p = F.log_softmax(p,dim=1)
                
                # beam
                score, indices = p.topk(beam_size)
                
                switching_prob_np = np.zeros((len(all_hypothesis)))
                candidate_all_hypothesis = []
                c2a_map = {}
                c_count= 0
                for hyp_idx in range(len(all_hypothesis)):    
                    prefix_steps = all_hypothesis[hyp_idx].id_list
                    hidden_cell_hyp = (hidden_cell[0][:,hyp_idx,:].unsqueeze(1) , hidden_cell[1][:,hyp_idx,:].unsqueeze(1))
                    hidden_cell_hyp_env = hidden_cell_hyp
                    prefix_alpha = all_hypothesis[hyp_idx].attn_alpha_list
                    prefix_score = all_hypothesis[hyp_idx].score

                   
                    
                    id_in_batch = all_hypothesis[hyp_idx].id_in_batch

                    nothing_executed = True
                    for beam_idx in range(beam_size):
                        new_prefix_steps = prefix_steps.copy()
                        new_prefix_steps.append(indices[hyp_idx][beam_idx].item())
                        
                        new_prefix_score = float(score[hyp_idx][beam_idx].item())  + prefix_score
                        
                        
                        new_prefix_alpha = prefix_alpha.copy()
                        new_prefix_alpha.append(alpha[hyp_idx].detach())

                        
                        preconds_file_path_new = all_hypothesis[hyp_idx].preconds_file_path
                        environment_new = all_hypothesis[hyp_idx].environment
                        
                        candidate_all_hypothesis.append(Hypothesis_env( id_list = new_prefix_steps, 
                                                                                    last_hidden_cell = hidden_cell_hyp, 
                                                                                    last_hidden_cell_env=None,
                                                                                    last_hidden_cell_combined=None,
                                                                                    attn_alpha_list = new_prefix_alpha, 
                                                                                    score = new_prefix_score, 
                                                                                    id_in_batch = id_in_batch,
                                                                                    preconds_file_path = preconds_file_path_new,
                                                                                    environment = environment_new 
                                                                                    ))
                        
                        c2a_map[c_count] = hyp_idx
                        c_count += 1



                # collect execution flags
                
                num_cores = multiprocessing.cpu_count()
                candidate_flags_nodes_tuple_list = Parallel(n_jobs=num_cores)(delayed(execution_check_v1)(candidate_all_hypothesis[i].id_list[1:], candidate_all_hypothesis[i].preconds_file_path, candidate_all_hypothesis[i].environment, self.id2action_object_pair) for i in range(len(candidate_all_hypothesis))) 
               
                curr_bid_considered= 0
                nothing_executed = True
                for hyp_idx in range(len(candidate_all_hypothesis)):
                    
                    if candidate_flags_nodes_tuple_list[hyp_idx][0] :
                    # if True: # without caring about executability
                        print("Going here")
                        nothing_executed = False
                        
                        if int(candidate_all_hypothesis[hyp_idx].id_list[-1]) == int(self.action_object_pair2id["<eop>"]):
                            early_completed_hypothesis.append(candidate_all_hypothesis[hyp_idx])
                            print("EC insertion")
                        else:    
                            temp_all_hypothesis.append(candidate_all_hypothesis[hyp_idx])
                            batch_pos[candidate_all_hypothesis[hyp_idx].id_in_batch] += 1
                
                    next_c2a = int(c2a_map[hyp_idx + 1]) if hyp_idx <= len(candidate_all_hypothesis) - 2 else -1
                    if (next_c2a != curr_bid_considered): 
                        if nothing_executed :
                            
                            # if none in beam was executable append this hypothesis as already completed  
                            old_hypothesis = all_hypothesis[c2a_map[hyp_idx]]
                            new_prefix_steps = old_hypothesis.id_list.copy() 
                            new_prefix_steps.append(int(self.action_object_pair2id["<eop>"]))
                            prefix_score = old_hypothesis.score
                            new_prefix_score = float(p[c2a_map[hyp_idx]][int(self.action_object_pair2id["<eop>"])].item())  + prefix_score
                            prefix_alpha = old_hypothesis.attn_alpha_list
                            new_prefix_alpha = prefix_alpha.copy()
                            new_prefix_alpha.append(alpha[c2a_map[hyp_idx]].detach())
                            early_completed_hypothesis.append(Hypothesis_env(id_list = new_prefix_steps, 
                                                                                last_hidden_cell = candidate_all_hypothesis[hyp_idx].last_hidden_cell, 
                                                                                last_hidden_cell_env = None,
                                                                                last_hidden_cell_combined = None,
                                                                                attn_alpha_list = new_prefix_alpha, 
                                                                                score = new_prefix_score,  
                                                                                id_in_batch = old_hypothesis.id_in_batch, 
                                                                                ))
                        else:
                            nothing_executed = True
                            curr_bid_considered += 1


                all_hypothesis = []
                start_id = 0
                end_id = 0
                for bid in range(batch_size):
                    end_id = start_id + batch_pos[bid]
                    tah = temp_all_hypothesis[start_id:end_id]
                    start_id = end_id
                    if len(tah) > 0:
                        tah = sorted(tah, key = lambda x: float(x.score)/(len(x.id_list)) )
                        try : 
                            if int((tah[-1].id_list)[-1]) == int(self.action_object_pair2id["<eop>"]):
                                break
                        except :
                            
                            import sys
                            sys.exit()
                            
                        if len(tah) > beam_size:
                            tah = tah[-beam_size:]
                    else:
                        tah = []
                        print("No hypothesis left in beam")
                        
                    
                    for hyp in tah:
                        # print(hyp)
                        all_hypothesis.append(hyp)
                
            ret_seq_full_batch = []
            alpha_seq_full_batch = []
            sp_full_batch = []
            all_hypothesis_specific_batch = []
            

            for bid in range(batch_size):
                # get early hypothesis
                all_hypothesis_specific_batch = []
                for e in early_completed_hypothesis:
                    if e.id_in_batch == bid:
                        all_hypothesis_specific_batch.append(e)
                
                
                for e in all_hypothesis:
                    if e.id_in_batch == bid:
                        all_hypothesis_specific_batch.append(e)

                #choose top hypothesis
                print("candidates")
                print(len(early_completed_hypothesis))
                if not len(all_hypothesis_specific_batch) > 0:
                    print("failing batch id = ", bid)
                assert len(all_hypothesis_specific_batch) > 0
            
                
                sorted_hypothesis_specific_batch = sorted(all_hypothesis_specific_batch, key = lambda x: float(x.score)/(len(x.id_list)) )
            
                ret_seq = sorted_hypothesis_specific_batch[-1].id_list.copy()
                
                if int(ret_seq[-1]) != int(self.action_object_pair2id["<eop>"]):
                    
                    ret_seq.append(self.action_object_pair2id["<eop>"])
                
                try :
                    ret_seq = torch.from_numpy(np.array(ret_seq))
                except:
                    
                    import sys
                    sys.exit()

                if torch.cuda.is_available():
                    ret_seq = ret_seq.cuda()
                

                ret_seq_full_batch.append(ret_seq)

                sp_list = torch.from_numpy(np.array(sorted_hypothesis_specific_batch[-1].switching_prob_list.copy())).cuda()
                sp_full_batch.append(sp_list)
                


            return ret_seq_full_batch, sp_full_batch



    def initHidden(self,batch_size):
        return torch.zeros(batch_size, 1, 300, device=device)
    
    def initHiddenTest(self):
        return torch.zeros(1, 1, 300, device=device)