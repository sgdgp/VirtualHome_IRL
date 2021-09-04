
from model import *

import time
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import re
from tqdm import tqdm
from metrics import *
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.nn.util import sequence_cross_entropy_with_logits
from tensorboardX import SummaryWriter
import gc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from test_model import *
from rewards import *
from sklearn import svm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, dataloader, config):
        self.config = config

        hidden_size_encoder = 100
        hidden_size_decoder = 100
        self.bidirectional_rnn_encoder = False
        if self.bidirectional_rnn_encoder :
            hidden_size_decoder *= 2
        embedding_size = 300
        self.encoder = EncoderRNN_with_history(hidden_size = hidden_size_encoder, 
                                    rnn_type="lstm",
                                    rnn_numlayers = 1,
                                    bidirectional_rnn = self.bidirectional_rnn_encoder,
                                    data=dataloader)
        self.encoder = self.encoder.to(device)
        self.decoder = AttnDecoderRNN_with_history(hidden_size_decoder, rnn_type="lstm",data=dataloader, config=config, encoder_hidden_size=hidden_size_encoder)
        self.decoder = self.decoder.to(device)

        if self.config.data_parallel:
            self.encoder = DP(self.encoder)
            self.decoder = DP(self.decoder)

        self.data = dataloader

        

        self.written_encoder_graph = False
        self.written_decoder_graph = False

        self.L2_loss_criterion = nn.MSELoss(reduction='none')

        if os.path.exists(self.config.encoder_model_path_load) and self.config.enable_model_load :
            if self.config.data_parallel : 
                self.encoder.module.load_state_dict(torch.load(self.config.encoder_model_path_load), strict=False)
                self.decoder.module.load_state_dict(torch.load(self.config.decoder_model_path_load), strict=False)
            else:
                self.encoder.load_state_dict(torch.load(self.config.encoder_model_path_load), strict=False)
                self.decoder.load_state_dict(torch.load(self.config.decoder_model_path_load), strict=False)
            print("Model loaded")
            




    def train_and_save(self, num_iters):       
        self.trainIters(num_iters)
        

        


    def train(self,input_tensor, env_tensor_entites, env_tensor_beginning_room,  noun_phrase, target_tensor, writer):
        batch_size = len(input_tensor)
        
        encoder_hidden = None
        encoder_cell = None
        encoder_output = None

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        
        loss = 0
        
        seq_lengths = torch.tensor([len(seq) for seq in input_tensor], dtype=torch.long, device=device)
        
        max_length = seq_lengths.max().item()
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        _, reverse_perm_idx = perm_idx.sort(0)


        
             
        # del seq_lengths
        word2id = json.load(open("./disjoint_data/word2id.json", "r"))
        id2word = json.load(open("./disjoint_data/id2word.json", "r"))
        id2word = {int(k) : str(v) for k,v in id2word.items()}

        padded_input_tensor = []
        mask_tensor = torch.ones((batch_size, max_length), device=device)
        
        for i in range(batch_size):
        # for sample in input_tensor
            padded_sample = input_tensor[i].copy()
            for j in range(len(input_tensor[i])):
                if int(padded_sample[j]) == int(word2id["<bos>"]) or int(padded_sample[j]) == int(word2id["<eos>"]):
                    mask_tensor[i][j] = 0.0 
            for j in range(max_length - len(input_tensor[i])):
                padded_sample.append(word2id["<pad>"])
                # print(len(input_tensor[i]) + j)
                mask_tensor[i][len(input_tensor[i]) + j] = 0.0
            
            padded_input_tensor.append(torch.tensor(padded_sample, dtype=torch.long, device=device))
        
        padded_input_tensor = torch.stack(padded_input_tensor)
        
        
        
        # noun phrase
        mask_noun_phrase = np.zeros((batch_size, 19))
        padded_noun_phrase = np.zeros((batch_size, 19))
        if self.config.np_context:
            for i in range(batch_size):
                np_sample = noun_phrase[i]
                for k in range(len(np_sample)):
                    mask_noun_phrase[i][k] = 1        
                    padded_noun_phrase[i][k] = np_sample[k]
        
        padded_noun_phrase = torch.from_numpy(padded_noun_phrase).long().cuda()
        mask_noun_phrase = torch.from_numpy(mask_noun_phrase).float().cuda()
        # pad target tensor as well
        
        target_seq_lengths = torch.tensor([len(seq) for seq in self.target_np], dtype=torch.long, device=device)
        target_max_length = target_seq_lengths.max().item()

        target_max_length -= 1
        # target_max_length = max(target_max_length, 10)
        # print("Target max length = ", target_max_length)

        padded_target_tensor = []
        padded_target_tensor_decoder_input = []
        weight_tensor = np.ones((batch_size, target_max_length))

        
        padded_env_tensor_entites = []

        sp_mask = np.ones((batch_size, target_max_length-2))
        padded_target_sp_tensor = []

        for i in range(batch_size):
            padded_env_tensor_entites_sample = env_tensor_entites[i].copy()
            temp_env_len = len(padded_env_tensor_entites_sample)
            for _ in range(target_max_length - temp_env_len):
                padded_env_tensor_entites_sample.append([0] * len(padded_env_tensor_entites_sample[0]))

            padded_env_tensor_entites.append(padded_env_tensor_entites_sample)

            padded_target = self.target_np[i].copy()
            padded_target = padded_target[1:]
            padded_target_input = np.zeros((target_max_length, self.config.embedding_dim_decoder))
            temp_len = len(padded_target)
            
            
            padded_target_sp_sample= self.target_sp_tensor[i][1:].copy()
            
            for j in range(target_max_length - temp_len):
                padded_target.append(0)
                weight_tensor[i][temp_len + j] = 0.0
                
                padded_target_sp_sample.append(0)
                sp_mask[i][j] = 0.0
            
            # padded_target_input[0] = self.data.bop_embedding
            temp_len = len(self.target_np[i]) -1
            
            for j in range(temp_len):
                vv = None
                if self.config.data_parallel:
                    vv = self.decoder.module.v.cpu().detach().numpy()
                else:
                    vv = self.decoder.v.cpu().detach().numpy()
                
                padded_target_input[j] = vv[self.target_np[i][j]]
                
            padded_target_tensor.append(torch.tensor(padded_target, dtype=torch.long, device=device)) 
            padded_target_tensor_decoder_input.append(torch.tensor(padded_target_input, dtype=torch.float, device=device)) 

            padded_target_sp_tensor.append(torch.tensor(padded_target_sp_sample, dtype=torch.float, device=device))
        
        padded_target_tensor = torch.stack(padded_target_tensor)
        padded_target_tensor_decoder_input = torch.stack(padded_target_tensor_decoder_input)
        weight_tensor = torch.from_numpy(weight_tensor)

        padded_target_sp_tensor= torch.stack(padded_target_sp_tensor)
        sp_mask = torch.from_numpy(sp_mask)
        
        env_mask_entities = torch.from_numpy(np.array(padded_env_tensor_entites)).cuda()
        env_mask_beginning_room = torch.from_numpy(np.array(env_tensor_beginning_room)).cuda()

        noun_tensor = torch.from_numpy(np.array(self.nouns)).cuda()
        # print(noun_tensor[0])
        # print("noun_tensor size = ", noun_tensor.size())
        

        if device == torch.device("cuda"):
            weight_tensor = weight_tensor.cuda()
            padded_target_tensor = padded_target_tensor.cuda()
            padded_target_tensor_decoder_input = padded_target_tensor_decoder_input.cuda()

            padded_target_sp_tensor = padded_target_sp_tensor.cuda()
            sp_mask = sp_mask.cuda()
            
            
           
        ################## ENCODER ###################################################
        # sort all input
        padded_input_tensor = padded_input_tensor[perm_idx]
        encoder_hidden = None
        encoder_cell = None
        if self.config.data_parallel:
            encoder_hidden = self.encoder.module.initHidden(batch_size)
            encoder_cell = self.encoder.module.initHidden(batch_size)
        else:
            encoder_hidden = self.encoder.initHidden(batch_size)
            encoder_cell = self.encoder.initHidden(batch_size)
            
        encoder_hidden = encoder_hidden.transpose(0,1)
        encoder_cell = encoder_cell.transpose(0,1)
        
        # contiguous states
        encoder_hidden = encoder_hidden.contiguous()
        encoder_cell = encoder_cell.contiguous()
        # get all hidden states
        all_encoder_hidden, (encoder_hidden, encoder_cell),ret_seq_length, self_attended_output = self.encoder(padded_input_tensor, seq_lengths.view(batch_size, -1), (encoder_hidden, encoder_cell))
        
        #unsort input and hidden state
        
        padded_input_tensor = padded_input_tensor[reverse_perm_idx]
        all_encoder_hidden = all_encoder_hidden[reverse_perm_idx]
        encoder_hidden = encoder_hidden[reverse_perm_idx]
        encoder_cell = encoder_cell[reverse_perm_idx]
        self_attended_output = self_attended_output[reverse_perm_idx]

        ret_seq_length = ret_seq_length.view(batch_size)
        ret_seq_length  = ret_seq_length[reverse_perm_idx]
        seq_lengths = seq_lengths[reverse_perm_idx]

        
        ################## DECODER ###################################################
        # the packing is done in order of decoder input tensors if teacher forcing is enabled
        # sort the decoder input sequences beforehand in decreasing length (only for teacher forcing this is important)
        # then using the same idx sort encoder hidden, encoder cell, all encoder hidden, weight tensors
        target_seq_lengths, target_perm_idx = target_seq_lengths.sort(0, descending=True)
        _, reverse_target_perm_idx = target_perm_idx.sort(0)
        
        padded_target_tensor_decoder_input = padded_target_tensor_decoder_input[target_perm_idx]
        padded_target_tensor = padded_target_tensor[target_perm_idx]
        weight_tensor = weight_tensor[target_perm_idx]
        all_encoder_hidden = all_encoder_hidden[target_perm_idx]
        encoder_cell = encoder_cell[target_perm_idx]   # batch_size x 1 x hidden_dim
        encoder_hidden = encoder_hidden[target_perm_idx]  # batch_size x 1 x hidden_dim
        mask_tensor = mask_tensor[target_perm_idx]
        padded_input_tensor = padded_input_tensor[target_perm_idx]
        self_attended_output = self_attended_output[target_perm_idx]
        

        env_mask_entities = env_mask_entities[target_perm_idx]
        env_mask_beginning_room = env_mask_beginning_room[target_perm_idx]
        noun_tensor = noun_tensor[target_perm_idx]
        padded_noun_phrase = padded_noun_phrase[target_perm_idx]
        mask_noun_phrase = mask_noun_phrase[target_perm_idx]
        # print(type(self.environment))
        
        self.environment = [self.environment[i] for i in target_perm_idx ]
        self.preconds_file_path = [self.preconds_file_path[i] for i in target_perm_idx ]
        
        padded_target_sp_tensor = padded_target_sp_tensor[target_perm_idx]
        sp_mask = sp_mask[target_perm_idx]

        if self.bidirectional_rnn_encoder: # concat if encoder was bidiectional
            encoder_hidden = encoder_hidden.view(encoder_hidden.size(0), 1 , -1)
            encoder_cell = encoder_cell.view(encoder_cell.size(0), 1 , -1)


        self_attended_output = self_attended_output.view(self_attended_output.size(0),1,self_attended_output.size(1))
        
        (decoder_hidden, decoder_cell) = (encoder_hidden.contiguous(), encoder_cell.contiguous())
        
        

        # (decoder_hidden, decoder_cell) = (self_attended_output.contiguous(), encoder_cell.contiguous())
        
        # remember to change the shapes of decoder_cell and decoder_hidden inside decoder forward appropriately, 
        # not changed here to have suitable scatter in DataParallel
        
        if self.config.training_procedure == "CE":
            # print("CE")
            use_teacher_forcing = True if random.random() < self.config.teacher_forcing_ratio  else False
            if int(self.config.teacher_forcing_ratio) == 1:
                use_teacher_forcing = True 
            # use_teacher_forcing = True
            use_teacher_forcing_tensor = None
            if use_teacher_forcing == True :
                use_teacher_forcing_tensor = torch.ones((batch_size, 1), dtype=torch.uint8,device=device)
            else :
                use_teacher_forcing_tensor = torch.zeros((batch_size, 1), dtype=torch.uint8, device=device)
            
            if use_teacher_forcing:
                max_decoder_length = target_max_length
            else:
                max_decoder_length = min(20, target_max_length)
     

            
            logits, all_sp_out = self.decoder(input = padded_target_tensor_decoder_input,
                        env_entities =  env_mask_entities, 
                        beginning_room_entities = env_mask_beginning_room,
                        noun_ids = noun_tensor,
                        padded_noun_phrase = padded_noun_phrase,
                        mask_noun_phrase = mask_noun_phrase,
                        hidden_cell = (decoder_hidden, decoder_cell), 
                        hidden_cell_env = (None, None), 
                        all_encoder_hidden = all_encoder_hidden, 
                        seq_lengths = target_seq_lengths.view(batch_size, -1), 
                        mask_tensor = mask_tensor.view(batch_size, -1), 
                        use_teacher_forcing_tensor = use_teacher_forcing_tensor,
                        max_decoder_length = max_decoder_length,
                        training_mode = True,
                        beam_size = 1)
            
            
            
            num_steps = target_max_length
            if not use_teacher_forcing :
                num_steps = max_decoder_length
                weight_tensor = weight_tensor.detach()[:, :num_steps]
                padded_target_tensor = padded_target_tensor.detach()[:, :num_steps]
                

            
            padded_target_tensor = padded_target_tensor.contiguous()
            weight_tensor = weight_tensor.contiguous()
            loss = sequence_cross_entropy_with_logits(logits,padded_target_tensor.view(batch_size,num_steps), weight_tensor.view(batch_size,num_steps), gamma=3.5)

            
            # backward
            loss.backward()
            loss_val = loss.item()
            
        elif self.config.training_procedure=="SCST":
            # print("SCST")
            # teacher forcing set to false for SCST
            use_teacher_forcing = False
            use_teacher_forcing_tensor = None
            if use_teacher_forcing == True :
                use_teacher_forcing_tensor = torch.ones((batch_size, 1), dtype=torch.uint8,device=device)
            else :
                use_teacher_forcing_tensor = torch.zeros((batch_size, 1), dtype=torch.uint8, device=device)
            
            max_decoder_length = 20
            # max_decoder_length = min(20, target_max_length)
            beam_size = 1 # Training
            

                    
            # logits_sample, step_ids_sample
            logits_sample, step_ids_sample, coverage_reward_sample, similarity_indicator_sample, distinct_flags_sample,new_nouns_covered_flag_sample, \
                coverage_reward_target, similarity_indicator_target, distinct_flags_target, new_nouns_covered_flag_target \
                    = self.decoder(input = padded_target_tensor,
                        env_entities =  env_mask_entities, 
                        beginning_room_entities = env_mask_beginning_room,
                        noun_ids = noun_tensor,
                        padded_noun_phrase = padded_noun_phrase,
                        mask_noun_phrase = mask_noun_phrase,
                        hidden_cell = (decoder_hidden, decoder_cell), 
                        hidden_cell_env = (None, None), 
                        all_encoder_hidden = all_encoder_hidden, 
                        seq_lengths = target_seq_lengths.view(batch_size, -1), 
                        mask_tensor = mask_tensor.view(batch_size, -1), 
                        use_teacher_forcing_tensor = use_teacher_forcing_tensor,
                        max_decoder_length = max_decoder_length,
                        training_mode = True,
                        beam_size = 1,
                        get_sampled_sequence=True)

            # create mask for sampled sequence
            step_ids_sample_np = step_ids_sample.detach().cpu().numpy()
            print("here step_ids_sample_np = ",step_ids_sample_np.shape)
            mask_sample = torch.zeros(step_ids_sample_np.shape[0], max_decoder_length).cuda()

            # print("Creating mask for sample : ")
            for i_b in range(step_ids_sample_np.shape[0]):
                ll = -1
                # print(step_ids_sample_np[i_b])
                for jj in range(max_decoder_length):
                    # print(int(step_ids_sample_np[i_b][jj]))
                    if int(step_ids_sample_np[i_b][jj]) == int(self.decoder.module.action_object_pair2id["<eop>"]):
                        ll = jj+1 
                        break
                if ll == -1:
                    ll = max_decoder_length
                for k in range(ll):
                    mask_sample[i_b][k] = 1.0
                

            mask_target = torch.zeros(padded_target_tensor.size(0), max_decoder_length).cuda()

            # print("Creating mask for sample : ")
            for i_b in range(padded_target_tensor.size(0)):
                ll = -1
                # print(step_ids_sample_np[i_b])
                for jj in range(max_decoder_length):
                    # print(int(step_ids_sample_np[i_b][jj]))
                    if int(padded_target_tensor[i_b][jj]) == int(self.decoder.module.action_object_pair2id["<eop>"]):
                        ll = jj+1 
                        break
                if ll == -1:
                    ll = max_decoder_length
                for k in range(ll):
                    mask_target[i_b][k] = 1.0

            

            # logits_greedy, step_ids_greedy
            logits_greedy, step_ids_greedy, coverage_reward_greedy, similarity_indicator_greedy, \
                distinct_flags_greedy, new_nouns_covered_flag_greedy = self.decoder(input = padded_target_tensor,
                        env_entities =  env_mask_entities, 
                        beginning_room_entities = env_mask_beginning_room,
                        noun_ids = noun_tensor,
                        padded_noun_phrase = padded_noun_phrase,
                        mask_noun_phrase = mask_noun_phrase,
                        hidden_cell = (decoder_hidden, decoder_cell), 
                        hidden_cell_env = (None, None), 
                        all_encoder_hidden = all_encoder_hidden, 
                        seq_lengths = target_seq_lengths.view(batch_size, -1), 
                        mask_tensor = mask_tensor.view(batch_size, -1), 
                        use_teacher_forcing_tensor = use_teacher_forcing_tensor,
                        max_decoder_length = max_decoder_length,
                        training_mode = True,
                        beam_size = 1)

            # create mask for greedy sequence
            step_ids_greedy_np = step_ids_greedy.detach().cpu().numpy()
            mask_greedy = torch.zeros(step_ids_greedy_np.shape[0], max_decoder_length).cuda()
            for i_b in range(step_ids_greedy_np.shape[0]):
                ll = -1
                for jj in range(max_decoder_length):
                    if int(step_ids_greedy_np[i_b][jj]) == int(self.decoder.module.action_object_pair2id["<eop>"]):
                        ll = jj+1 
                        break
                if ll == -1:
                    ll = max_decoder_length
                for k in range(ll):
                    mask_greedy[i_b][k] = 1.0

            # get masked logits, step_ids
            masked_step_ids_sample_np = (step_ids_sample * mask_sample).detach().cpu().numpy()
            masked_step_ids_greedy_np = (step_ids_greedy * mask_greedy).detach().cpu().numpy()


            

            
            # reward LCS calculation
            sample_score_LCS = LCS_batch_train(masked_step_ids_sample_np, padded_target_tensor.detach().cpu().numpy()[:,:max_decoder_length] , 
                                                mask_sample.detach().cpu().numpy(), weight_tensor.detach().cpu().numpy())

            greedy_score_LCS = LCS_batch_train(masked_step_ids_greedy_np,padded_target_tensor.detach().cpu().numpy()[:,:max_decoder_length] , 
                                                mask_greedy.detach().cpu().numpy(), weight_tensor.detach().cpu().numpy())

            reward_LCS_total = sample_score_LCS - greedy_score_LCS # this is not per step
            
            reward_LCS_stepped = torch.zeros(batch_size,max_decoder_length).cuda()
            reward_LCS_sample = torch.zeros(batch_size,max_decoder_length).cuda()
            reward_LCS_expert = torch.zeros(batch_size,max_decoder_length).cuda()
            reward_LCS_greedy = torch.zeros(batch_size,max_decoder_length).cuda()
            
            
            for bb in range(batch_size):
                tt = mask_sample[bb]
                tt = int(tt.sum())
                tt -= 1
                reward_LCS_sample[bb][tt] = sample_score_LCS[bb]
                # reward_LCS_stepped[bb][tt] = reward_LCS_total[bb]

                tt = mask_greedy[bb]
                tt = int(tt.sum())
                tt -= 1
                reward_LCS_greedy[bb][tt] = greedy_score_LCS[bb]

                tt = mask_target[bb]
                tt = int(tt.sum())
                tt -= 1
                reward_LCS_expert[bb][tt] = 1
            

            for bb in range(batch_size): 
                for tt in range(max_decoder_length):
                    reward_LCS_stepped[bb][tt] = reward_LCS_total[bb]
                    
            
            

            # calculate all rewards
            
            # calculate recall reward
            if "rec_diff" in self.config.list_of_rewards :
                recall_reward_sample = torch.from_numpy(recall_diff_reward(similarity_indicator_sample,noun_tensor, distinct_flags_sample, new_nouns_covered_flag_sample )).cuda()
                recall_reward_greedy = torch.from_numpy(recall_diff_reward(similarity_indicator_greedy,noun_tensor, distinct_flags_greedy,new_nouns_covered_flag_greedy )).cuda()
                recall_reward_target = torch.from_numpy(recall_diff_reward(similarity_indicator_target,noun_tensor, distinct_flags_target, new_nouns_covered_flag_target )).cuda()            
                masked_recall_reward_sample = mask_sample * recall_reward_sample
                masked_recall_reward_greedy = mask_greedy * recall_reward_greedy
                masked_recall_reward_target = mask_target * recall_reward_target

                recall_reward = masked_recall_reward_sample - masked_recall_reward_greedy

            # recall_reward = torch.from_numpy(recall_reward).cuda()

            # calculate precision reward
            if "prec" in self.config.list_of_rewards:
                prec_reward_sample = torch.from_numpy(precision_reward_new(similarity_indicator_sample,noun_tensor, distinct_flags_sample, mask_sample )).cuda()
                prec_reward_greedy = torch.from_numpy(precision_reward_new(similarity_indicator_greedy,noun_tensor, distinct_flags_greedy, mask_greedy )).cuda()
                prec_reward_target = torch.from_numpy(precision_reward_new(similarity_indicator_target,noun_tensor, distinct_flags_target, mask_target )).cuda()
                masked_prec_reward_sample = mask_sample * prec_reward_sample
                masked_prec_reward_greedy = mask_greedy * prec_reward_greedy
                masked_prec_reward_target = mask_target * prec_reward_target
                prec_reward = masked_prec_reward_sample - masked_prec_reward_greedy
            # prec_reward = torch.from_numpy(prec_reward).cuda()

            # calculate repition reward
            if "rep" in self.config.list_of_rewards:
                rep_reward_sample = torch.from_numpy(repition_reward(step_ids_sample_np, self.decoder.module.id2action_object_pair)).cuda()
                rep_reward_greedy =  torch.from_numpy(repition_reward(step_ids_greedy_np, self.decoder.module.id2action_object_pair)).cuda()
                rep_reward_target =  torch.from_numpy(repition_reward(padded_target_tensor.clone().detach().cpu().numpy()[:, :max_decoder_length], self.decoder.module.id2action_object_pair)).cuda()
                masked_rep_reward_sample = mask_sample * rep_reward_sample
                masked_rep_reward_greedy = mask_greedy * rep_reward_greedy
                masked_rep_reward_target = mask_target * rep_reward_target
                rep_reward = masked_rep_reward_sample - masked_rep_reward_greedy
            # reptition_reward = torch.from_numpy(reptition_reward).cuda()

            # recall_from_expert_program(pred_steps, mask_pred, expert_steps, mask_expert)
            # calculate recall from expert program
            if "rec_prog" in self.config.list_of_rewards:
                recall_prog_reward_sample = torch.from_numpy(recall_from_expert_program(step_ids_sample, mask_sample, padded_target_tensor.clone()[:, :max_decoder_length], mask_target)).cuda()
                recall_prog_reward_greedy = torch.from_numpy(recall_from_expert_program(step_ids_greedy, mask_greedy, padded_target_tensor.clone()[:, :max_decoder_length], mask_target)).cuda()
                recall_prog_reward_target = torch.from_numpy(recall_from_expert_program(padded_target_tensor.clone()[:, :max_decoder_length], mask_target, padded_target_tensor.clone()[:, :max_decoder_length], mask_target)).cuda()
                masked_recall_prog_reward_sample = mask_sample * recall_prog_reward_sample
                masked_recall_prog_reward_greedy = mask_greedy * recall_prog_reward_greedy
                masked_recall_prog_reward_target = mask_target * recall_prog_reward_target
                recall_prog_reward = masked_recall_prog_reward_sample - masked_recall_prog_reward_greedy
                

            # create pg versions of the rewards
            if "prec" in self.config.list_of_rewards:            
                reward_prec_stepped = torch.zeros((batch_size)).cuda()
            if "rec_diff" in self.config.list_of_rewards:
                reward_recall_stepped = torch.zeros((batch_size)).cuda()
            if "rep" in self.config.list_of_rewards:
                reward_rep_stepped = torch.zeros((batch_size)).cuda()
            if "rec_prog" in self.config.list_of_rewards:
                reward_recall_prog_stepped = torch.zeros((batch_size)).cuda()
            
            for bb in range(batch_size):
                if "prec" in self.config.list_of_rewards:
                    reward_prec_stepped[bb] = masked_prec_reward_sample[bb].sum() - masked_prec_reward_greedy[bb].sum()
                if "rec_diff" in self.config.list_of_rewards:
                    reward_recall_stepped[bb] = masked_recall_reward_sample[bb].sum() - masked_recall_reward_greedy[bb].sum()
                if "rec_prog" in self.config.list_of_rewards:
                    reward_recall_prog_stepped[bb] = masked_recall_prog_reward_sample[bb].sum() - masked_recall_prog_reward_greedy[bb].sum()
                if "rep" in self.config.list_of_rewards:
                    reward_rep_stepped[bb] = masked_rep_reward_sample[bb].sum() - masked_rep_reward_greedy[bb].sum()
            
            if "prec" in self.config.list_of_rewards:
                reward_prec_stepped = reward_prec_stepped.view(-1,1).repeat(1, max_decoder_length)
            if "rec_diff" in self.config.list_of_rewards:
                reward_recall_stepped = reward_recall_stepped.view(-1,1).repeat(1, max_decoder_length)
            if "rec_prog" in self.config.list_of_rewards:
                reward_recall_prog_stepped = reward_recall_prog_stepped.view(-1,1).repeat(1, max_decoder_length)
            if "rep" in self.config.list_of_rewards:
                reward_rep_stepped = reward_rep_stepped.view(-1,1).repeat(1, max_decoder_length)
            
            
            # exec reward
            if "exec" in self.config.list_of_rewards:
                num_cores = multiprocessing.cpu_count()
                sample_candidate_flags_nodes_tuple_list = Parallel(n_jobs=num_cores)(delayed(execution_check_v1)(step_ids_sample_np[i, :int(mask_sample[i].sum())], self.preconds_file_path[i], self.environment[i], self.decoder.module.id2action_object_pair) for i in range( int(step_ids_sample_np.shape[0]) )) 
                greedy_candidate_flags_nodes_tuple_list = Parallel(n_jobs=num_cores)(delayed(execution_check_v1)(step_ids_greedy_np[i, :int(mask_greedy[i].sum())], self.preconds_file_path[i], self.environment[i], self.decoder.module.id2action_object_pair) for i in range( int(step_ids_sample_np.shape[0]) )) 
                exec_reward_sample = torch.zeros(batch_size, max_decoder_length).cuda()
                exec_reward_greedy = torch.zeros(batch_size, max_decoder_length).cuda()
                exec_reward_expert = torch.zeros(batch_size, max_decoder_length).cuda()
                exec_reward_stepped = torch.zeros(batch_size, max_decoder_length).cuda()
                
                for r_id in range(int(step_ids_sample_np.shape[0])):
                    x1 = 0
                    x2 = 0
                    if sample_candidate_flags_nodes_tuple_list[r_id][0]:
                        exec_reward_sample[r_id][int(mask_sample[r_id].sum()) -1 ] = 1
                        x1 = 1 
                    
                    if greedy_candidate_flags_nodes_tuple_list[r_id][0]:
                        exec_reward_greedy[r_id][int(mask_greedy[r_id].sum()) -1 ] = 1
                        x2 = 1

                    for jj in range(max_decoder_length):
                        exec_reward_stepped[r_id][jj] = x1 - x2

                    exec_reward_expert[r_id][int(mask_target[r_id].sum()) -1 ] = 1

                masked_exec_reward_sample = mask_sample * exec_reward_sample
                masked_exec_reward_greedy = mask_greedy * exec_reward_greedy
                masked_exec_reward_target = mask_target * exec_reward_expert
                
            if "rec_diff" in self.config.list_of_rewards:
                mu_expert_recall = masked_recall_reward_target.sum(1) #/ mask_target.sum(1)      
            if "prec" in self.config.list_of_rewards:
                mu_expert_prec = masked_prec_reward_target.sum(1)# / mask_target.sum(1)      
            if "rep" in self.config.list_of_rewards:
                mu_expert_rep = masked_rep_reward_target.sum(1) #/ mask_target.sum(1)      
            if "LCS" in self.config.list_of_rewards:
                mu_expert_LCS = reward_LCS_expert.sum(1)
            if "rec_prog" in self.config.list_of_rewards:
                mu_expert_recall_prog = masked_recall_prog_reward_target.sum(1)
            if "exec" in self.config.list_of_rewards:
                mu_expert_exec = masked_exec_reward_target.sum(1)
            

            if "rec_diff" in self.config.list_of_rewards:
                mu_sample_recall = masked_recall_reward_sample.sum(1) #/ mask_sample.sum(1)      
            if "prec" in self.config.list_of_rewards:
                mu_sample_prec = masked_prec_reward_sample.sum(1) #/ mask_sample.sum(1)      
            if "rep" in self.config.list_of_rewards:
                mu_sample_rep = masked_rep_reward_sample.sum(1) #/ mask_sample.sum(1)      
            if "LCS" in self.config.list_of_rewards:
                mu_sample_LCS = reward_LCS_sample.sum(1)
            if "rec_prog" in self.config.list_of_rewards:
                mu_sample_recall_prog = masked_recall_prog_reward_sample.sum(1)
            if "exec" in self.config.list_of_rewards:
                mu_sample_exec = masked_exec_reward_sample.sum(1)
            

            if "rec_diff" in self.config.list_of_rewards:
                mu_greedy_recall = masked_recall_reward_greedy.sum(1) #/ mask_sample.sum(1)      
            if "prec" in self.config.list_of_rewards:
                mu_greedy_prec = masked_prec_reward_greedy.sum(1) #/ mask_sample.sum(1)      
            if "rep" in self.config.list_of_rewards:
                mu_greedy_rep = masked_rep_reward_greedy.sum(1) #/ mask_sample.sum(1)      
            if "LCS" in self.config.list_of_rewards:
                mu_greedy_LCS = reward_LCS_greedy.sum(1)
            if "rec_prog" in self.config.list_of_rewards:
                mu_greedy_recall_prog = masked_recall_prog_reward_greedy.sum(1)
            if "exec" in self.config.list_of_rewards:
                mu_greedy_exec = masked_exec_reward_greedy.sum(1)
            

            mu_expert = torch.zeros(self.config.num_rewards).cuda()
            if self.variance_calc:
                for r_id in range(batch_size):
                    if "prec" in self.config.list_of_rewards:
                        self.rewards_list_expert[self.config.reward_index["prec"]].append(float(mu_expert_prec[r_id]))
                    if "rec_diff" in self.config.list_of_rewards:
                        self.rewards_list_expert[self.config.reward_index["rec_diff"]].append(float(mu_expert_recall[r_id]))
                    if "rep" in self.config.list_of_rewards:
                        self.rewards_list_expert[self.config.reward_index["rep"]].append(float(mu_expert_rep[r_id]))
                    if "LCS" in self.config.list_of_rewards:
                        self.rewards_list_expert[self.config.reward_index["LCS"]].append(float(mu_expert_LCS[r_id]))
                    if "rec_prog" in self.config.list_of_rewards:
                        self.rewards_list_expert[self.config.reward_index["rec_prog"]].append(float(mu_expert_recall_prog[r_id]))
                    if "exec" in self.config.list_of_rewards:
                        self.rewards_list_expert[self.config.reward_index["exec"]].append(float(mu_expert_exec[r_id]))
                    
            
            # print("here 4 = ", type(self.rewards_list_expert[5]))
            if "prec" in self.config.list_of_rewards:
                mu_expert[self.config.reward_index["prec"]] = mu_expert_prec.sum()
            if "rec_diff" in self.config.list_of_rewards:
                mu_expert[self.config.reward_index["rec_diff"]] = mu_expert_recall.sum()
            if "rep" in self.config.list_of_rewards:
                mu_expert[self.config.reward_index["rep"]] = mu_expert_rep.sum()
            if "LCS" in self.config.list_of_rewards:
                mu_expert[self.config.reward_index["LCS"]] = mu_expert_LCS.sum()
            if "rec_prog" in self.config.list_of_rewards:
                mu_expert[self.config.reward_index["rec_prog"]] = mu_expert_recall_prog.sum()
            if "exec" in self.config.list_of_rewards:
                mu_expert[self.config.reward_index["exec"]] = mu_expert_exec.sum()
                
            
            


            mu_sample = torch.zeros(self.config.num_rewards).cuda()
            mu_greedy = torch.zeros(self.config.num_rewards).cuda()
            if self.variance_calc:
                for r_id in range(batch_size):
                    if "prec" in self.config.list_of_rewards:
                        self.rewards_list_sample[self.config.reward_index["prec"]].append(float(mu_sample_prec[r_id]))
                    if "rec_diff" in self.config.list_of_rewards:
                        self.rewards_list_sample[self.config.reward_index["rec_diff"]].append(float(mu_sample_recall[r_id]))
                    if "rep" in self.config.list_of_rewards:
                        self.rewards_list_sample[self.config.reward_index["rep"]].append(float(mu_sample_rep[r_id]))
                    if "LCS" in self.config.list_of_rewards:
                        self.rewards_list_sample[self.config.reward_index["LCS"]].append(float(mu_sample_LCS[r_id]))
                    if "rec_prog" in self.config.list_of_rewards:
                        self.rewards_list_sample[self.config.reward_index["rec_prog"]].append(float(mu_sample_recall_prog[r_id]))
                    if "exec" in self.config.list_of_rewards:
                        self.rewards_list_sample[self.config.reward_index["exec"]].append(float(mu_sample_exec[r_id]))
                    


                
                for r_id in range(batch_size):
                    if "prec" in self.config.list_of_rewards:
                        self.rewards_list_greedy[self.config.reward_index["prec"]].append(float(mu_greedy_prec[r_id]))
                    if "rec_diff" in self.config.list_of_rewards:
                        self.rewards_list_greedy[self.config.reward_index["rec_diff"]].append(float(mu_greedy_recall[r_id]))
                    if "rep" in self.config.list_of_rewards:
                        self.rewards_list_greedy[self.config.reward_index["rep"]].append(float(mu_greedy_rep[r_id]))
                    if "LCS" in self.config.list_of_rewards:
                        self.rewards_list_greedy[self.config.reward_index["LCS"]].append(float(mu_greedy_LCS[r_id]))
                    if "rec_prog" in self.config.list_of_rewards:
                        self.rewards_list_greedy[self.config.reward_index["rec_prog"]].append(float(mu_greedy_recall_prog[r_id]))
                    if "exec" in self.config.list_of_rewards:
                        self.rewards_list_greedy[self.config.reward_index["exec"]].append(float(mu_greedy_exec[r_id]))
                    


            
            if "prec" in self.config.list_of_rewards:
                mu_sample[self.config.reward_index["prec"]] = mu_sample_prec.sum()
            if "rec_diff" in self.config.list_of_rewards:
                mu_sample[self.config.reward_index["rec_diff"]] = mu_sample_recall.sum()
            if "rep" in self.config.list_of_rewards:
                mu_sample[self.config.reward_index["rep"]] = mu_sample_rep.sum()
            if "LCS" in self.config.list_of_rewards:
                mu_sample[self.config.reward_index["LCS"]] = mu_sample_LCS.sum()
            if "rec_prog" in self.config.list_of_rewards:
                mu_sample[self.config.reward_index["rec_prog"]] = mu_sample_recall_prog.sum()
            if "exec" in self.config.list_of_rewards:
                mu_sample[self.config.reward_index["exec"]] = mu_sample_exec.sum()
            
            
            if "prec" in self.config.list_of_rewards:
                mu_greedy[self.config.reward_index["prec"]] = mu_greedy_prec.sum()
            if "rec_diff" in self.config.list_of_rewards:
                mu_greedy[self.config.reward_index["rec_diff"]] = mu_greedy_recall.sum()
            if "rep" in self.config.list_of_rewards:
                mu_greedy[self.config.reward_index["rep"]] = mu_greedy_rep.sum()
            if "LCS" in self.config.list_of_rewards:
                mu_greedy[self.config.reward_index["LCS"]] = mu_greedy_LCS.sum()
            if "rec_prog" in self.config.list_of_rewards:
                mu_greedy[self.config.reward_index["rec_prog"]] = mu_greedy_recall_prog.sum()
            if "exec" in self.config.list_of_rewards:
                mu_greedy[self.config.reward_index["exec"]] = mu_greedy_exec.sum()
            


            self.mu_expert += mu_expert
            self.mu_sample += mu_sample
            self.mu_greedy += mu_greedy

            
            # sampled log prob wth mask
            B, T, C = logits_sample.size()
            prob_flat = logits_sample.contiguous().view(-1, C)
            logp_flat = F.log_softmax(prob_flat, dim=-1) # (BxT,C) log probabilities
            p_flat = F.softmax(prob_flat, dim=-1) # (BxT,C) probabilities
            logp = logp_flat.view(B,T,C)

            BG, TG, CG = logits_greedy.size()
            prob_flat_greedy = logits_greedy.contiguous().view(-1, CG)
            logp_flat_greedy = F.log_softmax(prob_flat_greedy, dim=-1) # (BxT,C) log probabilities
            prob_flat_greedy = F.softmax(prob_flat_greedy, dim=-1) # (BxT,C) probabilities
            logp_greedy = logp_flat_greedy.view(BG,TG,CG)



            plogp = p_flat.view(B,T,C) * logp
            entropy_elementwise = -plogp
            # print("entropy_elementwise = ", entropy_elementwise.size())            
            entropy_statewise = entropy_elementwise.sum(2)
            # print("entropy_statewise= ", entropy_statewise.size())
            masked_entropy_statewise = mask_sample * entropy_statewise
            # print("masked_entropy_statewise= ", masked_entropy_statewise.size())
            mean_entropy_statewise = masked_entropy_statewise.sum(1) / mask_sample.sum(1)
            
            ttttemp = step_ids_sample.unsqueeze(2)
            logp = torch.gather(logp, 2, step_ids_sample.unsqueeze(2)) # (B,T)
            logp = logp.squeeze(2)
            
            ttttemp_greedy = step_ids_greedy.unsqueeze(2)
            logp_greedy = torch.gather(logp_greedy, 2, step_ids_greedy.unsqueeze(2)) # (B,T)
            logp_greedy = logp_greedy.squeeze(2)
            
            
            
            masked_logp = (mask_sample * logp)
            masked_logp_greedy = (mask_greedy * logp_greedy)
            
            
            for _masked_logp in masked_logp.clone():
                self.all_masked_logp.append(_masked_logp.clone().detach())

            for _masked_logp_greedy in masked_logp_greedy.clone():
                self.all_masked_logp_greedy.append(_masked_logp_greedy.clone().detach())

            # print the rewards for debugging
            debug_flag = False
            if debug_flag :
                # printing the 1st sample of every batch
                print("===============================")
                print("rewards debug")
                print()
                # print text
                for zz in range(len(padded_input_tensor[0])):
                    print(id2word[int(padded_input_tensor[0][zz])], end=" , ")
                print()
                # print noun
                print("nouns")
                for zz in range(len(noun_tensor[0])):
                    print(id2word[int(noun_tensor[0][zz])], end=" , ")
                # print(noun_tensor[0])
                print()
                print("expert")
                # print target steps with reward
                for stepz in range(max_decoder_length):
                    print(self.decoder.module.id2action_object_pair[int(padded_target_tensor[0][stepz])], 
                                                    float(distinct_flags_target[0][stepz]), 
                                                    float(new_nouns_covered_flag_target[0][stepz]),
                                                    float(coverage_reward_target[0][stepz]),
                                                    float(similarity_indicator_target[0][stepz]),
                                                    float(masked_prec_reward_target[0][stepz]), float(masked_recall_reward_target[0][stepz]),
                                                    float(masked_rep_reward_target[0][stepz]), float(reward_LCS_expert[0][stepz]),
                                                    float(masked_recall_prog_reward_target[0][stepz]) )
                # print predicted steps with reward
                print()
                print("sampled")
                for stepz in range(max_decoder_length):
                    print(self.decoder.module.id2action_object_pair[int(step_ids_sample_np[0][stepz])],
                                                    float(distinct_flags_sample[0][stepz]), 
                                                    float(new_nouns_covered_flag_sample[0][stepz]),
                                                    float(coverage_reward_sample[0][stepz]),
                                                    float(similarity_indicator_sample[0][stepz]),
                                                    float(masked_prec_reward_sample[0][stepz]), float(masked_recall_reward_sample[0][stepz]),
                                                    float(masked_rep_reward_sample[0][stepz]), float(reward_LCS_sample[0][stepz]),
                                                    float(masked_recall_prog_reward_sample[0][stepz]) )
                print("===============================")
                print()
                print("greedy")
                for stepz in range(max_decoder_length):
                    print(self.decoder.module.id2action_object_pair[int(step_ids_greedy_np[0][stepz])],
                                                    float(distinct_flags_greedy[0][stepz]), 
                                                    float(new_nouns_covered_flag_greedy[0][stepz]),
                                                    float(coverage_reward_greedy[0][stepz]),
                                                    float(similarity_indicator_greedy[0][stepz]),
                                                    float(masked_prec_reward_greedy[0][stepz]), float(masked_recall_reward_greedy[0][stepz]),
                                                    float(masked_rep_reward_greedy[0][stepz]), float(reward_LCS_greedy[0][stepz]),
                                                    float(masked_recall_prog_reward_greedy[0][stepz]) )
                print("===============================")

                import sys
                sys.exit()
                debug_flag = False

            loss_val = None

            if not self.dummy_epoch:
                
                # pg version
                total_reward = (self.reward_weights[0] * reward_prec_stepped / self.rewards_variance_supervised_greedy[0]) + (self.reward_weights[1] * reward_recall_stepped / self.rewards_variance_supervised_greedy[1]) \
                    + (self.reward_weights[2] * reward_rep_stepped / self.rewards_variance_supervised_greedy[2]) + (self.reward_weights[3] * reward_LCS_stepped / self.rewards_variance_supervised_greedy[3]) \
                        + (self.reward_weights[4] * reward_recall_prog_stepped / self.rewards_variance_supervised_greedy[4])
                            # + (self.reward_weights[5] * exec_reward_stepped / self.rewards_variance_supervised_greedy[5])
                
            
                # for rl
                #     total_reward = (reward_LCS_stepped / self.rewards_variance_supervised_greedy[0]) + (0.1 * exec_reward_stepped / self.rewards_variance_supervised_greedy[1])
                

                reward_masked_logp = total_reward * masked_logp
                
                reward_masked_logp_sum = reward_masked_logp.sum(1) / mask_sample.sum(1)
                loss = - ( reward_masked_logp_sum + 0.05 * mean_entropy_statewise)
                #for rl no entropy reg
                # loss = - ( reward_masked_logp_sum )
                
                

                loss = loss.mean()
                
                
                # backward
                loss.backward()
                loss_val = loss.item()
                

            

            
            if self.this_irl_step and self.config.do_irl :
                # weight reassignment
                print("self.this irl step")
                self.mu_expert /= self.count_all_samples
                self.mu_sample /= self.count_all_samples
                self.mu_greedy /= self.count_all_samples

                # normalise
                print("before normalise mu expert = ", self.mu_expert)
                print("before normalise mu sample = ", self.mu_sample)
                print("before normalise mu greedy = ", self.mu_greedy)
                
                print("variance expert : ", self.rewards_variance_expert)
                print("variance supervised : ", self.rewards_variance_supervised)
                print("variance supervised greedy : ", self.rewards_variance_supervised_greedy)

                # self.mu_expert /= self.rewards_variance_expert 
                self.mu_expert /= self.rewards_variance_supervised_greedy 
                self.mu_sample /= self.rewards_variance_supervised_greedy
                self.mu_greedy /= self.rewards_variance_supervised_greedy

                print("after normalise mu expert = ", self.mu_expert)
                print("after normalise mu sample = ", self.mu_sample)
                print("after normalise mu greedy = ", self.mu_greedy)
                
                

                
                
                if self.config.irl_technique == "MaxEnt":
                    
                
                    for irl_update_id in range(2):
                        
                        # sample traj ids for expert and policy 
                        traj_ids = list(range(self.all_train_samples_count))
                        random.shuffle(traj_ids)
                        expert_traj_ids = traj_ids[:self.config.IRL_sample_count]
                        print("expert_traj_ids = ", expert_traj_ids)
                        random.shuffle(traj_ids)
                        policy_traj_ids = traj_ids[:self.config.IRL_sample_count]
                        print("policy_traj_ids = ", policy_traj_ids)
                        
                        # get mean rewards expert
                        mean_reward_expert = torch.zeros(self.config.num_rewards).cuda()
                        # print("here 3 a = ", type(self.rewards_list_expert[0]))
                        # print("here 3 b = ", type(self.rewards_list_expert[5]))
                        for traj_id in expert_traj_ids:
                            for abc in range(self.config.num_rewards):
                                mean_reward_expert[abc] += self.rewards_list_expert[abc][traj_id] #/ self.rewards_variance_supervised_greedy[0]
                            

                        mean_reward_expert /= self.config.IRL_sample_count
                        
                        print("before normalise mean_reward_expert = ", mean_reward_expert)
                        
                        
                        
                        # get mean reward policy
                        mean_reward_policy = torch.zeros(self.config.num_rewards).cuda()
                        sum_imp_weights = 0
                        all_curr_reward_policy = torch.zeros(len(policy_traj_ids),self.config.num_rewards).cuda()
                        all_imp_weights = torch.zeros(len(policy_traj_ids)).cuda()
                        for traj_id_index in range(len(policy_traj_ids)):
                            traj_id = policy_traj_ids[traj_id_index]
                            
                            
                            curr_reward_policy = torch.zeros(self.config.num_rewards).cuda()
                            for abc in range(self.config.num_rewards):
                                curr_reward_policy[abc] += self.rewards_list_greedy[abc][traj_id] #/ self.rewards_variance_supervised_greedy[0]
                            
                            print("traj_id = ", traj_id)
                            print("before normalise curr_reward_policy = ", curr_reward_policy)
                            
                            # do importance sampling
                            # curr_logp = self.all_masked_logp[traj_id]
                            curr_logp = self.all_masked_logp_greedy[traj_id]
                            curr_logp = curr_logp.sum()
                            curr_reward_val = self.reward_weights.view(1,self.config.num_rewards).matmul(curr_reward_policy.view(self.config.num_rewards,1)).view(-1)
                            
                            curr_traj_imp_weight = torch.exp(curr_reward_val - curr_logp - 100 )
                            all_imp_weights[traj_id_index] = curr_traj_imp_weight
                            all_curr_reward_policy[traj_id_index] = curr_reward_policy
                            

                        all_imp_weights /= all_imp_weights.sum()
                        all_imp_weights = all_imp_weights.view(-1,1).repeat(1,self.config.num_rewards)
                        mean_reward_policy = all_imp_weights * all_curr_reward_policy
                        mean_reward_policy = mean_reward_policy.sum(0)

                        
                        
                        
                        # get gradient
                        print("mean reward expert = ", mean_reward_expert)
                        print("mean reward policy = ", mean_reward_policy)
                        grad = mean_reward_expert - mean_reward_policy
                        print("grad = ", grad)
                        

                        # update
                        # show before, after just for debug, supposed to increase
                        before = (self.reward_weights.view(1,self.config.num_rewards).matmul((mean_reward_expert - mean_reward_policy).view(self.config.num_rewards,1))).item()
                        
                        self.reward_weights +=  grad
                        print("Updated weight = ", self.reward_weights)

                        after = (self.reward_weights.view(1,self.config.num_rewards).matmul((mean_reward_expert - mean_reward_policy).view(self.config.num_rewards,1))).item()
                        print("Updated; before = ", before, " after = ", after, " diff = ", after-before)


                    

                
                
        # update params
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # clearing up memory
        del padded_target_tensor
        del padded_input_tensor
        del weight_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        # retun loss value for keeping track
        # print("here 5 = ", type(self.rewards_list_expert[5]))

        return loss_val
    
    def get_n_params(self,model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp




    def trainIters(self, n_iters, print_every=None, plot_every=100, learning_rate=0.001):
        start = time.time()
        self.average_epoch_loss = []
        
        writer = SummaryWriter()
        lr = self.config.learning_rate
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr = lr)

        # load optimizer
        if os.path.exists(self.config.decoder_optimizer_path) and self.config.enable_model_load:
            self.decoder_optimizer.load_state_dict(torch.load(self.config.decoder_optimizer_path_load))
            self.encoder_optimizer.load_state_dict(torch.load(self.config.encoder_optimizer_path_load))
            for param_group in self.encoder_optimizer.param_groups:
                lr = param_group['lr']
                break
            print("Optimizer loaded")

        self.encoder_scheduler = ReduceLROnPlateau(self.encoder_optimizer, mode = "max", patience=2)
        self.decoder_scheduler = ReduceLROnPlateau(self.decoder_optimizer, mode = "max" , patience=2)

        all_train_samples = json.load(open("./disjoint_data/train_features_70_samples.json", "r"))
        

        total_batches = int(len(all_train_samples)/self.config.batch_size)
        if int(len(all_train_samples) % self.config.batch_size) != 0:
            total_batches += 1

        
        self.count_all_samples =  len(all_train_samples)
        self.all_train_samples_count = self.count_all_samples

        self.this_irl_step = False
        self.irl_init = False
        self.reward_weights = None
        self.mu_prime = None
        self.weights_error = None

        self.rewards_variance_expert = None # list of variances of each reward
        self.rewards_variance_supervised = None # list of variances of each reward

        self.rewards_list_expert = []
        self.rewards_list_sample = []
        self.rewards_list_greedy = []

        
        start_batch_num = 0 if self.config.training_procedure == "SCST" else 1
        if (self.config.training_procedure == "SCST" and self.config.irl_loading == True):
            start_batch_num = 1
        # assert (start_batch_num == 1)
        for iter in range(start_batch_num, n_iters + 1):
            if iter == 0 and self.config.training_procedure == "SCST" and not self.config.irl_loading :
                self.dummy_epoch = True
                self.variance_calc = True
                self.rewards_list_expert = []
                self.rewards_list_sample = []
                self.rewards_list_greedy = []
                for _ in range(self.config.num_rewards):
                    self.rewards_list_expert.append([])
                    self.rewards_list_sample.append([])
                    self.rewards_list_greedy.append([])
                
                

                self.mu_expert = torch.zeros(self.config.num_rewards).cuda()
                self.mu_sample = torch.zeros(self.config.num_rewards).cuda()
                self.mu_greedy = torch.zeros(self.config.num_rewards).cuda()

                self.all_masked_logp = []
                self.all_masked_logp_greedy = []
            else :
                self.dummy_epoch = False
                self.variance_calc = False
            if iter == 1 and self.config.training_procedure == "SCST" and not self.config.irl_loading:
                # self.irl_init :
                print("this irl init")
                self.mu_expert /= self.count_all_samples
                self.mu_sample /= self.count_all_samples
                
                print("before normalise self.expert = ", self.mu_expert)
                print("before normalise self.sample = ", self.mu_sample)
                # print("here 6 = ", type(self.rewards_list_expert[5]))

                # calc variance
                self.rewards_variance_expert = torch.zeros(self.config.num_rewards).cuda()
                self.rewards_variance_supervised = torch.zeros(self.config.num_rewards).cuda()
                self.rewards_variance_supervised_greedy = torch.zeros(self.config.num_rewards).cuda()
                for jk in range(self.config.num_rewards):
                    self.rewards_variance_expert[jk] = float(np.std(np.array(self.rewards_list_expert[jk]))) + 1e-8
                    self.rewards_variance_supervised[jk] = float(np.std(np.array(self.rewards_list_expert[jk]))) + 1e-8
                    self.rewards_variance_supervised_greedy[jk] = float(np.std(np.array(self.rewards_list_expert[jk]))) + 1e-8
                
                

               
                
                
                print("variance supervised greedy : ", self.rewards_variance_supervised_greedy)
                
                self.mu_expert /= self.rewards_variance_supervised_greedy
                self.mu_sample /= self.rewards_variance_supervised_greedy
                self.mu_greedy /= self.rewards_variance_supervised_greedy

                if self.config.do_irl :
                    
                    
                

                    

                    if self.config.irl_technique == "MaxEnt":
                        # first time IRL, init weight as random
                        self.reward_weights = torch.rand(self.config.num_rewards).float().cuda()
                        print("random init = ", self.reward_weights)
                        # do gradient ascent on weights in a loop
                    
                        for irl_update_id in range(2):
                            # self.nn_reward_net_optimizer.zero_grad()

                            # sample traj ids for expert and policy 
                            traj_ids = list(range(self.all_train_samples_count))
                            random.shuffle(traj_ids)
                            expert_traj_ids = traj_ids[:self.config.IRL_sample_count]
                            random.shuffle(traj_ids)
                            policy_traj_ids = traj_ids[:self.config.IRL_sample_count]
                            
                             # get mean rewards expert
                            mean_reward_expert = torch.zeros(self.config.num_rewards).cuda()
                            for traj_id in expert_traj_ids:
                                for abc in range(self.config.num_rewards):
                                    mean_reward_expert[abc] += self.rewards_list_expert[abc][traj_id]
                                
                            mean_reward_expert /= self.config.IRL_sample_count 
                            
                            # get mean reward policy
                            mean_reward_policy = torch.zeros(self.config.num_rewards).cuda()
                            sum_imp_weights = 0
                            all_curr_reward_policy = torch.zeros(len(policy_traj_ids),self.config.num_rewards).cuda()
                            all_imp_weights = torch.zeros(len(policy_traj_ids)).cuda()
                        
                            for traj_id_index in range(len(policy_traj_ids)):
                                traj_id = policy_traj_ids[traj_id_index]
                                
                                curr_reward_policy = torch.zeros(self.config.num_rewards).cuda()
                                for abc in range(self.config.num_rewards):
                                    curr_reward_policy[abc] += self.rewards_list_greedy[abc][traj_id]
                                
                                # do importance sampling
                                # curr_logp = self.all_masked_logp[traj_id]
                                curr_logp = self.all_masked_logp_greedy[traj_id]
                                # print(self.all_masked_logp_greedy)
                                curr_logp = curr_logp.sum()
                                print("curr_logp = ", torch.exp(curr_logp))
                                curr_traj_imp_weight = torch.exp(self.reward_weights.view(1,self.config.num_rewards).matmul(curr_reward_policy.view(self.config.num_rewards,1)).view(-1)) / torch.exp(curr_logp)
                                
                                print("importance weight = ", curr_traj_imp_weight)
                                all_imp_weights[traj_id_index] = curr_traj_imp_weight
                                all_curr_reward_policy[traj_id_index] = curr_reward_policy
                            
                                

                            # mean_reward_policy /= sum_imp_weights 
                            all_imp_weights /= all_imp_weights.sum()
                            all_imp_weights = all_imp_weights.view(-1,1).repeat(1,self.config.num_rewards)
                            mean_reward_policy = all_imp_weights * all_curr_reward_policy
                            mean_reward_policy = mean_reward_policy.sum(0)
                            # get gradient
                            print("mean reward expert = ", mean_reward_expert)
                            print("mean reward policy = ", mean_reward_policy)
                            grad = mean_reward_expert - mean_reward_policy
                            print("grad = ", grad)
                            
                            # update
                            # show before, after just for debug, supposed to increase
                            before = (self.reward_weights.view(1,self.config.num_rewards).matmul((mean_reward_expert - mean_reward_policy).view(self.config.num_rewards,1))).item()
                            
                            self.reward_weights +=  grad
                            print("Updated weight = ", self.reward_weights)

                            after = (self.reward_weights.view(1,self.config.num_rewards).matmul((mean_reward_expert - mean_reward_policy).view(self.config.num_rewards,1))).item()
                            print("Updated; before = ", before, " after = ", after, " diff = ", after-before)

                        
                    
                    
                    


            

            print ("epoch : ",iter)
            
            

            self.encoder.train()
            self.decoder.train()
            self.encoder.zero_grad()
            self.decoder.zero_grad()

            epoch_loss = []    
            
            
            total_batches = int(len(all_train_samples)/self.config.batch_size)
            self.count_all_samples =  int(self.config.batch_size * total_batches)
            if len(all_train_samples) % self.config.batch_size != 0:
                total_batches += 1
                self.count_all_samples = len(all_train_samples)
                
                
            
            for b_num in tqdm(range(total_batches)):
                self.this_irl_step = False



                # train_pair = next(train_pair_gen)
                # tokens, nouns, steps, entities in each step, beginning room
                if b_num != total_batches - 1:
                    train_pair = all_train_samples[b_num * self.config.batch_size : (b_num + 1) * self.config.batch_size  ]
                else:
                    train_pair = all_train_samples[b_num * self.config.batch_size :  ]
                self.input_tensor = [i[0] for i in train_pair]
                self.nouns = [i[1] for i in train_pair]
                self.target_np = [i[2] for i in train_pair]                
                self.env_tensor_entites = [i[3] for i in train_pair]
                self.env_tensor_beginning_room = [i[4] for i in train_pair]
                self.environment = [i[6] for i in train_pair]
                self.preconds_file_path = [i[5] for i in train_pair]
                self.target_sp_tensor = [i[7] for i in train_pair]
                if self.config.np_context : 
                    self.noun_phrase = [i[8] for i in train_pair]
                else :
                    self.noun_phrase = None

                self.preconds_file_path = [ x.replace("withoutconds", "initstate") for x in self.preconds_file_path]
                self.preconds_file_path = [ x.replace(".txt", ".json") for x in self.preconds_file_path]



                epoch_loss.append(self.train(self.input_tensor, self.env_tensor_entites, self.env_tensor_beginning_room, self.noun_phrase, None, writer=writer) )
                
            if not self.dummy_epoch :
                avg_loss = np.average(np.array(epoch_loss))
                print("Average loss for epoch = ", avg_loss)
                
            
                self.average_epoch_loss.append(avg_loss)
                writer.add_scalar('Avg. Loss', avg_loss, iter)
            
            if iter %5 == 0 and iter > 1 and self.config.training_procedure == "SCST" and self.config.do_irl :
                # time for IRL weight update
                self.dummy_epoch = True # prevent policy updates
                self.mu_expert = torch.zeros(self.config.num_rewards).cuda()
                self.mu_sample = torch.zeros(self.config.num_rewards).cuda()
                self.mu_greedy = torch.zeros(self.config.num_rewards).cuda()
                self.variance_calc = True
                self.rewards_list_expert = []
                self.rewards_list_sample = []
                self.rewards_list_greedy = []
                for _ in range(self.config.num_rewards):
                    self.rewards_list_expert.append([])
                    self.rewards_list_sample.append([])
                    self.rewards_list_greedy.append([])
                
                self.all_masked_logp = []
                self.all_masked_logp_greedy = []

                for b_num in tqdm(range(total_batches)):
                    if b_num == total_batches - 1 :
                        self.this_irl_step = True
                    else :
                        self.this_irl_step = False



                    # train_pair = next(train_pair_gen)
                    # tokens, nouns, steps, entities in each step, beginning room
                    if b_num != total_batches - 1:
                        train_pair = all_train_samples[b_num * self.config.batch_size : (b_num + 1) * self.config.batch_size  ]
                    else:
                        train_pair = all_train_samples[b_num * self.config.batch_size :  ]
                    self.input_tensor = [i[0] for i in train_pair]
                    self.nouns = [i[1] for i in train_pair]
                    self.target_np = [i[2] for i in train_pair]                
                    self.env_tensor_entites = [i[3] for i in train_pair]
                    self.env_tensor_beginning_room = [i[4] for i in train_pair]
                    
                    self.environment = [i[6] for i in train_pair]
                    self.preconds_file_path = [i[5] for i in train_pair]
                    self.target_sp_tensor = [i[7] for i in train_pair]
                    if self.config.np_context : 
                        self.noun_phrase = [i[8] for i in train_pair]
                    else :
                        self.noun_phrase = None
                    self.preconds_file_path = [ x.replace("withoutconds", "initstate") for x in self.preconds_file_path]
                    self.preconds_file_path = [ x.replace(".txt", ".json") for x in self.preconds_file_path]

                    epoch_loss.append(self.train(self.input_tensor, self.env_tensor_entites, self.env_tensor_beginning_room, self.noun_phrase, None, writer=writer) )
                    if self.this_irl_step:
                        # last weights added and irl updates done
                        print("reward weights = ", self.reward_weights.clone().detach().cpu().numpy())
                        print("reward weights error = ", self.weights_error)

            
            if iter % 5 == 0:
                # save
                if self.config.data_parallel:
                    torch.save(self.encoder.module.state_dict(),self.config.encoder_model_path.format(iter))
                    torch.save(self.decoder.module.state_dict(), self.config.decoder_model_path.format(iter))
                else:
                    torch.save(self.encoder.state_dict(),self.config.encoder_model_path.format(iter))
                    torch.save(self.decoder.state_dict(), self.config.decoder_model_path.format(iter))
                print("Model saved")

                # save optimizer dict
                torch.save(self.encoder_optimizer.state_dict(),self.config.encoder_optimizer_path.format(iter))
                torch.save(self.decoder_optimizer.state_dict(),self.config.decoder_optimizer_path.format(iter))
                print("Optimizer saved")
            
            
                  
            
        end = time.time()
        print("Total time taken : ", end-start)
        writer.export_scalars_to_json("./all_scalars_log.json")
        writer.close()