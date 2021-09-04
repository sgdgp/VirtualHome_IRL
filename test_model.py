
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
from utils import *
import json
import sys
import multiprocessing
from joblib import Parallel, delayed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Test_MODEL():
    def __init__(self, data, encoder, decoder, config, mode="test"):
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.config = config
        
        self.test_pair = json.load(open("./disjoint_data/val_features_185_samples.json"))
        # self.test_pair = json.load(open("./disjoint_data/test_features_500_samples.json"))
        
        
  
        
        
        self.test_input = [i[0] for i in self.test_pair]
        self.target_np = [i[2] for i in self.test_pair]    
        self.noun_ids = [i[1] for i in self.test_pair]
        self.env_tensor_entites = [i[3] for i in self.test_pair]
        self.beginning_room_entity_masks = [i[4] for i in self.test_pair]
        self.test_preconds_path = [i[5] for i in self.test_pair]
        self.environment = [i[6] for i in self.test_pair]
        
        self.target_sp_tensor = [i[7] for i in self.test_pair]
        

        self.word2id = json.load(open("./disjoint_data/word2id.json","r"))
        self.word2id = {str(k):int(v) for k,v in self.word2id.items()}
        self.id2word = json.load(open("./disjoint_data/id2word.json","r"))
        self.id2word = {int(k):str(v) for k,v in self.id2word.items()}
        


        self.test_preconds_path = [ x.replace("withoutconds", "initstate") for x in self.test_preconds_path]
        self.test_preconds_path = [ x.replace(".txt", ".json") for x in self.test_preconds_path]
        
        

        
        self.mode = mode

        self.bidirectional_rnn_encoder = False
        
    def evaluate(self):
        if self.mode == "test_random_sampling":
            return self.evaluate_random_sampling()

        if self.mode == "test_noun_grounding_preprocess":
            return self.do_noun_grounding_preprocess()

        encoder_hidden = None
        encoder_cell = None
        encoder_output = None
        loss = 0

        self.encoder.eval()
        self.decoder.eval()
        lcs_a_list = []
        lcs_o_list = []
        lcs_s_list = []
        lcs_ws_list = []
        act_acc_list = []
        ed_list = []
        exec_list = []
        all_sp_values = []
        cnt_p_greater_list = []
        cnt_p_lesser_list = []
        l2_loss_list = []

        sp_step_tuple_list = []

        text_step_tuple_list = []
        
        all_test_tuples = []

        count_exec = 0

        
        do_group_analysis = False
        if do_group_analysis :
            count_group = np.zeros((6,))
            lcs_score_group = np.zeros((6,))
            act_score_group = np.zeros((6,))
            ed_score_group = np.zeros((6,))
            exec_score_group = np.zeros((6,))

            def calc_group_id(length):
                if length <=25 :
                    return int((length-1)/5)
                else:
                    return 5


       


        # test_batch_size = 16
        test_batch_size = 16
        print("Total samples to test : ",len(self.test_input) )
        num_batches = int(len(self.test_input) / test_batch_size)
        if len(self.test_input) % test_batch_size != 0:
            num_batches += 1
        start_id = 0
        end_id = 0
        for idx in tqdm(range(num_batches)):
            batch_size = test_batch_size
            if idx != num_batches - 1:
                end_id = start_id + batch_size
            else :
                end_id = len(self.test_input)
                batch_size = max(int(len(self.test_input) % test_batch_size), 1)

            input_tensor = self.test_input[start_id:end_id]
            nouns_curr = torch.from_numpy(np.array(self.noun_ids[start_id:end_id]))
            env_tensor_beginning_room_curr = torch.from_numpy(np.array(self.beginning_room_entity_masks[start_id:end_id]))
            
            if self.config.np_context:
                noun_phrase_tensor = self.test_noun_phrase_tensor[start_id : end_id]
            
            

            seq_lengths = torch.tensor([len(seq) for seq in input_tensor], dtype=torch.long, device=device)
            # print(seq_lengths)
            max_length = seq_lengths.max().item()
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            _, reverse_perm_idx = perm_idx.sort(0)
            padded_input_tensor = []
            mask_tensor = torch.ones((batch_size, max_length), device=device)
            
            for i in range(batch_size):
            # for sample in input_tensor:
                padded_sample = input_tensor[i].copy()
                for j in range(max_length - len(input_tensor[i])):
                    padded_sample.append(self.word2id["<pad>"])
                    # print(len(input_tensor[i]) + j)
                    mask_tensor[i][len(input_tensor[i]) + j] = 0.0
                
                padded_input_tensor.append(torch.tensor(padded_sample, dtype=torch.long, device=device))
            
            padded_input_tensor = torch.stack(padded_input_tensor)
            
            padded_noun_phrase = torch.zeros((batch_size, 23)).long().cuda()
            mask_noun_phrase = torch.zeros((batch_size, 23)).float().cuda()
            if self.config.np_context:
                for i in range(batch_size):
                    for k in range(len(noun_phrase_tensor[i])):
                        padded_noun_phrase[i][k] = noun_phrase_tensor[i][k]
                        mask_noun_phrase[i][k] = 1
             
            
            
            padded_input_tensor = padded_input_tensor[perm_idx]
            encoder_hidden = self.encoder.module.initHidden(batch_size) if self.config.data_parallel else self.encoder.initHidden(batch_size)
            encoder_cell = self.encoder.module.initHidden(batch_size) if self.config.data_parallel else self.encoder.initHidden(batch_size)

            encoder_hidden = encoder_hidden.transpose(0,1)
            encoder_cell = encoder_cell.transpose(0,1)

            
            all_encoder_hidden, (encoder_hidden, encoder_cell), _ , self_attended_output = self.encoder(padded_input_tensor,seq_lengths.view(batch_size,-1), (encoder_hidden, encoder_cell))
            
            

            padded_input_tensor = padded_input_tensor[reverse_perm_idx]
            all_encoder_hidden = all_encoder_hidden[reverse_perm_idx]
            encoder_hidden = encoder_hidden[reverse_perm_idx]
            encoder_cell = encoder_cell[reverse_perm_idx]
            # seq_lengths = seq_lengths[reverse_perm_idx]
            self_attended_output = self_attended_output[reverse_perm_idx]

            if self.bidirectional_rnn_encoder:
                encoder_hidden = encoder_hidden.view(encoder_hidden.size(0), 1 , -1)
                encoder_cell = encoder_cell.view(encoder_cell.size(0), 1 , -1)

            self_attended_output = self_attended_output.view(self_attended_output.size(0),1,self_attended_output.size(1))

            # (decoder_hidden, decoder_cell) = (self_attended_output.contiguous(), encoder_cell.contiguous())
            (decoder_hidden, decoder_cell) = (encoder_hidden.contiguous(), encoder_cell.contiguous())
            # (decoder_hidden_env, decoder_cell_env) = (encoder_hidden.contiguous(), encoder_cell.contiguous())
            (decoder_hidden_env, decoder_cell_env) = (torch.zeros(encoder_hidden.size()).cuda(), torch.zeros(encoder_cell.size()).cuda())
            
            pred = []
            use_teacher_forcing = False 
            use_teacher_forcing_tensor = torch.zeros((batch_size, 1), dtype=torch.uint8, device=device)
            target_seq_lengths = torch.tensor([len(self.target_np[bid]) for bid in range(start_id, end_id)], dtype=torch.long, device=device)
            
            max_lll = target_seq_lengths.max().item()
            
           
            beam_size = int(sys.argv[2]) if self.mode != "validation" else 1
            max_decoder_length = 10
            # max_decoder_length = max_lll
            # pred,alpha_seq

            env_mask_beginning_room = torch.from_numpy(np.array(env_tensor_beginning_room_curr)).cuda()
            noun_tensor = torch.from_numpy(np.array(nouns_curr)).cuda()
            
            pred_all, sp_all = self.decoder(input = None, 
                                        env_entities = None,
                                        beginning_room_entities = env_mask_beginning_room,
                                        noun_ids = noun_tensor,
                                        padded_noun_phrase = padded_noun_phrase,
                                        mask_noun_phrase = mask_noun_phrase,
                                        hidden_cell= (decoder_hidden, decoder_cell), 
                                        hidden_cell_env= (decoder_hidden_env, decoder_cell_env), 
                                        all_encoder_hidden = all_encoder_hidden, 
                                        seq_lengths = target_seq_lengths.view(batch_size, -1), 
                                        mask_tensor = mask_tensor.view(batch_size,-1), 
                                        use_teacher_forcing_tensor = use_teacher_forcing_tensor, 
                                        max_decoder_length = max_decoder_length,
                                        training_mode = False,
                                        beam_size=beam_size,
                                        get_sampled_sequence = False, # sampling false
                                        preconds_file_path = self.test_preconds_path[start_id :end_id],
                                        environment = self.environment[start_id : end_id])
                                        
            
            
            
            for bid in range(batch_size):
                pred = pred_all[bid]
                pred = pred.detach().cpu().numpy()
                sp = sp_all[bid]
                sp = sp.detach().cpu().numpy()
                for _sp in sp:
                    all_sp_values.append(_sp)
                
                sp_g = sp[sp > 0.5]
                sp_l = sp[sp <= 0.5]
                
                cnt_p_greater = 0
                cnt_p_lesser = 0
                
                # assert (cnt_p_greater + cnt_p_lesser == 1.0)
                cnt_p_greater_list.append(cnt_p_greater)
                cnt_p_lesser_list.append(cnt_p_lesser)

                pred = pred[1:-1]
                gt = self.target_np[start_id + bid][1:-1]
                gt_sp = self.target_sp_tensor[start_id + bid]
                


                
                if self.mode == "test":
                    test_tuple = {}
                    print()
                    print("==========================================")
                    print("Text")
                    abc_text = [self.id2word[id] for id in self.test_input[start_id + bid]]
                    # abc_text = [self.id2word_augment[id] for id in self.test_input[start_id + bid]]
                    abc_text = " ".join(abc_text)
                    print(abc_text)
                    
                    nouns = [self.id2word[id] for id in self.noun_ids[start_id + bid] if int(id) != 3 ]
                    nouns = " , ".join(nouns)
                    print("Nouns : ", nouns)
                    
                    test_tuple["text"] = abc_text
                    test_tuple["preconds_path"] = self.test_preconds_path[start_id + bid]
                    test_tuple["environment"] = self.environment[start_id + bid]

                    print()
                    print()
                    print("path : ", self.test_preconds_path[start_id + bid])
                    
                    print("Ground Truth program : ")
                    test_tuple["gt_ids"] = gt
                    abc = get_prog_from_step_ids(gt, self.decoder.id2action_object_pair)
                    # abc = get_prog_from_step_ids(gt, self.id2step_augment)
                    test_tuple["gt_text"] = abc
                    for step in abc:
                        print(step)

                    print()
                    print()

                    print("Pred program : ")     
                    test_tuple["pred_ids"] = pred.tolist()
                    abc = get_prog_from_step_ids(pred, self.decoder.id2action_object_pair)
                    # abc = get_prog_from_step_ids(pred, self.id2step_augment)
                    test_tuple["pred_text"] = abc
                    all_test_tuples.append(test_tuple)    
                    
                    text_step_tuple_list.append((abc_text, pred.tolist(), sp.tolist()))
                    for i  in range(len(abc)):
                        step=abc[i]
                        print(step,end=" ")
                        
                        
                        if i < len(gt_sp):
                            # l2_loss_list.append((sp[i] - gt_sp[i])**2)
                            l2_loss_list.append(0)
                        
                        print()
                
                self.config.id2action_object_pair = self.decoder.id2action_object_pair
                # self.config.id2action_object_pair = self.id2step_augment

                lcs_a, lcs_o, lcs_s = LCS_A_O_S(pred, gt, self.config)
                
                ed = edit_distance(pred, gt)
                
                
                if self.mode == "test":
                    print("==========================================")
                    print()
                
                lcs_a_list.append(lcs_a)
                lcs_o_list.append(lcs_o)
                lcs_s_list.append(lcs_s)
                
                ed_list.append(ed)
                # exec_list.append(exec)

                if do_group_analysis:
                    group_id = calc_group_id(len(gt))
                    count_group[group_id] += 1
                    lcs_score_group[group_id] += lcs
                    ed_score_group[group_id] += ed
                    act_score_group[group_id] += act_acc
                    exec_score_group[group_id] += exec


            start_id =  end_id

        avg_lcs_a = np.average(np.array(lcs_a_list))
        avg_lcs_o = np.average(np.array(lcs_o_list))
        avg_lcs_s = np.average(np.array(lcs_s_list))
        avg_l2_loss = np.average(np.array(l2_loss_list))
        
        if self.mode == "validation":
            print(self.mode)
            return avg_lcs_a, avg_lcs_o, avg_lcs_s
        
        print("Average_LCS Action = {:.3f}".format(avg_lcs_a))
        print("Average_LCS Object = {:.3f}".format(avg_lcs_o))
        print("Average_LCS Step = {:.3f}".format(avg_lcs_s))
        print ("Average LCS = {:.3f}".format(float(avg_lcs_a + avg_lcs_o + avg_lcs_s)/3.0 ))
        avg_lcs = float(avg_lcs_a + avg_lcs_o + avg_lcs_s)/3.0 
        
        avg_ed = np.average(np.array(ed_list))
        print("Average_ED = {:.3f}".format(avg_ed))
        print("Average_L2 Loss = {:.3f}".format(avg_l2_loss))
        print("Exec count : ", count_exec)

        avg_cnt_p_greater = np.average(np.array(cnt_p_greater_list))
        avg_cnt_p_lesser = np.average(np.array(cnt_p_lesser_list))

        print("Average_cnt_p_greater = {:.3f}".format(avg_cnt_p_greater))
        print("Average_cnt_p_lesser = {:.3f}".format(avg_cnt_p_lesser))
        
        with open(self.config.all_test_tuples_filename,"w") as f111:
            json.dump(all_test_tuples, f111) 
        
        with open("nosp.txt","w") as f:
            for _sp_step in sp_step_tuple_list:
                print(_sp_step)
                f.write(str(_sp_step[0]) + " " + str(_sp_step[1]))
                f.write("\n")

        json.dump(text_step_tuple_list, open("./text_step_tuple_list_v2.json","w"))
        
        
        if do_group_analysis:
            # normalise
            for gid in range(6):
                lcs_score_group[gid] /= float(count_group[gid])
                ed_score_group[gid] /= float(count_group[gid])
                act_score_group[gid] /= float(count_group[gid])
                exec_score_group[gid] /= float(count_group[gid])
            print("Groupwise scores :\t<=5\t>5 <=10\t>10 <=15\t>15 <=20\t>20 <=25\t>25")
            print("LCS ",end="\t")
            for gid in range(6):
                print("{:.2f}".format(lcs_score_group[gid]), end="\t")
            print()
            print("ED ",end="\t")
            for gid in range(6):
                print("{:.2f}".format(ed_score_group[gid]), end="\t")
            print()
            print("Act acc ",end="\t")
            for gid in range(6):
                print("{:.2f}".format(act_score_group[gid]), end="\t")
            print()
            print("Exec ",end="\t")
            for gid in range(6):
                print("{:.2f}".format(exec_score_group[gid]), end="\t")
            print()
            
        all_metrics = avg_lcs_a, avg_lcs_o, avg_lcs_s, avg_lcs, avg_ed
        
        return all_metrics


    def evaluate_random_sampling(self):
        lcs_a_list = []
        lcs_o_list = []
        lcs_s_list = []
        ed_list = []
        all_test_tuples = []
        self.config.id2action_object_pair = json.load(open("./disjoint_data/id2step.json", "r"))
        self.config.id2action_object_pair = {int(k):str(v) for k,v in self.config.id2action_object_pair.items()}

        self.config.action_object_pair2id = json.load(open("./disjoint_data/step2id.json", "r"))
        self.config.action_object_pair2id = {str(k):int(v) for k,v in self.config.action_object_pair2id.items()}
        
        for idx in tqdm(range(len(self.test_input))):
            test_tuple = {}
            abc_text = [self.id2word[id] for id in self.test_input[idx]]
            abc_text = " ".join(abc_text)
            # print(abc_text)
            
            nouns = [self.id2word[id] for id in self.noun_ids[idx] if int(id) != 3 ]
            nouns = " , ".join(nouns)
            
            test_tuple["text"] = abc_text
            test_tuple["preconds_path"] = self.test_preconds_path[idx]
            test_tuple["environment"] = self.environment[idx]
            
            gt = self.target_np[idx]
            gt = gt[1:-1]
            test_tuple["gt_ids"] = gt
            abc = get_prog_from_step_ids(gt, self.config.id2action_object_pair)
            test_tuple["gt_text"] = abc
            
            pred = [-1] # <bop> not needed to be same as index of <bop> in step embedding matrix, this is just a placeholder
            # wont be considered in lcs and other metric calcualtion


            for i in range(20):
                random_step_id = np.random.random_integers(low=0, high=int(len(self.config.action_object_pair2id))-1)
                pred.append(random_step_id)
                if (random_step_id) == int(self.config.action_object_pair2id["<eop>"]):
                    break

            if int(pred[-1]) != int(self.config.action_object_pair2id["<eop>"]):
                pred.append(int(self.config.action_object_pair2id["<eop>"]))
            pred = np.array(pred[1:-1])
            test_tuple["pred_ids"] = pred.tolist()
            abc = get_prog_from_step_ids(pred, self.config.id2action_object_pair)
            test_tuple["pred_text"] = abc
            all_test_tuples.append(test_tuple) 
            
            # get metrics
            LCS_A, LCS_O, LCS_S = LCS_A_O_S(pred,gt, self.config)
            ed = edit_distance(pred, gt)
            lcs_a_list.append(LCS_A)
            lcs_o_list.append(LCS_O)
            lcs_s_list.append(LCS_S)
            ed_list.append(ed)
            

        avg_lcs_a = np.average(np.array(lcs_a_list))
        print("Average_LCS Action = {:.3f}".format(avg_lcs_a))
        avg_lcs_o = np.average(np.array(lcs_o_list))
        print("Average_LCS Object = {:.3f}".format(avg_lcs_o))
        avg_lcs_s = np.average(np.array(lcs_s_list))
        print("Average_LCS Step = {:.3f}".format(avg_lcs_s))
        print ("Average LCS = {:.3f}".format(float(avg_lcs_a + avg_lcs_o + avg_lcs_s)/3.0 ))
        avg_ed = np.average(np.array(ed_list))
        print("Average_ED = {:.3f}".format(avg_ed))
        with open(self.config.all_test_tuples_filename,"w") as f111:
            json.dump(all_test_tuples, f111) 


