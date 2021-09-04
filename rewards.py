import json
import numpy as np

def get_coverage_similarity_score(logits, noun_ids):
        # get noun_embedding
        batch_size = logits.size(0)
        noun_embs = self.word_embedding_layer(noun_ids)
        logit_argmax = torch.argmax(logits, dim=1)
        all_max_scores = []
        for b in range(batch_size):
            max_score = -100
            step_id = logit_argmax[b]
            if step_id == int(self.action_object_pair2id["<eop>"]):
                all_max_scores.append(1.0)
                continue
            step = self.id2action_object_pair[int(step_id)]
            step = step.split("||")
            if len(step) == 1 :
                max_score = 1.0
            else:
                objs = step[1:]
                for obj in objs:
                    objid = int(self.object2id[obj])
                    obj_emb = self.obj_emb[objid]
                    obj_score = (obj_emb.matmul(noun_embs[b].view(300,-1))).view(-1)
                    # print(obj_score)
                    obj_score /= obj_score.sum() 
                    # print(obj_score)
                    obj_score = obj_score.max()
                    if obj_score > max_score:
                        max_score = obj_score.item()

            all_max_scores.append(max_score)

        print (all_max_scores)
        import sys
        sys.exit()
        return all_max_scores
    


def recall_diff_reward (similarity_tensor, noun_ids, distinct_flag, new_noun_covered):
    print(similarity_tensor.size())
    similarity_tensor = similarity_tensor.clone().detach().cpu().numpy()
    noun_ids = noun_ids.clone().detach().cpu().numpy()
    # get counts
    batch_size = similarity_tensor.shape[0]
    pred_step_count = similarity_tensor.shape[1]
    noun_max_count = noun_ids.shape[1]


    recall_diff_reward = np.zeros((batch_size, pred_step_count))
    # recall_all = np.zeros((batch_size, pred_step_count))
    for bid in range(batch_size):
        for j in range(noun_max_count):
            if int(noun_ids[bid][j]) == 3:
                break
        
        noun_len_to_consider = j
        curr_pred_similarities = similarity_tensor[bid]
        curr_nouns = noun_ids[bid][:noun_len_to_consider]

        den = len(curr_nouns)
        if den == 0:
            # print("No nouns : ", noun_ids[bid])
            continue

        for step_id in range(pred_step_count):
            if int(curr_pred_similarities[step_id]) == 0:
                continue
            if int(distinct_flag[bid][step_id]) == 0 :
                continue
            
            if int(new_noun_covered[bid][step_id]) == 0:
                continue

            
            recall_diff_reward[bid][step_id] = float(1)/ float(den)

    return recall_diff_reward



def precision_diff_reward (similarity_tensor, noun_ids):
    similarity_tensor = similarity_tensor.clone().detach().cpu().numpy()
    noun_ids = noun_ids.clone().detach().cpu().numpy()
    # get counts
    batch_size = similarity_tensor.shape[0]
    pred_step_count = similarity_tensor.shape[1]
    noun_max_count = noun_ids.shape[1]


    prec_diff_reward = np.zeros((batch_size, pred_step_count))
    prec_all = np.zeros((batch_size, pred_step_count))

    for bid in range(batch_size):
        for j in range(noun_max_count):
            if int(noun_ids[bid][j]) == 3:
                break
        
        noun_len_to_consider = j
        curr_pred_similarities = similarity_tensor[bid]
        curr_nouns = noun_ids[bid][:noun_len_to_consider]

        for step_id in range(pred_step_count):
            temp = curr_pred_similarities[:step_id+1]
            num = np.sum(temp)
            den = step_id + 1
            precision = float(num)/float(den)
            # if bid == 0:
            #     print(precision)
            prec_all[bid][step_id] = precision
            if step_id > 0:
                prec_diff_reward[bid][step_id] = prec_all[bid][step_id] - prec_all[bid][step_id-1]
            else :
                prec_diff_reward[bid][step_id] = prec_all[bid][step_id]

    return prec_diff_reward


def precision_diff_reward (similarity_tensor, noun_ids):
    similarity_tensor = similarity_tensor.clone().detach().cpu().numpy()
    noun_ids = noun_ids.clone().detach().cpu().numpy()
    # get counts
    batch_size = similarity_tensor.shape[0]
    pred_step_count = similarity_tensor.shape[1]
    noun_max_count = noun_ids.shape[1]


    prec_diff_reward = np.zeros((batch_size, pred_step_count))
    prec_all = np.zeros((batch_size, pred_step_count))

    for bid in range(batch_size):
        for j in range(noun_max_count):
            if int(noun_ids[bid][j]) == 3:
                break
        
        noun_len_to_consider = j
        curr_pred_similarities = similarity_tensor[bid]
        curr_nouns = noun_ids[bid][:noun_len_to_consider]

        for step_id in range(pred_step_count):
            temp = curr_pred_similarities[:step_id+1]
            num = np.sum(temp)
            den = step_id + 1
            precision = float(num)/float(den)
            # if bid == 0:
            #     print(precision)
            prec_all[bid][step_id] = precision
            if step_id > 0:
                prec_diff_reward[bid][step_id] = prec_all[bid][step_id] - prec_all[bid][step_id-1]
            else :
                prec_diff_reward[bid][step_id] = prec_all[bid][step_id]

    # print(prec_diff_reward[0])
    # import sys
    # sys.exit()
    return prec_diff_reward


def precision_reward_new (similarity_tensor, noun_ids, distinct_flag, mask):
    similarity_tensor = similarity_tensor.clone().detach().cpu().numpy()
    noun_ids = noun_ids.clone().detach().cpu().numpy()
    # get counts
    batch_size = similarity_tensor.shape[0]
    pred_step_count = similarity_tensor.shape[1]
    noun_max_count = noun_ids.shape[1]


    prec_reward = np.zeros((batch_size, pred_step_count))
    
    for bid in range(batch_size):
        for j in range(noun_max_count):
            if int(noun_ids[bid][j]) == 3:
                break
        
        noun_len_to_consider = j
        len_to_consider = float(mask[bid].sum())
        curr_pred_similarities = similarity_tensor[bid]
        curr_nouns = noun_ids[bid][:noun_len_to_consider]

        for step_id in range(pred_step_count):
            # if not distinct continue
            if int(distinct_flag[bid][step_id]) == 0:
                continue
            # else see if similar. if not penalise
            if int(curr_pred_similarities[step_id]) == 0:
                prec_reward[bid][step_id] = -1
        
        prec_reward[bid] /=  len_to_consider

    # print(prec_diff_reward[0])
    # import sys
    # sys.exit()
    return prec_reward



def bigram_util(steps):
    d = {}
    l = len(steps)
    for i in range(l-1):
        curr_bigram = (steps[i], steps[i+1])
        if curr_bigram not in d.keys():
            d[curr_bigram] = 1
        else:
            continue
        for j in range(i+1, l-1):
            temp = (steps[j], steps[j+1])
            if temp in d.keys():
                d[temp] += 1
            
    return d



def repition_reward(steps, id2step):
    # calculate repiting bigrams
    repition_reward = np.zeros(steps.shape)
    num_steps = steps.shape[1]
    for bid in range(repition_reward.shape[0]):
        l = 0
        if id2step[steps[bid][0]] == "<eop>":
            continue
        for i in range(1, num_steps):
            if id2step[steps[bid][i]] == "<eop>":
                l = i+1
            curr_len = i+1
            curr_step_seq = steps[bid][:i+1]

            d = bigram_util(curr_step_seq)
            num = 0
            b = (steps[bid][i-1], steps[bid][i])
            if d[b] > 1 :
                repition_reward[bid][i] = -1 

        if l == 0:
            l = num_steps
        repition_reward[bid] /= l

    return repition_reward



def recall_from_expert_program(pred_steps, mask_pred, expert_steps, mask_expert):
    batch_size = int(pred_steps.size(0))
    pred_total_length = int(pred_steps.size(1))
    reward = np.zeros((batch_size, pred_total_length))
    for bid in range(batch_size):
        pred_set = set()
        expert_set = set()

        len_pred = int(mask_pred[bid].sum())
        len_expert = int(mask_expert[bid].sum())
        pred_steps_curr = pred_steps[bid][:len_pred]
        expert_steps_curr = expert_steps[bid][:len_expert]

        for z in range(len_expert):
            expert_set.add(int(expert_steps_curr[z]))
        den = len(expert_set)
        for z in range(len_pred):
            if int(pred_steps_curr[z]) in expert_set:
                # reward if not rewarded before
                if int(pred_steps_curr[z]) not in pred_set : 
                    reward[bid][z] = 1
                    pred_set.add(int(pred_steps_curr[z]))
        
        reward[bid] /= den

    return reward
            


def edit_distance_reward_stepwise(pred_steps, mask_pred, expert_steps, mask_expert):
    raise NotImplementedError

    batch_size = mask_pred.size(0)
    max_steps = pred_steps.size(1)
    reward = np.zeros(batch_size, max_steps)
    for bid in range(batch_size):
        pred_prog = pred_steps[bid]
        mask_pred_curr = mask_pred[bid]

        expert_prog = expert_steps[bid]
        mask_expert_curr = mask_expert[bid]
