import sys
from utils import *
import warnings
import json

def LCS(pred,gt):
    # pred and gt are sequence of act-obj ids 
    m = len(pred) 
    n = len(gt) 
    
    # print("pred : ", pred)
    # print("gt : ", gt)

    LCS = [[None]*(n + 1) for i in range(m + 1)] 
  
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                LCS[i][j] = 0
            elif int(pred[i-1]) == int(gt[j-1]): 
                LCS[i][j] = LCS[i-1][j-1]+1
            else: 
                LCS[i][j] = max(LCS[i-1][j], LCS[i][j-1]) 
  
    # print(LCS)
    retval = float(LCS[m][n]) / (float)(max(m,n))
    # print(retval)
    return retval

def LCS_with_word_similarity(pred,gt, id2action_object_pair, word2id, embedding_matrix):
    # pred and gt are sequence of act-obj ids 
    m = len(pred) 
    n = len(gt) 
    
    # print("pred : ", pred)
    # print("gt : ", gt)

    LCS = [[None]*(n + 1) for i in range(m + 1)] 
  
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                LCS[i][j] = 0
            elif pred[i-1] == gt[j-1] or step_similarity_score(pred[i-1], gt[j-1], id2action_object_pair,word2id, embedding_matrix) > 0.8 : 
                LCS[i][j] = LCS[i-1][j-1]+1
            else: 
                LCS[i][j] = max(LCS[i-1][j], LCS[i][j-1]) 
  
    # print(LCS)
    retval = float(LCS[m][n]) / (float)(max(m,n))
    # print(retval)
    return retval




def LCS_A_O_S(pred,gt, config):
    # get LCS for action, object and step
    step_LCS = LCS(pred, gt)

    action2id = json.load(open(config.action2id_dict_path, "r"))
    object2id = json.load(open(config.object2id_dict_path, "r"))

    
    pred_a = []
    pred_obj1 = []
    pred_obj2 = []
    gt_a = []
    gt_obj1 = []
    gt_obj2 = []

    id2action_object_pair = config.id2action_object_pair
    for step_id in pred:
        step = id2action_object_pair[int(step_id)]
        if step == "<eop>" or step == "<bop>":
            continue
        s = step.split("||")
        pred_a.append(int(action2id[s[0]]))
        if len(s) >= 2:
            pred_obj1.append(int(object2id[s[1]]))
        else:
            pred_obj1.append(-1)

        if len(s) == 3:
            pred_obj2.append(int(object2id[s[2]]))
        else:
            pred_obj2.append(-1)

    for step_id in gt:
        step = id2action_object_pair[int(step_id)]
        if step == "<eop>" or step == "<bop>":
            continue
        s = step.split("||")
        gt_a.append(int(action2id[s[0]]))
        if len(s) >= 2:
            gt_obj1.append(int(object2id[s[1]]))
        else:
            gt_obj1.append(-1)

        if len(s) == 3:
            gt_obj2.append(int(object2id[s[2]]))
        else:
            gt_obj2.append(-1)

    # print(pred_obj1)
    # print(gt_obj1)
    act_lcs = LCS(pred_a, gt_a)
    obj1_lcs = LCS(pred_obj1, gt_obj1)
    obj2_lcs = LCS(pred_obj2, gt_obj2)
    obj_lcs = 0.5 * (obj1_lcs + obj2_lcs)

    # return act_lcs, obj_lcs, step_LCS
    return act_lcs, obj1_lcs, step_LCS

def LCS_batch_train(pred_batch, gt_batch, mask_pred, mask_gt):
    batch_size = pred_batch.shape[0]
    mask_pred = np.sum(mask_pred, 1)
    mask_gt = np.sum(mask_gt, 1)
    # print(mask_pred.shape)
    # print(mask_gt.shape)
    
    score = []
    for i in range(batch_size):
        pred = pred_batch[i][:int(mask_pred[i])]
        gt = gt_batch[i][:int(mask_gt[i])]
        score.append(LCS(pred,gt))

    return np.array(score) # B size array

def activity_acc(pred, gt, m, lcs_based = False):
    # print("LCS_Based = ", lcs_based)
    
    if not lcs_based :
        warnings.warn('Use LCS based action accuracy, one-to-one matching not recommended')
        # one to one matching, not recommended
        count = 0
        min_len = min (len(pred), len(gt))
        max_len = max (len(pred), len(gt))
        # print ("gt = ", gt)
        # print("pred = ", pred)
        for i in range(min_len):
            p = m[int(pred[i])]
            g = m[int(gt[i])]
            p = p.split("||")
            g = g.split("||")
            p_act = str(p[0]).strip().lower()
            g_act = str(g[0]).strip().lower()
            
            if p_act == g_act:
                count += 1
        retval = (float)(count)/(float)(min_len)


    else :
        # LCS based action accuracy, recommended
        l_pred = len(pred) 
        l_gt = len(gt) 
        
        act_pred = []
        act_gt = []
        for i in range(l_pred):
            step = m[int(pred[i])]
            step = step.split("||")
            act = str(step[0]).strip().lower()
            act_pred.append(act)

        for i in range(l_gt):
            step = m[int(gt[i])]
            step = step.split("||")
            act = str(step[0]).strip().lower()
            act_gt.append(act)
    
        LCS = [[None]*(l_gt + 1) for i in range(l_pred + 1)] 
    
        for i in range(l_pred + 1): 
            for j in range(l_gt + 1): 
                if i == 0 or j == 0 : 
                    LCS[i][j] = 0
                elif act_pred[i-1] == act_gt[j-1] : 
                    LCS[i][j] = LCS[i-1][j-1]+1
                else: 
                    LCS[i][j] = max(LCS[i-1][j], LCS[i][j-1]) 
    
        # print(LCS)
        retval = float(LCS[l_pred][l_gt]) / (float)(max(l_pred,l_gt))

    return retval



def executability_score(pred,config, id2action_object_pair):
    # true false score
    # return (1/0, message) instead of boolean so that percentage of executable programms can be calculated easily later in test/eval functions
    if len(pred)==0:
        return 1, None
    sys.path.append(config.simulator_folder)
    from unity_simulator.comm_unity import UnityCommunication
    try:
        comm = UnityCommunication()
        # print("pred : ", pred)
        prog = get_prog_from_step_ids(pred, id2action_object_pair)
        # print("Prog")
        # print(prog)
        comm.reset()

        response, message = comm.render_script(prog, capture_screenshot=True, skip_execution=True, gen_vid=False)
        if response:
            return 1,None
        else:
            return 0,message
    except Exception as e:
        print("No simulator running")
        print(e)
        sys.exit()




def edit_distance(pred,gt):
    m = len(pred)
    n = len(gt)

    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
  
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
            elif pred[i-1] == gt[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    retval =  float(dp[m][n])/float(max(m,n))
    return retval
