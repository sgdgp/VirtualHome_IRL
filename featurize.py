import json
import numpy as np
import spacy
import re
import gensim
import sys
sys.path.append("./virtualhome/simulation/")
sys.path.append("./virtualhome/dataset_utils/")
import add_preconds
import evolving_graph.check_programs as check_programs
from tqdm import tqdm
import random


# parse all data files in VirtualHome and form a json file 
# keys : 
#   1) tokens (words in task description)
#   2) program
#   3) environment
#   4) nouns (use spacy)
#   5) path (path of the program)
#  Note : in the original publicly avaialable VirtualHome repository, path of program and precondition path of program have one to one correspondence
#           that is, given the path, replacing a few words in the path will lead to preconditon file path. Please maintain this correspondence
#   6) np_concat (concat all noun phrases in description)




all_data_train = json.load(open("data.json","r")) # "data.json" is a placeholder, replace accordingly

max_len = 27
features = [] 




beginning_room_list = ["living_room", "entrance_hall",  "bathroom", "bedroom", "dining_room", "home_office", "kitchen"]

word2id = json.load(open("./disjoint_data/word2id.json", "r"))
step2id = json.load(open("./disjoint_data/step2id.json", "r"))
step_embedding_matrix = np.load("./disjoint_data/step_embedding_matrix.npy")
step_embedding_matrix_copy = step_embedding_matrix.copy()
all_object2id_extended_graph = json.load(open("./disjoint_data/all_object2id_map_extended_graph.json","r"))
action2id = json.load(open("./disjoint_data/action2id.json", "r"))
object2id = json.load(open("./disjoint_data/object2id.json", "r"))
# bop eop
if "<bop>" not in step2id.keys():
    step2id["<bop>"] = len(step2id)
    step_embedding_matrix = np.vstack([step_embedding_matrix, np.random.normal(scale=0.6, size=(300, ))])

if "<eop>" not in step2id.keys():
    step2id["<eop>"] = len(step2id)
    step_embedding_matrix = np.vstack([step_embedding_matrix, np.random.normal(scale=0.6, size=(300, ))])


list_len = []

for d in all_data:
    tokens = d["tokens"]
    nouns = d["nouns"]
    token_list = [ word2id[t] for t in tokens]
    nouns_list = [ word2id[t] for t in nouns ]
    np_concat = d["np_concat"].split()
    np_concat_list = []
    
    for x in np_concat:
        y = x.split(".")[0]
        y = y.split(",")[0]
        y = y.split("'")[0]
        y = y.replace('"', '')
        y = y.replace('(', '')
        if "-" in y or "/" in y:
            y = y.replace("-"," ")
            y = y.replace("/"," ")
            y = y.split()
            for _y in y :
                np_concat_list.append(_y)
        else:
            np_concat_list.append(y) 
    np_concat_list = [word2id[t] for t in np_concat_list]
    np_concat_list = list(set(np_concat_list))
    list_len.append(len(np_concat_list))
    
    
    temp_len = len(nouns_list)
    for _ in range(max_len - temp_len):
        nouns_list.append(word2id["<invalid>"])
    step_list = [step2id["<bop>"]]
    prog = d["program"]
    for p in prog:
        t1 = re.findall(r"\[([A-Za-z0-9_]+)\]", p)
        t2 = re.findall(r"\<([A-Za-z0-9_]+)\>", p) # objects

        step_key = ""
        assert (len(t1) == 1)
        for _t1 in t1:
            step_key = _t1
            
        for _t2 in t2:
            step_key += "||" + _t2
            
        step_list.append(step2id[step_key])
    step_list.append(step2id["<eop>"])


    entities_mask_list = [] # not used anywhere in the model, can be left as empty list
    
    
    beginning_mask_list = [] # not used anywhere in the model, can be left as empty list
    


    path = d["path"]
    env = d["environment"]
    sp_val = [] # not used anywhere in the model, can be left as empty list
    features_single = [token_list, nouns_list, step_list, entities_mask_list, beginning_mask_list, path, env, sp_val, np_concat_list]
    features.append(features_single)

json.dump(features, open("features.json", "w"))   # save accordingly
