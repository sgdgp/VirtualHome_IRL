import numpy as np
import json
import os
import re
import sys
# SIM_FOLDER = "./"
# DATA_UTILS_FOLDER = "./" 
# sys.path.append(SIM_FOLDER)
# sys.path.append(DATA_UTILS_FOLDER)

# the following two imports will work after the simulation folder is included
# import add_preconds
# import evolving_graph.check_programs as check_programs

def convert_to_one_hot(id, total):
    a = np.zeros(total)
    a[id] = 1
    return a

    
def get_prog_from_step_ids(steps, d):
    prog = []
    # print(d)
    for step in steps:
        p = d[step]
        p = p.split("||")
        temp = "["+ p[0] +"] "
        for obj in p[1:] :
            temp += "<" + obj + "> (1) "
        temp = temp.strip()
        prog.append(temp)
    return prog


def get_word_similarity(word1, word2, word2id, embedding_matrix):
    word1 = word1.split("_")
    word2 = word2.split("_")
    v1 = []
    v2 = []
    for w in word1 :
        try:
            t = embedding_matrix[word2id[w]]
            v1.append(t)
        except :
            t = np.zeros((embedding_matrix.shape[1]))
            v1.append(t)
    
    for w in word2 :
        try:
            t = embedding_matrix[word2id[w]]
            v2.append(t)
        except :
            t = np.zeros((embedding_matrix.shape[1]))
            v2.append(t)

    vec1 = np.average(np.array(v1))
    vec2 = np.average(np.array(v2))
    

    cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    
    return cos_sim

def step_similarity_score(step_id1, step_id2, id2action_object_pair, word2id,  m):
    step1 = id2action_object_pair[step_id1].split("||")
    step2 = id2action_object_pair[step_id2].split("||")

    x = min(len(step1), len(step2))
    a = 0
    for i in range(x):
        a += get_word_similarity(step1[i], step2[i],word2id, m)
    
    a /= float(x)
    return a



def convert_step_ids_to_text_by_template(steps, config):
    retval = ""
    
    id2action_object_pair = config.id2action_object_pair
    for step in steps:
        step_text = id2action_object_pair[step]
        retval += convert_using_template(step_text) + " "

    retval = retval.strip().lower()
    return retval


def convert_word_ids_to_text(gt_word_ids, config) :
    # print(gt_word_ids)
    id2word = json.load(open(config.id2word_dict_path, "r"))

    text = ""
    for id in gt_word_ids:
        # print(id)
        if int(id) <0 :
            continue

        if int(id) >= len(id2word): 
            continue
        word = id2word[str(int(id))] 
        word = word.strip(",").strip()
        if word != "<bos>" and word != "<eos>" and word != "<pad>":
            text += word + " "
    
    return text.strip().lower()


def get_nouns_from_word_ids(gt_word_ids, config):
    gt_text = convert_word_ids_to_text(gt_word_ids, config)
    import spacy
    nlp = spacy.load("en_core_web_sm")

    gt_text.replace(",", " ,")
    gt_text.replace(".", " .")
    gt_text.replace("?", " ?")
    gt_text.replace("!", " !")

    processed = nlp(gt_text)
    pos = [token.pos_ for token in processed]
    return_list = []
    for i in range(len(pos)):
        if pos[i] == "NOUN":
            return_list.append(str(processed[i].text))
    
    return return_list

def convert_noun_list_to_vector(noun_list, emb_matrix, config):
    word2id = json.load(open(config.word2id_dict_path, "r"))
    vec_list = []
    for noun in noun_list:
        try : 
            id = int(word2id[noun])
            vec = emb_matrix[id]
            vec_list.append(vec)
        except:
            vec_list.append(np.zeros((emb_matrix.shape[1])))
    
    return vec_list


def execution_check_v1(id_list, preconds_path = None, graph_env = None, id2action_object_pair = None):
    if str(id2action_object_pair[int(id_list[-1])]) == "<eop>":
        return True, []
        
    prog = get_prog_from_step_ids(id_list, id2action_object_pair)
    preconds = json.load(open(preconds_path,"r"))

    env2path = {
                "tts1" : "./example_graphs/TrimmedTestScene1_graph.json",
                "tts2" : "./example_graphs/TrimmedTestScene2_graph.json",
                "tts3" : "./example_graphs/TrimmedTestScene3_graph.json",
                "tts4" : "./example_graphs/TrimmedTestScene4_graph.json",
                "tts5" : "./example_graphs/TrimmedTestScene5_graph.json",
                "tts6" : "./example_graphs/TrimmedTestScene6_graph.json",
                "tts7" : "./example_graphs/TrimmedTestScene7_graph.json"
                }
    graph_input = json.load(open(env2path[graph_env],"r"))
    id_mapping = {}
    # modify the preconds path accordingly to the initstate folder
    info = check_programs.check_script(prog, preconds, graph_path=None, inp_graph_dict=graph_input, id_mapping=id_mapping)
    message, final_state, graph_state_list, graph_dict, id_mapping, info, helper, modif_script = info
    exec = (message == 'Script is executable')
    if info is None:
        node_name_list = [[]]
    else:
        try : 
            node_name_list = info["node_name_list"]
        except:
            node_name_list = [[]]
    if node_name_list is not None:
        return exec, node_name_list[-1]
    else:
        return exec, []

class Hypothesis:
    def __init__(self, id_list, last_hidden_cell, attn_alpha_list, score, id_in_batch, graph_state= None):
        self.id_list = id_list
        self.last_hidden_cell = last_hidden_cell
        self.attn_alpha_list = attn_alpha_list
        self.score = score
        self.id_in_batch = id_in_batch
        self.graph_state = graph_state

class Hypothesis_env:
    def __init__(self, id_list, last_hidden_cell, last_hidden_cell_env,
                    last_hidden_cell_combined,
                     attn_alpha_list, 
                    switching_prob_list=[], score=0, id_in_batch=-1, 
                    beginning_room_entities = None, curr_node_list= None,
                    preconds_file_path = None,
                    environment = None):

        self.id_list = id_list
        self.last_hidden_cell = last_hidden_cell
        self.last_hidden_cell_env = last_hidden_cell_env
        self.last_hidden_cell_combined = last_hidden_cell_combined
        self.attn_alpha_list = attn_alpha_list
        self.score = score
        self.id_in_batch = id_in_batch
        # self.graph_state = graph_state
        self.switching_prob_list = switching_prob_list
        
        self.beginning_room_entities = beginning_room_entities
        self.curr_node_list = curr_node_list
        self.preconds_file_path = preconds_file_path
        self.environment = environment


def get_env_mask(nodelist):
    entity2id = json.load(open("./disjoint_data/all_object2id_map_extended_graph.json","r"))
    # replace above path if needed appropriately
    mask = np.zeros((len(entity2id)))
    for n in nodelist:
        mask[int(entity2id[n])] = 1
    
    return mask