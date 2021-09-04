import os

class Config:
    def __init__(self):
        self.simulator_folder = "./simulation/" 
        
        self.action2id_dict_path = "./disjoint_data/action2id.json"
        self.object2id_dict_path = "./disjoint_data/object2id.json"
        

        self.write_processed_data = True

        self.data_parallel = True

        self.embedding_name = "word2vec"
       
        self.task_name = "desc2steps"
        
        self.training_procedure = "SCST"
        # self.training_procedure = "CE"
        self.np_context = False
        self.irl_technique = "MaxEnt"
        self.list_of_rewards = ["prec", "rec_diff", "rep", "LCS", "rec_prog"]
        # self.list_of_rewards = ["rep", "LCS"]
        # self.list_of_rewards = ["LCS", "rec_prog"]
        # self.list_of_rewards = ["LCS", "exec"]
        self.num_rewards = len(self.list_of_rewards)

        self.reward_index = {}
        for i in range(len(self.list_of_rewards)):
            self.reward_index[self.list_of_rewards[i]] = i 
            
        self.IRL_sample_count = 5
        self.do_irl = True
        self.irl_loading = False

        self.teacher_forcing_ratio = 1
        self.learning_rate = 1e-3

        

        self.reward_LCS = False
        self.reward_BLEU = False

        self.grounding_score_weight = 1.0
        self.rev_map_score_weight = 1.0
        
        self.use_grounding = False

       
        
        if self.embedding_name == "word2vec":
            

            self.embedding_dim = 300
            self.embedding_dim_decoder = 300
            
            self.batch_size = 256
           
            self.step_vec_type = "word2vec"
        
        # replace model paths accordingly
        self.model_folder_path = "./trained_model_irl/"    
        self.model_folder_path_load = "./trained_model/"    
        self.all_test_tuples_filename = "./predictions.json"

        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
            
        self.encoder_model_path = self.model_folder_path + "encoder_ce_{}ep.pth"
        self.decoder_model_path = self.model_folder_path + "decoder_ce_{}ep.pth"
        self.encoder_optimizer_path = self.model_folder_path + "encoder_optimizer_ce_{}ep.pth"
        self.decoder_optimizer_path = self.model_folder_path + "decoder_optimizer_ce_{}ep.pth"

        
        self.enable_model_load = True
        self.encoder_model_path_load = self.model_folder_path_load + "encoder_ce_50ep.pth"
        self.decoder_model_path_load = self.model_folder_path_load + "decoder_ce_50ep.pth"
        self.encoder_optimizer_path_load = self.model_folder_path_load + "encoder_optimizer_ce_50ep.pth"
        self.decoder_optimizer_path_load = self.model_folder_path_load + "decoder_optimizer_ce_50ep.pth"

        # copy all files to keep track, useful if needed to test/re-train a certain model if a lot has been modified later
        # and this way easier to restore back to that setting
        from glob import glob
        dest_path = self.model_folder_path + "py_files/"
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for f in glob("./*.py"):
            command = "cp " + f + " " + dest_path
            os.system(command)