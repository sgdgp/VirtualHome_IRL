from config import Config


from train_model import *
from test_model import *

import os
import random
import numpy as np
import torch

from model import *




def main_desc2steps():
    # Setting random seeds
    seed_val = 123
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)

    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    config = Config()

    datareader = None
    
    trainer = Trainer(dataloader = datareader, config=config)
    # print(sys.argv[1])
    
    if sys.argv[1] == "train":

        trainer.train_and_save(400)

        print("Training completed, starting test....")
    
    else :
        # create a Tester object
        # print(sys.argv)
        if str(sys.argv[2]) == "random_sampling":
            mode = "test_random_sampling"
        elif str(sys.argv[2]) == "random_retrieval":
            mode = "test_random_retrieval"
        elif str(sys.argv[2]) == "noun_grounding_preprocess":
            mode = "test_noun_grounding_preprocess"
        else :
            mode = "test"

        if len(sys.argv) >= 4:
            config.rev_map_score_weight = float(sys.argv[3])
            print("Rev map score weight = ", config.rev_map_score_weight)

        else:
            config.use_grounding = False
        tester = Test_MODEL(data=datareader, encoder=trainer.encoder, decoder = trainer.decoder, config=config, mode=mode)
        return tester.evaluate()


if __name__ == "__main__":
    task = "desc2steps"
    main_desc2steps()
    