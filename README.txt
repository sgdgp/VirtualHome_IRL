Instructions

1. If only interested in training without execution reward use "python main.py train". 
       a) Choose appropriate mode "CE" for supervised and "SCST" for RL or IRL
       b) In config file "do_irl" set to true will perform MaxEnt IRL, otherwise RL
       c) Choose required reward components 
2. For using simulator and exec reward:
       a) Clone publicly available simulator of VirtualHome : https://github.com/xavierpuigf/virtualhome
       b) Mention the appropriate folder paths for simulation and dataset utils in the utils file
       d) featurize the data (from VirtualHome) using featurize.py so that preconditions file is mapped accordingly in the features now. 
       c) Also modify the preconds path to get approprate preconditions
3. For testing "python main.py test 3", 3 is the beam size. choose the trained models from the model folder and update the path in config.
       IRL Model trained on 70 samples using (all - exec) reward is provided.
