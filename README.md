# Mapping Language to Programs using Multiple Reward Components with Inverse Reinforcement Learning

Pytorch code for the paper "Mapping Language to Programs using Multiple Reward Components with Inverse Reinforcement Learning" accepted at Findings of EMNLP 2021.

# Instructions

Before training a model, mention the specifications in the config file (`config.py`) and trainer file (`train_model.py`)
1. Type of training: `self.training_procedure = "SCST"` for RL/IRL training or `self.training_procedure = "CE"` for MLE training
2. Specify set of rewards, for example, `self.list_of_rewards = ["prec", "rec_diff", "rep", "LCS", "rec_prog"]`.
3. Specify the size of training data to be used in the trainer file. For example `all_train_samples = json.load(open("./disjoint_data/train_features_70_samples.json", "r"))` says to use 70 labeled examples to train 

Similarly before testing specify val or test split in the tester file(`test_model.py`):
`self.test_pair = json.load(open("./disjoint_data/val_features_185_samples.json"))` or `self.test_pair = json.load(open("./disjoint_data/test_features_500_samples.json"))`


To start training:

`python main.py train`

To test:

`python main.py test 3`
(here 3 denotes the beam size)
