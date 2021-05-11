import torch as T
from SAC.sac_torch import Agent
from SAC.networks import ActorNetwork, CriticNetwork, ValueNetwork

checkpoint_file = '/home/chandandeep/Masterarbeit_ws/src/Masterarbeit/Code/output/cloth_sewts_exp2_2/checkpoint_tmp/ckpt_value_920.ckpt'
ckpt_actor = T.load(checkpoint_file)
print(ckpt_actor)
