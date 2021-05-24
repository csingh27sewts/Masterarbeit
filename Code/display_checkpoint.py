import torch as T
from SAC.sac_torch import Agent
from SAC.networks import ActorNetwork, CriticNetwork, ValueNetwork

checkpoint_file = '/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/cloth_test/cloth_corner/checkpoint_tmp/ckpt_actor_9999.ckpt'
ckpt_actor = T.load(checkpoint_file)
print(ckpt_actor)
