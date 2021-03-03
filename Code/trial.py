import torch as T
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import SAC.sac_torch
from SAC.sac_torch import Agent
import SAC.networks
import torch.nn as nn
from torch.distributions.normal import Normal

observation = (0, 0, 1, 0, 0)
state = T.Tensor([observation])
mu, sigma = SAC.networks.ActorNetwork.forward(SAC.networks.ActorNetwork(alpha=0.0003,
                input_dims=(5,), fc1_dims=256, fc2_dims=256,max_action = 1,
                n_actions=1, name = "trial"), state)
mu = mu.item()
sigma = sigma.item()
print(mu)
print(sigma)
domain = np.linspace(-2,2,1000)
plt.plot(domain, norm.pdf(domain, mu, sigma))
plt.show()


probabilities = T.distributions.Normal(mu, sigma)

print(probabilities)
actions = probabilities.sample()
print(actions)
action = T.tanh(actions)*T.tensor(1)
log_probs = probabilities.log_prob(actions)
log_probs -= T.log(1-action.pow(2) + 1e-6)
# log_probs = log_probs.sum(1, keepdim=True)
print(log_probs.view(-1))
