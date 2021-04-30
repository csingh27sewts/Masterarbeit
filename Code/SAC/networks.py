import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import os

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
            name, chkpt_dir=os.path.join(os.getcwd(),'SAC/tmp/sac')):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac.ckpt')
        print("Checkpoint file")
        print(self.checkpoint_file)
        # I think this breaks if the env has a 2D state representation
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action,
            n_actions, name, chkpt_dir=os.path.join(os.getcwd(),'SAC/tmp/sac')):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac.ckpt')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        
        # Adding tanh to make mu in the range of [-1, 1]
        mu = T.tanh(mu)

        sigma = self.sigma(prob)

        # Clamping sigma values between the range min and max, if values are lower than reparam noise, initialize them to reparam noise
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1) 
        # authors use -20, 2 -> doesn't seem to work for my implementation

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):

        # Feed the state to the model adn get mu and sigma
        # mu and sigma each have size equal to no. of actions, in cloth_v0 it is 12
        # state = state * 100
        mu, sigma = self.forward(state)
        probabilities = T.distributions.Normal(mu, sigma)
        # print("PROBABILITIES\n")
        # print(probabilities)
        # print(mu, sigma)
        # Creates a normal distribution from mean and variance
        # https://pytorch.org/docs/stable/distributions.html
        
        if reparameterize:
            actions = probabilities.rsample() # reparameterizes the policy
        else:
            actions = probabilities.sample()
        # https://www.youtube.com/watch?v=MswxJw-8PvE
        # https://www.youtube.com/watch?v=QrOsBIn1Gto
        
        # rsample stores gradients, sample does not
        # print(T.tensor(self.max_action))
        
        # Why is this multiplied with max action which is [1,1,1,1,1,1,1,1,1,1,1,1]
        # multiplying the sampled action with all of the actions
        # tensor([1.3426, 1.3426, 1.3426, 1.3426, 1.3426, 1.3426, 1.3426, 1.3426, 1.3426,
        # 1.3426, 1.3426, 1.3426])
        # This is because it was defined for the original enironment InvertedPendulum
        # Which has action as just one value. Max_action = 1 in this case
        # But it is also ok in our case
        # I think this is defined for the case when the max_action is not equal to 1 
        # since for max_action = 1 it is not needed
        action = T.tensor(actions) * T.from_numpy(self.max_action).float().to(self.device)
        # print("ACTIONS")
        # print(mu,sigma)
        # print(probabilities)
        # print(action)
        # action = action * 100
        # Take log_probs of action, i.e. log(p(a|pi_theta(s))
        log_probs = probabilities.log_prob(actions)
        print("LOG PROBS STEP 1")
        print(log_probs)

        # log_prob = log_prob(actions) - log(1 - action^2 + reparam_noise)
        # Where does this formula come from and what is its significance ?
        # log_probs = T.log(1-action.pow(2) + self.reparam_noise)

        print("LOG PROBS STEP 2")
        print(log_probs)

        log_probs = log_probs.sum(-1, keepdim=True)

        print("LOG PROBS STEP 3")
        print(log_probs)

        # log_probs = T.tensor([float(-1e9)]*len(actions))
        # log_probs = log_probs / int(1e8)
        # action = action / 100
        return action, log_probs

    def sample_mvnormal(self, state, reparameterize=True):
        """
            Doesn't quite seem to work.  The agent never learns.
        """
        # state = state * 100
        mu, sigma = self.forward(state)
        n_batches = sigma.size()[0]

        cov = [sigma[i] * T.eye(self.n_actions).to(self.device) for i in range(n_batches)]
        cov = T.stack(cov)
        probabilities = T.distributions.MultivariateNormal(mu, cov)

        if reparameterize:
            actions = probabilities.rsample() # reparameterizes the policy
        else:
            actions = probabilities.sample()
        # actions = actions * 100
        # action = T.tanh(actions) # enforce the action bound for (-1, 1)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.sum(T.log(1-action.pow(2) + self.reparam_noise))
        log_probs = log_probs.sum(-1, keepdim=True)
        
        # actions = actions / 100
        return actions, log_probs
        # return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
            name, chkpt_dir='SAC/tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        chkpt_dir = os.path.join(os.getcwd(),chkpt_dir)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        # v = T.tensor([1., 1.])
        print("STATE  !!!!!!!!!!!!!!!!!!!!")
        print(state)
        print("STATE VALUE !!!!!!!!!!!!!!!!!!!!")
        print(state_value)
        print("VALUE !!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(v)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

