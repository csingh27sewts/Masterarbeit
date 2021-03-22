import os
import torch as T
import torch.nn.functional as F
import numpy as np
from SAC.buffer import ReplayBuffer
from SAC.networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            env_id, gamma=0.99, 
            n_actions=2, max_size=1000000, layer1_size=256,
            layer2_size=256, batch_size=100, reward_scale=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        print("Actions", n_actions)
        # Initialize actor network
        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name=env_id+'_actor', 
        #                          max_action=env.action_space.high)
                                  max_action=env.action_spec().maximum)
        
        # Initialize critic network 1
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_1')
        
        # Initialize critic network 2
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_2')
       
        # Initialize value network 
        self.value = ValueNetwork(beta, input_dims, layer1_size,
                                      layer2_size, name=env_id+'_value')
        
        # Initialize target value network
        self.target_value = ValueNetwork(beta, input_dims, layer1_size,
                                         layer2_size, name=env_id+'_target_value')

        self.scale = reward_scale
        
        # Initialize Target Value Network parameters same as Value Network parameters
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # Convert to Pytorch Tensor
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        #actions, _ = self.actor.sample_mvnormal(state)
        # actions is an array of arrays due to the added dimension in state
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        # Would not start learning till the ReplayBuffer is filled up to the batch size, 256
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # State is fed to the value network
        # Next state is fed to the target value network
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0
       
        # Get actions by feeding state to Actor Network and log probabilities
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        #actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        # Get Q value by feeding state-action pairs to two critic networks and taking a min of the Q values from both these networks
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        # Loss for training the value network 
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Get actions from a new sample fed into the network
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        #actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Loss for training the actor network
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Loss for training the critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update network parameters for the Target Value Network
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        
        # Tau is time lag
        if tau is None:
            tau = self.tau

        # Get value network and target value network parameters (weights and biases)        
        # Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # Updates weights and biases for the Target Value Network
        # Target value[param] = tau * Value[param] + (1-tau) * Value[param]
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        # Loads the updated parameter dictionary for the Target Value Network
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    
