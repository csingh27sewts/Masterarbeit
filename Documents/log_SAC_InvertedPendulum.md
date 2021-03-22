# Masterarbeit


SAC Code  

env_id = 'InvertedPendulumBulletEnv-v0'


Initialize the Actor Network, Critic Network 1, Critic Network 2, Value Network, Target Network

Observation is a 5 tuple 
(x x_dot cos(theta) sin(theta) theta_dott)
where
theta = angle of the cart pole with the base
theta_dot = angular velocity
x = position of base
x_dot = linear velocity of base

Observations are fed as input to Actor Network
Output is Normal distribution representing actions
Sample from the distribution
ReLu activation to get action

Take action
Note observation, reward, info 
keep noteof score which is the cumulation of rewards
update observations


state, new state, action and reward are stored in the memory
random batches of size 100 are selected from the memory
state, new state, action and reward from these batches are selected
convert to tensors

Value of a state and target value of a new state are calculated by feeding them to Value Networks. Both these are sampled from batch size

States are fed as input to Actor Network
Output is Normal distribution representing actions
Sample from the distribution
ReLu activation to get action
Get Q values by feeding states and actions to the critic networks

value_target = critic_value - log_probs
value loss, actor loss and critic loss are calculated

https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#sac

