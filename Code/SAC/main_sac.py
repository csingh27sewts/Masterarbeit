# the following 3 lines are helpful if you have multiple GPUs and want to train
# agents on multiple GPUs. I do this frequently when testing.
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np
import os
import torch as T

if __name__ == '__main__':
    #env_id = 'LunarLanderContinuous-v2'
    #env_id = 'BipedalWalker-v2'
    #env_id = 'AntBulletEnv-v0'
    env_id = 'InvertedPendulumBulletEnv-v0'
    #env_id = 'CartPoleContinuousBulletEnv-v0'
    # Here are all the environments defined 
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/__init__.py
    env = gym.make(env_id)
    print(env.observation_space.shape)
    # SAC Agent
    # Defines the Actor Network, Critic Network 1, Critic Network 2, Value Network, Target Network
    # observation_space = [-inf -inf -inf -inf -inf] to [inf inf inf inf inf]
    # action_space = [-1, 1]
    # 
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=256, layer1_size=256, layer2_size=256,
                n_actions=env.action_space.shape[0])
    n_games = 250

    # Define filename to store plots
    filename = env_id + '_'+ str(n_games) + 'games_scale' + str(agent.scale) + \
                    '_clamp_on_sigma.png'
    figure_file = 'SAC/plots/' + filename
    figure_file = os.path.join(os.getcwd(), figure_file)

    # best_score initialized as lower limit of reward range (-inf, inf) , i.e. best_score = -inf
    best_score = env.reward_range[0]
    score_history = []
    
    load_checkpoint = True
    # display 
    if load_checkpoint:
        # Loads checkpoints of Actor Network, Critic Network 1, Critic Network 2, Value Network, Target Network
        agent.load_models()
        env.render(mode='human')
    steps = 0
    for i in range(n_games):
        observation = env.reset()
        # Observation is a 5 tuple 
        # (x x_dot cos(theta) sin(theta) theta_dott)
        # where
        # theta = angle of the cart pole with the base
        # theta_dot = angular velocity
        # x = position of base
        # x_dot = linear velocity of base
        print("Observation : (x x_dot cos(theta) sin(theta) theta_dott) \n")
        print(observation)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            print("Step")
            print(action, " ", observation_, " ", reward, done, info)
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score,
                'trailing 100 games avg %.1f' % avg_score, 
                'steps %d' % steps, env_id, 
                ' scale ', agent.scale)
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
