# Import packages
from PIL import Image, ImageTk
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect
import os
import torch as T
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
import gc
import gym
from stable_baselines3 import SAC
import dmc2gym
import stable_baselines_logging

if __name__ == '__main__':
   
    # Define environment'
    n_games = 1
    # Load environment

    # EXPERIMENT #0
    # env_id = 'pendulum'
    # env = gym.make('Pendulum-v0')   
    
    # EXPERIMENT #1
    # env_id = 'cloth_sewts_minimal'
    # env = dmc2gym.make(domain_name='cloth_sewts_minimal_1', task_name='easy', seed=1) #dmc2gym package
    
    # EXPERIMENT #2
    env_id = 'cloth_sewts_minimal'
    env = dmc2gym.make(domain_name='cloth_sewts_minimal_2', task_name='easy', seed=1) #dmc2gym package
    
    # EXPERIMENT #3
    # env_id = 'cloth_sewts_minimal'
    # env = dmc2gym.make(domain_name='cloth_sewts_minimal_3', task_name='easy', seed=1) #dmc2gym package

    # create output folder for current experiment
    path_to_output = os.getcwd() + '/output'
    os.chdir(path_to_output)
    observation = env.reset()
    model = SAC("MlpPolicy", env, learning_starts=1024, 
                tensorboard_log='/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4',
                batch_size=256, verbose=1)
    step = 0
    reward = 0
    for i in range(n_games):
        print("INSIDE LOOP") 
        # Reset environment
        # print(env.reset())
        done = False
        print("GAME no.",i,"/n/n/n")
        # for init_step in range(500):
        #    env.step(np.array([0.,0.])) # for cloth_corner
        # Change directory to current environment path
        print("LEARNING .../n")
        model.set_env(env)
        observation = env.reset()     
        
        # model.learn(total_timesteps=1000000, reset_num_timesteps = True, callback = stable_baselines_logging.ImageRecorderCallback(), log_interval=1) # log_interval = no. of episodes
        # timesteps = log_interval * 2000
        
        print("/nLEARNT")
        
        # EXPERIMENT #0 
        # model.save("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_3/cloth_sewts_minimal")
        # model = SAC.load("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_1/pendulum")
        
        # EXPERIMENT #1
        # model.save("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_3/cloth_sewts_minimal")
        # model = SAC.load("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_2/cloth_sewts_minimal")
        
        # EXPERIMENT #2
        # model.save("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_3/cloth_sewts_minimal")
        model = SAC.load("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_3/cloth_sewts_minimal")
        
        # EXPERIMENT #3
        # model.save("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_3/cloth_sewts_minimal")
        # model = SAC.load("/home/chandandeep/Project/Masterarbeit/src/Masterarbeit/Code/output/sac_baselines_4/SAC_4/cloth_sewts_minimal")
 

        while done is False :
            # Move to game folder
            # observation = env.reset()
            # print(observation)
            # observation = np.array([1.,1.,0.])
            action, observation_ = model.predict(observation, deterministic=True)
            step = step + 1
            print("STEP",step,"/n")
            # while not done:
            # action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print("Evaluating")
            print("observation")
            print(obs)
            print("action")
            print(action)
            print("reward")
            print(reward)
            observation = obs
            if step == 500: # No. of steps should be > batch size of 250 as the agent learns only when the batch is full
                done = True
                obs = env.reset()
            # image = env.render()
            # "HWC" specify the dataformat of the image, here channel last
            # (H for height, W for width, C for channel)
            # See https://pytorch.org/docs/stable/tensorboard.html
            # for supported formats
            image = env.render(mode="rgb_array") 
            plt.imshow(image)
            plt.show(block=False)
            plt.pause(0.2)
            plt.close()
            # plt.savefig('ok.png')
            print("Done /n")

