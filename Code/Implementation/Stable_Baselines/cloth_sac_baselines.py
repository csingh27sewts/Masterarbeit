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

    # EXPERIMENT #SAMPLE
    # env_id = 'pendulum'
    # env = gym.make('Pendulum-v0')   
    # Environment : 
    # /home/chandandeep/anaconda3/envs/rlpyt/lib/python3.8/site-packages/dm_control/suite/
    
    # EXPERIMENT #2
    env_id = 'cloth_sewts_minimal_1_2'
    # Make z 0
    # location_1 = "/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/SAC_2_1/cloth_sewts_minimal"
    # Make corner move to fixed position
    location_1 = "/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/SAC_1_2/cloth_sewts_minimal"
    # Make adjacent point move to fixed position
    location_2 = "/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/SAC_1_4/cloth_sewts_minimal"
    location_3 = "/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/SAC_1_5/cloth_sewts_minimal"

    location_6 = "/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/SAC_1_6/cloth_sewts_minimal"
    location_7 = "/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/SAC_1_7/cloth_sewts_minimal"
    location_8 = "/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/SAC_1_8/cloth_sewts_minimal"
    
    env = dmc2gym.make(domain_name=env_id, task_name='easy', seed=1) #dmc2gym package
    location_set = [location_6]
    # location_set = [location_1, location_2, location_3]
    # create output folder for current experiment
    path_to_output = os.getcwd() + '/output'
    os.chdir(path_to_output)
    # for i in range(100):
    observation = env.reset()
        # image = env.render(mode="rgb_array", height=256, width=256) 
        # plt.imshow(image)
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
    
    model = SAC("MlpPolicy", env, learning_starts=1024, 
                tensorboard_log='/home/chandandeep/Masterarbeit/Code/Implementation/Stable_Baselines/output/',
                batch_size=256, verbose=1)
    step = 0
    reward = 0
    for i in range(len(location_set)):
        print("##################################################")
        print("LOOP",i + 1,"\n")
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

            # model = SAC.load(location) # load and continue from previous
            model.set_env(env)
            # if i == 0:
            #     observation = env.reset()   
            print(observation)
            image = env.render(mode="rgb_array", height=256, width=256)
            # plt.imshow(image)
            # plt.show(block=False)
            # plt.pause(0.25)
            # plt.close()
            model.learn(total_timesteps=150000, reset_num_timesteps = False, callback = stable_baselines_logging.ImageRecorderCallback(), log_interval=1) # log_interval = no. of episodes
            print("/nLEARNT")
            # EXPERIMENT #
            model.save(location_set[i])
            model = SAC.load(location_set[i])

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
                if step == 500 or reward > 490: # EXPERIMENT_1 # EXPERIMENT_0
                # if step == 500 or reward > 0.99: # EXPERIMENT_1_1
                # if step == 500 or reward > -2 and reward < 2: # EXPERIMENT_2_1
                # if step == 500 or reward > 1490: # EXPERIMENT_3
                    done = True
                    # if i == 0:
                    #     obs = env.reset()
                # image = env.render()
                # "HWC" specify the dataformat of the image, here channel last
                # (H for height, W for width, C for channel)
                # See https://pytorch.org/docs/stable/tensorboard.html
                # for supported formats
                
                image = env.render(mode="rgb_array", height=256, width=256) 
                plt.imshow(image)
                plt.show(block=False)
                plt.pause(0.25)
                plt.close()
                
                # plt.savefig('ok.png')
                print("Done /n")
        i = i + 1
