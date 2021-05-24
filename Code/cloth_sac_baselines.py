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

if __name__ == '__main__':
   
    # Define environment'
    env_id = 'cloth_corner'
    n_games = 5
    score_max = []
    time_step_counter_max = []

    # Load environment
    env = dmc2gym.make(domain_name='cloth_corner', task_name='easy', seed=1) #dmc2gym package

    # create output folder for current experiment
    path_to_output = os.getcwd() + '/output'
    os.chdir(path_to_output)
    
    if not os.path.exists(env_id):
        subprocess.call(["mkdir","-p", env_id])

    # create experiment
    path_to_environment = path_to_output + '/' + env_id
    os.chdir(path_to_environment)

    
    # Define log dictionary  
    log_dict = {
                "Image" : [],
                "Game_no" : [],
                "Step_no" : [],
                "State" : [],
                "Action" : [],
                "Reward" : [],
                }
    # Loop for a no. of games
    for i in range(n_games):
        
        # Reset environment
        print(env.reset())
        observation  = env.reset()
        
        reward_history = []
        average_value = []
        average_critic = []
        average_entropy = []
        average_reward = []
        average_target_value = []
        step = 0
        reward = 0
        done = False
        print("GAME no.",i,"/n")
        for init_step in range(500):
            env.step(np.array([0.,0.,0.])) # for cloth_corner
        # Change directory to current environment path
        os.chdir(path_to_environment)
        
        # Make folder for each game
        game_no = 'game_' + str(i+1)
        subprocess.call([ 'mkdir', '-p', game_no ])

        log_file = path_to_environment + "/log"+str(i)+".csv"
        log_sac_file = path_to_environment + "/log_sac"+str(i)+".csv"
        log_sac_file_inputs = path_to_environment + "/log_sac_inputs"+str(i)+".csv"
        log_sac_loss_file = path_to_environment + "/log_sac_loss"+str(i)+".csv"
        print("Environment")
        # print(env)
        #print(env.observation_space.spaces["location"])
        model = SAC("MlpPolicy", env, batch_size = 64, verbose=1)

        while done is False : 

            # Move to game folder
            path_to_game = path_to_environment + '/' + game_no
            os.chdir(path_to_game)
            action, observation_ = model.predict(observation, deterministic=True)
            print("STEP",i,"/n")
        
            #print(action,"/n")
            # print(observation,"/n")
            action_fed = np.array([action[0], action[1], action[2]]) #cloth_corner
            done = False
            obs = env.reset()
            # while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #print(env.step(action))
            print("Learning")
            model.learn(total_timesteps=1000 ,log_interval=10)

            image_data = env.physics.render(width = 64, height = 64 , camera_id = 0)
            img = Image.fromarray(image_data, 'RGB')
            img_loc = path_to_game + "/frame-%.10d.png" % step
            img_name = "frame-%.10d.png" % step
            img.save(img_name)
            # Increment step
            step += 1

            observation = obs
            # reward_history.append(reward)
            print(reward) 
            # Add a logger to capture states, rewards, actions coupled with the images captured to understand how the agent operates

            # log_dict["Image"].append(img_loc)
            # log_dict["Game_no"].append(i + 1)
            # log_dict["Step_no"].append(step)
            # log_dict["State"].append(observation)
            # log_dict["Action"].append(action)
            # log_dict["Reward"].append(reward)
        # score_max.append(np.amax(reward_history))
        # Define filename to store plots
        filename = env_id + '_'+ str(n_games) + 'plot.png'
        figure_file = filename
        figure_file = os.path.join(os.getcwd(), figure_file)

        # Plot learning curves

        subprocess.call([ 'mkdir', '-p', 'Plots' ])
        os.chdir('Plots')
        

