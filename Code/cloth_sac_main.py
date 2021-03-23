# The following 3 lines are helpful if you have multiple GPUs and want to train
# agents on multiple GPUs
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Import packages
import Packages.dm_control
from Packages.dm_control.dm_control import suite
from Packages.dm_control.dm_control import viewer
from Packages.dm_control.dm_control.suite.wrappers import pixels
from Packages.dm_env.dm_env import specs
from PIL import Image, ImageTk
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect
from SAC.sac_torch import Agent
from SAC.utils import plot_learning_curve
import os
import torch as T
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
   
    # Check Torch version
    print(T.__version__)

    # Get location of DM Control Suite
    # print(inspect.getfile(dm_control.suite)) # built in path
    # /home/chandandeep/anaconda3/envs/rlpyt/lib/python3.7/site-packages/dm_control/suite/
    # Choose environment
    # env_id = 'cloth_v0'
    # env_id = 'cloth_sewts_exp1'
    env_id = 'cloth_sewts_exp2'
    

    # Define main variables 
    n_games = 10        
    score_max = []
    time_step_counter_max = []

    # Load environment
    env = suite.load(domain_name=env_id, task_name="easy")
    
    # Define action space and action shape
    action_space = env.action_spec()
    action_shape = action_space.shape[0]

    # Define observation space and observation shape
    observation_space = env.observation_spec()
    observation_shape = observation_space['position'].shape
    print(observation_shape)
    # Initialize SAC Agent 
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims= observation_shape, tau=0.005,
                env=env, batch_size=256, layer1_size=256, layer2_size=256,
                n_actions=action_shape)

    # create output folder for current experiment
    path_to_output = os.getcwd() + '/output'
    os.chdir(path_to_output)
    if not os.path.exists(env_id):
        subprocess.call(["mkdir","-p", env_id])
    
    # create experiment
    path_to_environment = path_to_output + '/' + env_id
    os.chdir(path_to_environment)

    # Load models from checkpoint
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        #viewer.launch(env)
    
    
    # Define log dictionary  
    log_dict = {
                "Image" : [],
                "Game_no" : [],
                "Step_no" : [],
                "State" : [],
                "Action" : [],
                "Reward" : []
                }

    log_file = path_to_environment + "/log.csv"

    # Loop for a no. of games
    for i in range(n_games):
        
        # Reset environment
        time_step = env.reset()

        # Define variables
        observation = time_step.observation['position']
        reward_history = []
        step = 0
        reward = 0
        done = False

        # Change directory to current environment path
        os.chdir(path_to_environment)
        
        # Make folder for each game
        game_no = 'game_' + str(i+1)
        subprocess.call([ 'mkdir', '-p', game_no ])

        # Display current game no.
        print("GAME NO.", i,"\n")

        while done is False : 

            # Move to game folder
            path_to_game = path_to_environment + '/' + game_no
            os.chdir(path_to_game)
            
            # Take action 
            action = agent.choose_action(observation)
            # print("ACTION \n")
            # print(action) # Print action, uncomment to display
            # action[2] = 0. # Uncomment for cloth_sewts_v1 env
            time_step = env.step(action)
            print("TIME STEP\n")
            print(time_step)

            # Get next observation and reward
            observation_ = time_step.observation['position']
            reward = time_step.reward
            # print(observation_) # Print oberservation
            # print(reward) # Print reward

            # Render image from environment for the time step and save
            # viewer.launch(env)
            image_data = env.physics.render(width = 640, height = 480, camera_id = 1)
            #img = Image.open(image_data)
            # image_array = np.asarray(image_data)
            img = Image.fromarray(image_data, 'RGB')
            img_loc = path_to_game + "/frame-%.10d.png" % step
            img_name = "frame-%.10d.png" % step
            img.save(img_name)
            
            # Increment step
            step += 1

            # Define terminal state (Max no. of steps 100 or when we reach close to the maximum reward of 0)
            if step == 10:
            # if step == 10000 or reward > -0.005: # No. of steps should be > batch size of 250 as the agent learns only when the batch is full
                done = True

            # Add current observation, action, reward and next observation to the Replay Buffer
            agent.remember(observation, action, reward, observation_, done)
            
            # Learn parameters of SAC Agent
            if not load_checkpoint:
                agent.learn()
            
            # Update observation with next observation
            observation = observation_

            # Add to the list of rewards for each time step 
            reward_history.append(reward)
            
            # Add a logger to capture states, rewards, actions coupled with the images captured to understand how the agent operates
            log_dict["Image"].append(img_loc)
            log_dict["Game_no"].append(game_no)
            log_dict["Step_no"].append(step)
            log_dict["State"].append(observation)
            log_dict["Action"].append(action)
            log_dict["Reward"].append(reward)

            # Save agent models
            if not load_checkpoint:
                agent.save_models()

        # Get maximum reward for the current game
        score_max.append(np.amax(reward_history))
        print("MAX REWARD FOR GAME NO.", i, "/n")
        print(score_max)

        # Define filename to store plots
        filename = env_id + '_'+ str(n_games) + 'plot.png'
        figure_file = filename
        figure_file = os.path.join(os.getcwd(), figure_file)

        # Plot learning curve
        x = [i for i in range(step)]
        plot_learning_curve(x, reward_history, figure_file)

        filename = str(n_games)
        subprocess.call([
                        'ffmpeg', '-framerate', '50', '-y', '-i',
                        'frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p','video.mp4'                
                        ])
    
    # In case you want to save the log_dict in a separate log.txt file
    df = pd.DataFrame(log_dict).to_csv(log_file, header = True, index = False)
    # df.to_csv(log_file)
    # df = pd.read_csv(log_file, index_col=0)
    # print(df)
    # file_csv = open(log_file, "w")
    # text_file = csv.writer(file_csv)
    # log_dict = zip(log_dict.items())
    # for key in log_dict.items():
    #     text_file.writerow([key])
    #for value in log_dict.items():
    #    text_file.writerow([value])
    # file_csv.close()


