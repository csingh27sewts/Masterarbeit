# Import packages
import Packages.dm_control
from Packages.dm_control.dm_control import suite
from Packages.dm_control.dm_control import viewer
from Packages.dm_control.dm_control.suite.wrappers import pixels
from Packages.dm_env.dm_env import specs
import torch.nn.functional as F
from PIL import Image, ImageTk
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect
import os
import torch as T
import torch.optim as optim
import pandas as pd
import sys
from SAC.buffer import ReplayBuffer
np.set_printoptions(threshold=sys.maxsize)
from SAC.networks import ActorNetwork

from SAC.utils import plot_learning_curve
from SAC import buffer
import pandas as pd

if __name__ == '__main__':
   
    # Check Torch version
    print(T.__version__)

    env_id = 'cloth_corner'
    # Load environment
    env = suite.load(domain_name=env_id, task_name="easy")

   
    # Define action space and action shape
    action_space = env.action_spec()
    action_shape = action_space.shape[0]
    # print("action_shape")
    # print(action_shape)
    # Define observation space and observation shape
    observation_space = env.observation_spec()
    observation_shape = observation_space['position'].shape
    print(observation_shape)

    # Define main variables 
    n_games =  1      
    score_max = []

    # create output folder for current experiment
    path_to_output = os.getcwd() + '/output/cloth_test/'
    os.chdir(path_to_output)
    if not os.path.exists(env_id):
        subprocess.call(["mkdir","-p", env_id])
    
    # create experiment
    path_to_environment = path_to_output + env_id
    os.chdir(path_to_environment)
    
    # Define log dictionary  
    log_dict = {
                "Image" : [],
                "Game_no" : [],
                "Step_no" : [],
                "State" : [],
                "Action" : [],
                "Reward" : [],
                "Actor_loss" : [],
                }
    
    log_file = path_to_environment + "/log_minimal.csv"

    # Initialize Replay Buffer
    max_size = 1000
    def random_policy():
        action = np.array([0.,0.,1.])

        return action

    # Loop for a no. of games
    for i in range(n_games):
        
        # Reset environment
        time_step = env.reset()
        observation = time_step.observation['position']
        reward_history = []
        step = 0
        
        done = False
        
        # Setting time
        for init_step in range(500):
            env.step(np.array([0.,0.,0.]))
        print("INITIALIZATION DONE")
        # Change directory to current environment path
        os.chdir(path_to_environment)
        
        # Make folder for each game
        game_no = 'game_' + str(i+1)
        subprocess.call([ 'mkdir', '-p', game_no ])

        # Display current game no.
        print("GAME NO.", i,"\n")
        actor_loss = 0 
        x = np.array([0])
        while done is False : 

            # Move to game folder
            path_to_game = path_to_environment + '/' + game_no
            os.chdir(path_to_game)
            
            observation = time_step.observation['position']
            # if step < 100:
            alpha=0.0006
            input_dims= observation_shape
            layer1_size=256
            layer2_size=256
            n_actions=action_shape
            observation = T.tensor(observation).to(T.device('cuda:0'))

            actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name=env_id+'_actor',
            #                     max_action=env.action_space.high) # for cart pole
                                  max_action=env.action_spec().maximum) # for cloth

            mu, sigma = ActorNetwork.forward(actor, observation)
            # action = random_policy()
            probabilities = T.distributions.Normal(mu, sigma)
            print("MU")
            print(mu)
            reparameterize = False
            if reparameterize:
                actions = probabilities.rsample() # reparameterizes the policy
            else:
                actions = probabilities.sample()
            print(actions)
            action = T.tensor(actions).to(T.device('cuda:0'))
            action = action * 20
            action = T.tanh(action) 

            print("ACTION")
            print(action)
            # action = action * 20 # dividing by 0.05 so that 1 correspondes to 0.05 movement
            
            
            # Get next observation and reward
            observation_ = time_step.observation['position']
            reward = time_step.reward
            print("REWARD")
            print(reward)

            actor_loss = -T.tensor(action).to(T.device('cuda:0')) + T.tensor([0.,0.,0.]).to(T.device('cuda:0'))
            # actor_loss = F.mse_loss(T.tensor([1,1,1]).to(T.device('cuda:0')), action)
            actor_loss = T.mean(actor_loss)
            actor.optimizer.zero_grad()
            actor_loss.requires_grad = True
            actor_loss.backward(retain_graph=True)
            actor.optimizer.step()
            action = action.cpu().detach().numpy()
            actor_loss = actor_loss.cpu().detach().numpy()
            time_step = env.step(action)
            print("TIME STEP\n")
            print(time_step)

            image_data = env.physics.render(width = 640, height = 480, camera_id = 0)
            #img = Image.open(image_data)
            # image_array = np.asarray(image_data)
            img = Image.fromarray(image_data, 'RGB')
            img_loc = path_to_game + "/frame-%.10d.png" % step
            img_name = "frame-%.10d.png" % step
            img.save(img_name)
            step += 1

            # Define terminal state (Max no. of steps 100 or when we reach close to the maximum reward of 0)
            if step == 10001:            # if step == 10000 or reward > -0.005: # No. of steps should be > batch size of 250 as the agent learns only when the batch is full
                done = True
            observation = observation_

            # Add to the list of rewards for each time step 
            reward_history.append(reward)
            
            # Add a logger to capture states, rewards, actions coupled with the images captured to understand how the agent operates
            log_dict["Image"].append(img_loc)
            log_dict["Game_no"].append(i + 1)
            log_dict["Step_no"].append(step)
            log_dict["State"].append(observation)
            log_dict["Action"].append(action)
            log_dict["Reward"].append(reward)
            log_dict["Actor_loss"].append(actor_loss)

            path_to_checkpoint = path_to_environment + '/checkpoint_tmp'

            intermediate_ckpt_actor = path_to_checkpoint + '/ckpt_actor_' + str(step) + '.ckpt'
            # if not load_checkpoint:
            T.save(actor.state_dict(), intermediate_ckpt_actor)       # Get maximum reward for the current game

        step_size = []
        length = len(log_dict["Actor_loss"])
        for j in range(length):
            step_size.append(j)
            if(log_dict["Actor_loss"][j]):
                if(isinstance(log_dict["Actor_loss"][j], np.ndarray)):
                    log_dict["Actor_loss"][j] = log_dict["Actor_loss"][j].item()
            else:
                log_dict["Actor_loss"][j] = 0

        filename_actor_loss = env_id + '_'+ str(i) + 'plot_actor_loss.png'
        figure_file_actor_loss = filename_actor_loss
        figure_file_actor_loss = os.path.join(os.getcwd(), figure_file_actor_loss)
        plot_learning_curve(step_size, log_dict["Actor_loss"], figure_file_actor_loss)


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
    df = pd.DataFrame(log_dict)
    df.to_csv(log_file, header = True, index = False)

