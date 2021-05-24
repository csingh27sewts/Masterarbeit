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
import os
import torch as T
import torch.optim as optim
import pandas as pd
import sys
from SAC.buffer import ReplayBuffer
np.set_printoptions(threshold=sys.maxsize)

from SAC.utils import plot_learning_curve
from SAC import buffer
import pandas as pd

if __name__ == '__main__':
   
    # Check Torch version
    print(T.__version__)

    # Get location of DM Control Suite
    # print(inspect.getfile(dm_control.suite)) # built in path
    # /home/chandandeep/anaconda3/envs/rlpyt/lib/python3.7/site-packages/dm_control/suite/
    # Choose environment
    # env_id = 'cloth_v0'
    # env_id = 'cloth_sewts_exp2_2'
    env_id = 'cloth_corner'
    

    # Define main variables 
    n_games =  1       
    score_max = []
    time_step_counter_max = []

    # Load environment
    env = suite.load(domain_name=env_id, task_name="easy")
    
    # Define action space and action shape
    action_space = env.action_spec()
    action_shape = action_space.shape[0]
    action_max = env.action_spec().maximum
    action_min = env.action_spec().minimum

    # Define observation space and observation shape
    observation_space = env.observation_spec()
    observation_shape = observation_space['position'].shape
    print("Observation Space")
    print(observation_space)
    # Initialize SAC Agent 

    # create output folder for current experiment
    path_to_output = os.getcwd() + '/output/cloth_minimal_main_2/'
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
                "Reward" : []
                }

    
    log_file = path_to_environment + "/log_minimal.csv"
    log_good_file = path_to_output + "/log_good.csv"

    # df_good_reward = pd.read_csv(log_good_file)

    # Initialize Replay Buffer
    max_size = 1000
    input_dims = observation_shape
    n_actions = action_shape
    memory = ReplayBuffer(max_size, input_dims, n_actions)

    # Replay Buffer Filtered based on reward values

    # States and actions used to train the NN

    def train_nn(df):
            
        # Step 1. Random policy initially for a specific no. of steps say 250 
        # Step 2. Collect states, actions, next_states, rewards in a replay buffer
        # Step 3. Filter replay buffer based on reward values 
        # Step 4. Store entries with > -0.10 in a new replay buffer
        # Step 5. Feed state and action values from new replay buffer to a neural network (Train it)
        # Step 6. Test on new states for the learnt parameters of the neural network
        
        beta = 0.01
        fc1_dims = 250
        print(observation.shape)
        print(fc1_dims)
        print(n_actions)
        fc1_layer = T.nn.Sequential(T.nn.Linear(observation.shape[0], fc1_dims)
                    ,T.nn.Linear(fc1_dims, n_actions))
        T.optimizer = optim.Adam(fc1_layer.parameters(), lr=beta)
        # T.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # T.to(T.device) 

        # nn_state = []
        # nn_action = []      
        # nn_state_ = []
        # nn_reward = []
        # nn_done = []
        pred_action = []
        
        rewards = [reward for reward in df[df.columns[6]]]
        rewards = np.array(list(rewards))
        states = [state for state in df[df.columns[4]]]
        states = np.array(list(states))
        actions = [action for action in df[df.columns[5]]]
        actions = np.ndarray(list(actions), type = float)
        print(rewards)
        print(states)
        print(actions)
        
        # for i in range(len(rewards)):
            # states[i] = np.ndarray(list(states[i]),dtype = float)
            # print(states[i])
            # states[i] = T.from_numpy(states[i])
            # print(type(states[i]))
            # print(states[i])
            # states[i] = T.tensor(states[i], dtype = T.float)
            # pred_action.append(fc1_layer(states[i]))
        
        # print(pred_action)
        
        # for i in range(memory.mem_cntr):
        #    nn_state.append(memory.state_memory[memory.mem_cntr])
        #    nn_action.append(memory.action_memory[memory.mem_cntr])
        #    nn_state_.append(memory.new_state_memory[memory.mem_cntr])
        #    nn_reward.append(memory.reward_memory[memory.mem_cntr])
        #    nn_done.append(memory.terminal_memory[memory.mem_cntr])
        #    print("-----------------")
        #    print(i)
        #    print(nn_state[i])
        #    print(nn_action[i])
        #    print(nn_state_[i])
        #    print(nn_reward[i])
            
         
        #for i in range(memory.mem_cntr):
         #   pred_action.append(fc1_layer(T.tensor(nn_state[i], dtype = T.double)))
         #   print("Fed")
         #   print(nn_state[i])
         #   print(nn_action[i])
         #   print("Predicted")
         #   print(pred__action[i])
         #   print("\n")
         #   print(fc1_layer.parameters())

    def random_policy():

        # POLICY TYPE 5 - Actions proportional to the step in the range [-1,0] (Just pulling away in (x,y,z) direction
        # action = np.random.uniform(low=action_min*step*0.01,
        # high=0,
        # size=action_shape)

        # POLICY TYPE 4 - Random actions in the range [-1,0] (Just pulling away) in (x,y,z) direction 
        # action = np.random.uniform(low=action_min,
        # high=action_max,
        # size=action_shape)

        # POLICY TYPE 3 - Random actions in the range [-1,1] in (x,y) direction 
        # action = np.random.uniform(low=action_min,
        # high=action_max,
        # size=action_shape)
        # action[2] = 0.

        # POLICY TYPE 2 - Random actions in the range [-1,1] in (x,y,z) direction 
        # action = np.random.uniform(low=action_min,
        # high=action_max,
        # size=action_shape)

        action = np.array([0.,0.,0.])

        # POLICY TYPE 1 - Pulling in x direction by fixed amount
        #action = np.random.uniform([-1.,0.,0.])
        #action = np.array([0., 0., 0.])
        return action

        # Reading log_good.csv


    # Loop for a no. of games
    for i in range(n_games):
        
        # Reset environment
        time_step = env.reset()
        #image_data = env.physics.render(width = 640, height = 480, camera_id = 1)
        #plt.imshow(image_data)
        #plt.show()
        # Define variables
        observation = time_step.observation['position']
        reward_history = []
        step = 0
        # reward = -0.45
        done = False
        
        # Setting time
        for init_step in range(500):
            env.step(np.array([0.,0.,0.]))
        #image_random = env.physics.render(width = 640, height = 480, camera_id = 1)
        #plt.imshow(image_random)
        #plt.show()

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
            # action = np.random.uniform(low=action_min,
            #               high=action_max,
            #               size=action_shape)
            # print("ACTION \n")
            # print(action) # Print action, uncomment to display
            #            action[1] = 0.
            # action = define_policy(step)
            observation = time_step.observation['position']
            if step < 100:
                action = random_policy()
            #else:
            #    train_nn(df_good_reward)
            # action[1] = 0. # Uncomment for cloth_sewts_v1 / v2 env
            # action[2] = 0. # Uncomment for cloth_sewts_v1 / v2 env
            time_step = env.step(action)
            print("TIME STEP\n")
            print(time_step)

            # Get next observation and reward
            observation_ = time_step.observation['position']
            reward = time_step.reward
            print(reward)
            # Store transitions with good rewards
            #if step < 100 and reward > -0.15:
            #    ReplayBuffer.store_transition(memory, observation, action, reward, observation_, done)
            
            # print(observation_) # Print oberservation
            # print(reward) # Print reward

            # Render image from environment for the time step and save
            # viewer.launch(env)
            image_data = env.physics.render(width = 640, height = 480, camera_id = 0)
            #img = Image.open(image_data)
            # image_array = np.asarray(image_data)
            img = Image.fromarray(image_data, 'RGB')
            img_loc = path_to_game + "/frame-%.10d.png" % step
            img_name = "frame-%.10d.png" % step
            img.save(img_name)
            # img.show()        
            # Increment step
            step += 1

            # Define terminal state (Max no. of steps 100 or when we reach close to the maximum reward of 0)
            if step == 101:
            # if step == 10000 or reward > -0.005: # No. of steps should be > batch size of 250 as the agent learns only when the batch is full
                done = True

            # Add current observation, action, reward and next observation to the Replay Buffer
            # agent.remember(observation, action, reward, observation_, done)
            
            # Learn parameters of SAC Agent
            # if not load_checkpoint:
            #    agent.learn()
            
            # Update observation with next observation
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

            # Save agent models
            # if not load_checkpoint:
            #    agent.save_models()

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
    #df_good = df[df.Reward > -0.10] 
    #df_good.to_csv(log_good_file, mode = 'a', header = False)
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
