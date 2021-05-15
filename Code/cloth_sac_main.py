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
# from simulation import simulation
np.set_printoptions(threshold=sys.maxsize)
import gc

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
    # total steps = 500000 0> n_games = 50, n_steps = 10000
    n_games = 50
    # n_games = 50       
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
    agent = Agent(alpha=0.0006, beta=0.0003, reward_scale=1, env_id=env_id, 
    # agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims= observation_shape, tau=0.005,
                env=env, batch_size=1024, layer1_size=256, layer2_size=256,
    #           env=env, batch_size=1024,layer1_size=256, layer2_size=256,
               n_actions=action_shape)

    # create output folder for current experiment
    path_to_output = os.getcwd() + '/output'
    os.chdir(path_to_output)
    
    if not os.path.exists(env_id):
        subprocess.call(["mkdir","-p", env_id])

    # create experiment
    path_to_environment = path_to_output + '/' + env_id
    os.chdir(path_to_environment)

    if not os.path.exists('checkpoint_tmp'):
        subprocess.call(["mkdir","-p", 'checkpoint_tmp'])

    # Path to all intermediate checkpoints
    path_to_checkpoint = path_to_environment + '/checkpoint_tmp'

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
                "Reward" : [],
                }
    log_sac_dict = {
#                "State" : [],
#                "Next_State" : [],
#                "Action" : [],
#                "Sampled_Action" : [],
                "Reward" : [],
                "Value" : [],
                "Critic" : [],
                "Target_Value" : [],
                "Entropy" : []
                }
    log_sac_loss_dict = {
                "Value_loss" : [],
                "Critic_loss" : [],
                "Actor_loss" : []
                }
    log_sac_dict_inputs = {
                "State" : [],
                "Next_State" : [],
                "Action" : [],
                "Sampled_Action" : []
                }
    # Loop for a no. of games
    for i in range(n_games):
        
        # Reset environment
        time_step = env.reset()

        # Define variables
        print(time_step)
        # observation = time_step.observation
        observation = time_step.observation['position']
        reward_history = []
        average_value = []
        average_critic = []
        average_entropy = []
        average_reward = []
        step = 0
        reward = 0
        done = False
        print("GAME no.",i,"/n")
        for init_step in range(500):
            # env.step(np.array([0.,0.])) # for cloth_sewts_exp2_2
            env.step(np.array([0.,0.,0.])) # for cloth_corner

        
        # Change directory to current environment path
        os.chdir(path_to_environment)
        
        # Make folder for each game
        game_no = 'game_' + str(i+1)
        subprocess.call([ 'mkdir', '-p', game_no ])

        # Display current game no.
        print("GAME NO.", i,"\n")

        log_file = path_to_environment + "/log"+str(i)+".csv"
        log_sac_file = path_to_environment + "/log_sac"+str(i)+".csv"
        log_sac_file_inputs = path_to_environment + "/log_sac_inputs"+str(i)+".csv"
        log_sac_loss_file = path_to_environment + "/log_sac_loss"+str(i)+".csv"

        while done is False : 

            # Move to game folder
            # gc.collect()
            path_to_game = path_to_environment + '/' + game_no
            os.chdir(path_to_game)
            
            # Take action 
            
            # action = np.array([0.1,0.1])
            action = agent.choose_action(observation)
            # gc.collect()
            # T.cpu().empty_cache()
            # print("ACTION \n")
            # print(action) # Print action, uncomment to display
            # action[1] = 0.
            # Do not artificially change one action component. The agent gives the action [x,y,z] which would move it closer to goal
            # Changing one component is essentially not using what the policy predicts. Change in the main file if you need to.
            print("ACTION IT IS \n")
            # action = np.array([1,1,0])
            print(action)
            action_fed = np.array([action[0], action[1], action[2]]) #cloth_corner
            # action_fed = np.array([action[0], action[1]])
            print(action_fed)
            # action[2] = 0. # Uncomment for cloth_sewts_v1 / v2 env
            time_step = env.step(action_fed)
            # Inference in simulation
            # print("ACTION FED")
            # print(action_fed)
            
            # INFERENCE IN SIMULATION !
            # if load_checkpoint:
            #    simulation(action_fed)
            
            
            # print("TIME STEP\n")
            # print(time_step)

            # Get next observation and reward
            observation_ = time_step.observation['position']
            reward = time_step.reward
            # print(observation) # Print oberservation
            # print(reward) # Print reward

            # Render image from environment for the time step and save
            # viewer.launch(env)

            
            

            image_data = env.physics.render(width = 64, height = 64 , camera_id = 0)
            #img = Image.open(image_data)
            # image_array = np.asarray(image_data)
            # print("Saving frames")
            img = Image.fromarray(image_data, 'RGB')
            img_loc = path_to_game + "/frame-%.10d.png" % step
            img_name = "frame-%.10d.png" % step
            img.save(img_name)
            
            # print(img_loc)
            # print(os.getcwd())
            # img.show(img)
            

            # Increment step
            step += 1

            # Define terminal state (Max no. of steps 100 or when we reach close to the maximum reward of 0)
            # if step == 100 or reward > 0.9 and step > 20: # cloth_corner
            if step == 5000 or reward > 0.9 and step > 1000:
            # if step == 10000 or reward > -0.005: # No. of steps should be > batch size of 250 as the agent learns only when the batch is full
                done = True

            # Add current observation, action, reward and next observation to the Replay Buffer
            agent.remember(observation, action, reward, observation_, done)
            # gc.collect() 
            # Learn parameters of SAC Agent
            if not load_checkpoint:    
                # agent.value.fc1.weight.data.fill_(0.1)
                # agent.value.fc1.bias.data.fill_(0.1)
                # agent.value.fc2.weight.data.fill_(0.1)
                # agent.value.fc2.bias.data.fill_(0.1)
                # agent.value.v.weight.data.fill_(0.1)
                # agent.value.v.bias.data.fill_(0.1)
                # T.nn.init.uniform_(T.empty(3, 5))
                T.save(agent.value.state_dict(), os.path.join(path_to_environment,'checkpoint_tmp/init_value.ckpt'))
                #obs_trial = [0]*12
                #obs_trial = T.tensor(obs_trial)
                #print("OBS TRIAL \n")
                #print(agent.value(obs_trial))
                T.save(agent.actor.state_dict(), os.path.join(path_to_environment, 'checkpoint_tmp/init_actor.ckpt'))
                T.save(agent.critic_1.state_dict(),  os.path.join(path_to_environment,'checkpoint_tmp/init_critic.ckpt'))
                T.save(agent.target_value.state_dict(), os.path.join(path_to_environment,'checkpoint_tmp/init_target_value.ckpt'))
                agent.learn()
             
            # Update observation with next observation
            observation = observation_
            # gc.collect()
            # Add to the list of rewards for each time step 
            reward_history.append(reward)
            
            # Add a logger to capture states, rewards, actions coupled with the images captured to understand how the agent operates

            log_dict["Image"].append(img_loc)
            log_dict["Game_no"].append(i + 1)
            log_dict["Step_no"].append(step)
            log_dict["State"].append(observation)
            log_dict["Action"].append(action)
            log_dict["Reward"].append(reward)
             
            if agent.learn() is not None:
                # print("PRINTING :: : :: : :: : :: : :")
                # print(agent.learn()[0].detach().cpu().numpy())
                log_sac_dict_inputs["State"].append(agent.learn()[0].detach().cpu().numpy())
                log_sac_dict_inputs["Next_State"].append(agent.learn()[1].detach().cpu().numpy())
                log_sac_dict_inputs["Action"].append(agent.learn()[2].detach().cpu().numpy())
                log_sac_dict_inputs["Sampled_Action"].append(agent.learn()[3].detach().cpu().numpy())
                log_sac_dict["Reward"].append(agent.learn()[4].detach().cpu().numpy())
                log_sac_dict["Value"].append(agent.learn()[5].detach().cpu().numpy())
                log_sac_dict["Target_Value"].append(agent.learn()[6].detach().cpu().numpy())
                log_sac_dict["Critic"].append(agent.learn()[7].detach().cpu().numpy())
                log_sac_dict["Entropy"].append(agent.learn()[8].detach().cpu().numpy())
                log_sac_loss_dict["Value_loss"].append(agent.learn()[9].detach().cpu().numpy())
                log_sac_loss_dict["Critic_loss"].append(agent.learn()[10].detach().cpu().numpy())
                log_sac_loss_dict["Actor_loss"].append(agent.learn()[11].detach().cpu().numpy())
        
            else:
                log_sac_dict_inputs["State"].append(None)
                log_sac_dict_inputs["Next_State"].append(None)
                log_sac_dict_inputs["Action"].append(None)
                log_sac_dict_inputs["Sampled_Action"].append(None)
                log_sac_dict["Reward"].append(None)
                log_sac_dict["Value"].append(None)
                log_sac_dict["Target_Value"].append(None)
                log_sac_dict["Critic"].append(None)
                log_sac_dict["Entropy"].append(None)
                log_sac_loss_dict["Value_loss"].append(None)
                log_sac_loss_dict["Critic_loss"].append(None)
                log_sac_loss_dict["Actor_loss"].append(None)
            # gc.collect() 
        # Save agent models
            if not load_checkpoint:
                agent.save_models()

            intermediate_ckpt_value = path_to_checkpoint + '/ckpt_value_' + str(step) + '.ckpt'
            if not load_checkpoint:
                T.save(agent.value.state_dict(), intermediate_ckpt_value)

            intermediate_ckpt_critic_1 = path_to_checkpoint + '/ckpt_critic_1' + str(step) + '.ckpt'
            if not load_checkpoint:
                T.save(agent.critic_1.state_dict(), intermediate_ckpt_critic_1)

            intermediate_ckpt_critic_2 = path_to_checkpoint + '/ckpt_critic_2' + str(step) + '.ckpt'
            if not load_checkpoint:
                T.save(agent.critic_2.state_dict(), intermediate_ckpt_critic_2)
 
            intermediate_ckpt_actor = path_to_checkpoint + '/ckpt_actor_' + str(step) + '.ckpt'
            if not load_checkpoint:
                T.save(agent.actor.state_dict(), intermediate_ckpt_actor)       # Get maximum reward for the current game

        score_max.append(np.amax(reward_history))
        #print("MAX REWARD FOR GAME NO.", i, "/n")
        #print(score_max)

        # Define filename to store plots
        filename = env_id + '_'+ str(n_games) + 'plot.png'
        figure_file = filename
        figure_file = os.path.join(os.getcwd(), figure_file)

        # Plot learning curves

        subprocess.call([ 'mkdir', '-p', 'Plots' ])
        os.chdir('Plots')
        
        # gc.collect()

        x = [i for i in range(step)]
        plot_learning_curve(x, reward_history, figure_file)
        for i in range(len(log_sac_dict["Entropy"])):
            filename_entropy = env_id + '_'+ str(i) + 'plot_entropy.png'
            figure_file_entropy = filename_entropy
            figure_file_entropy = os.path.join(os.getcwd(), figure_file_entropy)
            if(log_sac_dict["Entropy"][i]) is not None:
                step_size = []
                length = len(log_sac_dict["Entropy"][i])
                average_entropy.append(sum(log_sac_dict["Entropy"][i]) / len(log_sac_dict["Entropy"][i]))
                for j in range(length):
                    step_size.append(j)
                plot_learning_curve(step_size, log_sac_dict["Entropy"][i], figure_file_entropy)

        for i in range(len(log_sac_dict["Value"])):
            filename_value = env_id + '_'+ str(i) + 'plot_value.png'
            figure_file_value = filename_value
            figure_file_value = os.path.join(os.getcwd(), figure_file_value)
            if(log_sac_dict["Value"][i]) is not None:
                step_size = []
                length = len(log_sac_dict["Value"][i])
                average_value.append(sum(log_sac_dict["Value"][i]) / len(log_sac_dict["Value"][i]))
                for j in range(length):
                    step_size.append(j)
                plot_learning_curve(step_size, log_sac_dict["Value"][i], figure_file_value)      
        
        for i in range(len(log_sac_dict["Critic"])):
            filename_critic = env_id + '_'+ str(i) + 'plot_critic.png'
            figure_file_critic = filename_critic
            figure_file_critic = os.path.join(os.getcwd(), figure_file_critic)
            if(log_sac_dict["Critic"][i]) is not None:
                step_size = []
                length = len(log_sac_dict["Critic"][i])
                average_critic.append(sum(log_sac_dict["Critic"][i]) / len(log_sac_dict["Critic"][i]))
                for j in range(length):
                    step_size.append(j)
                plot_learning_curve(step_size, log_sac_dict["Critic"][i], figure_file_critic)  

        for i in range(len(log_sac_dict["Reward"])):
            filename_reward = env_id + '_'+ str(i) + 'plot_reward.png'
            figure_file_reward = filename_reward
            figure_file_reward = os.path.join(os.getcwd(), figure_file_reward)
            if(log_sac_dict["Reward"][i]) is not None:
                step_size = []
                log_sac_dict["Reward"][i] = np.array(log_sac_dict["Reward"][i])               
                length = len(log_sac_dict["Reward"][i])
                # print("REWARD")
                # print(log_sac_dict["Reward"])
                average_reward.append(sum(log_sac_dict["Reward"][i]) / len(log_sac_dict["Reward"][i]))
                for j in range(length):
                    step_size.append(j)
                # print(log_sac_dict["Reward"][i].dtype) 
                plot_learning_curve(step_size, log_sac_dict["Reward"][i], figure_file_reward)  
       
        # gc.collect()

        step_size = []
        #if(log_sac_loss_dict["Value_loss"]) is not None:
        length = len(log_sac_loss_dict["Value_loss"])
        print(log_sac_loss_dict["Value_loss"])
        for j in range(length):
            step_size.append(j)
            if(log_sac_loss_dict["Value_loss"][j]):
                if(isinstance(log_sac_loss_dict["Value_loss"][j], np.ndarray)):
                    log_sac_loss_dict["Value_loss"][j] = log_sac_loss_dict["Value_loss"][j].item()
            else:
                log_sac_loss_dict["Value_loss"][j] = 0  
            if(log_sac_loss_dict["Critic_loss"][j]):
                if(isinstance(log_sac_loss_dict["Critic_loss"][j], np.ndarray)):
                    log_sac_loss_dict["Critic_loss"][j] = log_sac_loss_dict["Critic_loss"][j].item()
            else:
                 log_sac_loss_dict["Critic_loss"][j] = 0
            if(log_sac_loss_dict["Actor_loss"][j]):
                if(isinstance(log_sac_loss_dict["Actor_loss"][j], np.ndarray)):
                    log_sac_loss_dict["Actor_loss"][j] = log_sac_loss_dict["Actor_loss"][j].item()
            else:
                log_sac_loss_dict["Actor_loss"][j] = 0         
                       
        print("STEP !!!!!!!!!!!!!!")
        print(step_size)
        print(log_sac_loss_dict["Value_loss"])
        filename_value_loss = env_id + '_'+ str(i) + 'plot_value_loss.png'
        figure_file_value_loss = filename_value_loss
        figure_file_value_loss = os.path.join(os.getcwd(), figure_file_value_loss)
        plot_learning_curve(step_size, log_sac_loss_dict["Value_loss"], figure_file_value_loss)  
        
        filename_critic_loss = env_id + '_'+ str(i) + 'plot_critic_loss.png'
        figure_file_critic_loss = filename_critic_loss
        figure_file_critic_loss = os.path.join(os.getcwd(), figure_file_critic_loss)
        plot_learning_curve(step_size, log_sac_loss_dict["Critic_loss"], figure_file_critic_loss)  
        
        filename_actor_loss = env_id + '_'+ str(i) + 'plot_actor_loss.png'
        figure_file_actor_loss = filename_actor_loss
        figure_file_actor_loss = os.path.join(os.getcwd(), figure_file_actor_loss)
        plot_learning_curve(step_size, log_sac_loss_dict["Actor_loss"], figure_file_actor_loss)  
        
        filename = str(n_games)
        subprocess.call([
                        'ffmpeg', '-framerate', '50', '-y', '-i',
                        'frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p','video.mp4'                
                        ])

        # os.chdir(path_to_environment)
        
        filename_average_critic = str(i) + 'plot_average_critic.png'
        figure_file_average_critic = filename_average_critic
        figure_file_average_critic = os.path.join(os.getcwd(), figure_file_average_critic)


        filename_average_value = str(i) + 'plot_average_value.png'
        figure_file_average_value = filename_average_value
        figure_file_average_value = os.path.join(os.getcwd(), figure_file_average_value)       

        filename_average_entropy = str(i) + 'plot_average_entropy.png'
        figure_file_average_entropy = filename_average_entropy
        figure_file_average_entropy = os.path.join(os.getcwd(), figure_file_average_entropy)
 
        filename_average_reward = str(i) + 'plot_average_reward.png'
        figure_file_average_reward = filename_average_reward
        figure_file_average_reward = os.path.join(os.getcwd(), figure_file_average_reward)
        
        steps_avg = []
        for x in range(len(average_value)):
            steps_avg.append(x)
        
        # print(steps_avg)
        # print(average_entropy)
        # print(figure_file_average_entropy)

        plot_learning_curve(steps_avg, average_entropy, figure_file_average_entropy)  
        plot_learning_curve(steps_avg, average_critic, figure_file_average_critic)  
        plot_learning_curve(steps_avg, average_value, figure_file_average_value)  
        plot_learning_curve(steps_avg, average_reward, figure_file_average_reward)  
   
        # Uncomment when you want to save all important values in a CSV
        # In case you want to save the log_dict in a separate log.txt file
        print("Saving CSVs ... /n")
        df = pd.DataFrame(log_dict).to_csv(log_file, header = True, index = False)
        # print(log_sac_dict.keys())
        # df = pd.DataFrame(log_sac_dict).to_csv(log_sac_file, header = True, index = False)
        # df = pd.DataFrame(log_sac_dict_inputs).to_csv(log_sac_file_inputs, header = True, index = False)
        df = pd.DataFrame(log_sac_loss_dict).to_csv(log_sac_loss_file, header = True, index = False)
            
        log_dict = {key: [] for key in log_dict}
        log_sac_loss_dict = {key: [] for key in log_sac_loss_dict}
        log_sac_dict = {key: [] for key in log_sac_dict}
        log_sac_dict_inputs = {key: [] for key in log_sac_dict_inputs}
        print("Saved CSVs ... /n")     
        #log_sac_loss_dict.clear()
        #log_sac_dict.clear()
        #log_sac_dict_inputs.clear()

    # print("Reward")
    # print(len(log_sac_dict["Reward"][260]))
    # print("Value \n")
    # print(len(log_sac_dict["Value"][260]))
    # print("Target Value \n")
    # print(len(log_sac_dict["Target_Value"][260]))
    # print("Critic \n")
    # print(len(log_sac_dict["Critic"][260]))
    # print("Entropy  \n")   
    # print(len(log_sac_dict["Entropy"][260]))
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

    # Define filenames to store network output plots
    # os.chdir(path_to_environment)
    # path_to_plots = path_to_environment + '/Plots/'
    # print(path_to_plots)

    # if not os.path.exists(path_to_plots):
    #    subprocess.call(["mkdir","-p", "Plots"])

       
    # filename_critic = env_id + '_'+ str(n_games) + 'plot_critic.png'
    # figure_file_critic = filename_critic
    # figure_file_critic = os.path.join(path_to_plots, figure_file_critic)

    # filename_target_value = env_id + '_'+ str(n_games) + 'plot_target_value.png'
    # figure_file_target_value = filename_target_value
    # figure_file_target_value = os.path.join(path_to_plots, figure_file_target_value)

    # filename_entropy = env_id + '_'+ str(n_games) + 'plot_entropy.png'
    # figure_file_entropy = filename_entropy
    # figure_file_entropy = os.path.join(path_to_plots, figure_file_entropy)

    # Plot outputs of all neural networks implemented in SAC
    # value = agent.value_collect
    # critic = agent.critic_collect
    # target_value = agent.target_value_collect
    # entropy = agent.entropy_collect
    # learn_step = agent.learn_step

    # print(value)
    # print(critic)
    # print(target_value)
    # print(entropy)
    # print(learn_step)

 
    #print("PLOTS\n")
    #for j in range(len(value)):
    #    value_plot = []
    #    value_step = []     
    #    filename_value = env_id + '_'+ str(j) + 'plot_value.png'
    #    figure_file_value = filename_value
    #    figure_file_value = os.path.join(path_to_plots, figure_file_value)

    #    for i in range(len(value[j])):
    #        value_plot.append(value[j].detach().numpy())
    #        value_step.append(i)
    #    plot_learning_curve(value_plot, value_step , figure_file_value)

    #print(value)
    #print(critic)
    #print(target_value)
    #print(entropy)
    #print(learn_step)


    #plot_learning_curve(learn_step, critic, figure_file_critic)
    #plot_learning_curve(learn_step, target_value, figure_file_target_value)
    #plot_learning_curve(learn_step, entropy, figure_file_entropy)


