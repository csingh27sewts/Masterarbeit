# IMPORT PACKAGES

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
import time 

if __name__ == '__main__':

    # CONDA ENVIRONMENT : 
    # /home/chandandeep/anaconda3/envs/rlpyt/lib/python3.8/site-packages/dm_control/suite/

    # PATH TO DMC2GYM :
    # print(inspect.getfile(dmc2gym)) # built in path

    # DEFINE PATHS
    location_minimal = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Minimal/output"
    location_intermediate = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output"
    location_full = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Full/output"
    path_to_output = location_minimal + '/output'

    # DEFINE ENVIRONMENTS

    ######### SAMPLE  ######################

    # EXPERIMENT #SAMPLE
    # env_id = 'pendulum'
    # env = gym.make('Pendulum-v0')   

    ######### MINIMAL ######################

    # FIXED INITIALIZATION #
    env_id_m_0_1 = 'cloth_sewts_minimal_0_1'
    env_id_m_0_2 = 'cloth_sewts_minimal_0_2'
    
    reward_max_m_0_1 = 499
    reward_max_m_0_2 = 499
    
    location_SAC_m_0_1 = location_minimal + "/SAC_m_0_1/"
    location_m_0_1 = location_SAC_m_0_1 + "cloth_sewts_minimal"

    location_SAC_m_0_2 = location_minimal + "/SAC_m_0_2/"
    location_m_0_2 = location_SAC_m_0_2 + "cloth_sewts_minimal"

    # RANDOM INITIALIZATION #
    env_id_m_1_1 = 'cloth_sewts_minimal_1_1'
    env_id_m_1_2 = 'cloth_sewts_minimal_1_2'
    env_id_m_1_3 = 'cloth_sewts_minimal_1_3'

    reward_max_m_1_1 = 499
    reward_max_m_1_2 = 499
    reward_max_m_1_3 = 499

    location_SAC_m_1_1 = location_minimal + "/SAC_m_1_1/"
    location_m_1_1 = location_SAC_m_1_1 + "cloth_sewts_minimal"

    location_SAC_m_1_2 = location_minimal + "/SAC_m_1_2/"
    location_m_1_2 = location_SAC_m_1_2 + "cloth_sewts_minimal"

    location_SAC_m_1_3 = location_minimal + "/SAC_m_1_3/"
    location_m_1_3 = location_SAC_m_1_3 + "cloth_sewts_minimal"


    # PREMANIPULATION #
    env_id_m_2_1 = 'cloth_sewts_minimal_2_1'

    reward_max_m_2_1 = 499

    location_SAC_m_2_1 = location_minimal + "/SAC_m_2_1/"
    location_m_2_1 = location_SAC_m_2_1 + "cloth_sewts_minimal"

    ######### INTERMEDIATE  ######################

    # FIXED INITIALIZATION #
    env_id_i_0_1 = 'cloth_sewts_intermediate_0_1'
    env_id_i_0_2 = 'cloth_sewts_intermediate_0_2'

    reward_max_i_0_1 = 499
    reward_max_i_0_2 = 499

    location_SAC_i_0_1 = location_intermediate + "/SAC_m_0_1/"
    location_i_1_1 = location_SAC_m_1_1 + "cloth_sewts_minimal"

    location_SAC_i_0_2 = location_intermediate + "/SAC_m_0_2/"
    location_i_0_2 = location_SAC_m_0_2 + "cloth_sewts_minimal"

    # RANDOM INITIALIZATION #
    env_id_i_1_1 = 'cloth_sewts_intermediate_1_1'
    env_id_i_1_2 = 'cloth_sewts_intermediate_1_2'
    env_id_i_1_3 = 'cloth_sewts_intermediate_1_3'

    reward_max_i_1_1 = 499
    reward_max_i_1_2 = 499
    reward_max_i_1_3 = 499

    location_SAC_i_1_1 = location_intermediate + "/SAC_i_1_1/"
    location_i_1_1 = location_SAC_i_1_1 + "cloth_sewts_minimal"

    location_SAC_i_1_2 = location_intermediate + "SAC_i_1_2/"
    location_i_1_2 = location_SAC_i_1_2 + "cloth_sewts_minimal"

    location_SAC_i_1_3 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_1_3/"
    location_i_1_3 = location_SAC_i_1_3 + "cloth_sewts_minimal"


    # PREMANIPULATION #
    env_id_i_2_1 = 'cloth_sewts_intermediate_2_1'

    reward_max_i_2_1 = 499

    location_SAC_i_2_1 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_1/"
    location_i_2_1 = location_SAC_i_2_1 + "cloth_sewts_minimal"

    ######### FULL  ######################

    # FIXED INITIALIZATION #
    env_id_f_0_1 = 'cloth_sewts_full_0_1'
    env_id_f_0_2 = 'cloth_sewts_full_0_2'
    
    reward_max_f_0_1 = 499
    reward_max_f_0_2 = 499

    location_SAC_f_0_1 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Full/output/SAC_f_0_1/"
    location_f_0_1 = location_SAC_f_0_1 + "cloth_sewts_minimal"

    location_SAC_f_0_2 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Full/output/SAC_f_0_2/"
    location_f_0_2 = location_SAC_f_0_2 + "cloth_sewts_minimal"

    # RANDOM INITIALIZATION #
    env_id_f_1_1 = 'cloth_sewts_minimal_1_1'
    env_id_f_1_2 = 'cloth_sewts_minimal_1_2'
    env_id_f_1_3 = 'cloth_sewts_minimal_1_3'

    reward_max_f_1_1 = 499
    reward_max_f_1_2 = 499
    reward_max_f_1_3 = 499

    location_SAC_f_1_1 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_1_1/"
    location_f_1_1 = location_SAC_f_1_1 + "cloth_sewts_minimal"

    location_SAC_f_1_2 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_1_2/"
    location_f_1_2 = location_SAC_f_1_2 + "cloth_sewts_minimal"

    location_SAC_f_1_3 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_1_3/"
    location_f_1_3 = location_SAC_f_1_3 + "cloth_sewts_minimal"


    # PREMANIPULATION #
    env_id_f_2_1 = 'cloth_sewts_minimal_2_1'

    reward_max_f_2_1 = 499

    location_SAC_f_2_1 = "/home/chandandeep/Masterarbeit/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_2_1/"
    location_f_2_1 = location_SAC_f_2_1 + "cloth_sewts_minimal"


    ######################################

    # DEFINE  VARIABLES
    n_games = 50
 
    ######################################

    # SELECT EXPERIMENTS 

    env_id = [env_id_m_0_1]
    reward_set = [reward_max_m_0_1]
    location_set = [location_m_0_1] 


    if not os.path.exists(location_SAC_m_0_1):
        os.mkdir(location_SAC_m_0_1)


    for i in range(len(location_set)): 

        # LOAD EXPERIMENT
        os.chdir(location_SAC_m_0_1)
        env = dmc2gym.make(domain_name=env_id[i], task_name='easy', seed=1) #dmc2gym package
        # DEFINE MODEL

        tensorboard_log = location_SAC_m_0_1 + "/Log"
        if not os.path.exists(tensorboard_log):
            os.mkdir(tensorboard_log)

        model = SAC("MlpPolicy", env, learning_starts=1024, 
                tensorboard_log=tensorboard_log,
                batch_size=256, verbose=1)

        print("##################################################")
        print("ENVIRONMENT",i + 1," /",location_set,"\n")

        for j in range(n_games):
            
            # INITIALIZE VARIABLES

            done = False
            step = 0
            reward = 0

            print("GAME no.",j + 1,"/n/n/n")

            # LOAD MODEL IF ALREADY EXISTING

            if os.path.isfile(location_set[i] + '.zip'):
               model = SAC.load(location_set[i])

            # SET ENVIRONMENT
            model.set_env(env)
            os.chdir(location_SAC_m_0_1)

            # TRAIN MODEL
            model.learn(total_timesteps=150000, reset_num_timesteps = True, callback = stable_baselines_logging.ImageRecorderCallback(), log_interval=1) # log_interval = no. of episodes

            # EXPERIMENT #
            model.save(location_set[i])

            observation = env.reset()  

            for z in range(100):
                env.step(np.array([0.,0.]))

            while done is False :

                
                # MODEL PREDICTION
                
                action, observation_ = model.predict(observation, deterministic=True)
                
                # UPDATE STEPS
                
                step = step + 1
                print("STEP",step,"/n")

                # TAKE STEP

                obs, reward, done, info = env.step(action)
                
                
                # PRINTING RESULTS

                print("Evaluating : step ",step)
                print("observation")
                print(obs)
                print("action")
                print(action)
                print("reward")
                print(reward)

                # UPDATE OBSERVATION

                observation = obs

                # TERMINAL CONDITION
                print(i)
                if step == 500 or reward > reward_set[i]: # EXPERIMENT_1 # EXPERIMENT_0
                    done = True
                    print("\n####################################DONE################################\n")

                # SAVE FIGURES
                
                image = env.render(mode="rgb_array", height=256, width=256) 
                plt.imshow(image)
                # plt.show()
                plt.show(block=False)
                game_folder = "Game_" + str(j+1)
                if not os.path.exists(game_folder):
                    os.mkdir(game_folder)
                game_location = location_SAC_m_0_1 + game_folder
                name = game_location + "/" + str(time.time()) + ".png"
                plt.savefig(name)
                print("Done /n")
            j = j + 1
        i = i + 1
