# IMPORT DM CONTROL LIBRARIES
import dm_control
from dm_control import suite
from dm_control import viewer
from dm_control.suite.wrappers import pixels
from dm_control.suite.cloth_corner import Cloth
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_env import specs

# IMPORT GENERAL LIBRARIES
from PIL import Image
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect
import sys
import time

np.set_printoptions(threshold=sys.maxsize)

# NOTE : 
# THIS IS JUST A SKELETON ALGORITHM CONTAINING STATES, ACTIONS, REWARDS
# IT TAKES ACTIONS AND OBSERVES STATES AND REWARDS
# IT DOES NOT REACH TOWARDS A GOAL STATE (DOES NOT TRY TO MAXIMIZE REWARDS)
# IT IS USEFUL SIMPLY TO TEST WHETHER THE STATES, ACTIONS AND REWARDS WORK AS DEFINED IN DOMAIN
# IT CAN ALSO BE USED TO TRY ANY RANDOM OPERATIONS LIKE APPLYING FORCES, ETC.

if __name__ == '__main__':
    
    # LOAD TASK
    print(inspect.getfile(dm_control))
    env = suite.load(domain_name="cloth_sewts_minimal_2_4", task_name="easy") 
    # env = suite.load(domain_name="cloth_sewts_exp2", task_name="easy") 

    # DEFINE ACTION SPEC 
    action_spec = env.action_spec()
    
    # DEFINE VARIABLES
    time_step = env.reset()
    time_step_counter = 0
    reward = []

    # GETTING X,Y,Z POSITIONS OF PARTICLES
    # print(len(env.physics.named.data.geom_xpos[:, :2]))
    # APPLYING FORCE ON A RANDOM PARTICLE
    # env.physics.named.data.xfrc_applied['B3_4', :3] = np.array([-2,-2,-2])
    # VISUALIZE ON VIEWER
    # viewer.launch(env)

    # RESET FRAMES FOLDER

    subprocess.call([ 'rm', '-rf', '-frames_minimal_2' ])
    subprocess.call([ 'mkdir', '-p', 'frames_minimal_2' ])


    # LOOP TILL time_step_counter
    while not time_step.last() and time_step_counter < 10000:
        # TAKE RANDOM ACTION IN X,Y DIRECTION
        action = np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)
        print("ACTION")
        # action[2] = 0.

        # ACTION FOR 4 PARTICLES (1 CORNER + ITS NEIGHBOURS)
        # action = np.random.uniform(low=action_spec.minimum,
        #                   high=action_spec.maximum,
        #                   size=action_spec.shape)
        # action[2] = 0.
        # action[5] = 0.
        # action[8] = 0.
        # action[12] = 0.

        # SHOW ACTION TAKEN IN EACH STEP
        print("ACTION")
        print(action)

        # TAKE STEP WITH SPECIFIED ACTION AND OBSERVE STATES AND REWARDS
        time_step = env.step(action)
        
        # SHOW REWARD RECEIVED IN EACH STEP
        print("REWARD")
        print(time_step.reward)

        # APPEND ALL REWARDS TO VISUALIZE LATER
        reward.append(time_step.reward)
        
        # SAVE IMAGE FRAMES OF THE ENVIRONMENT
        image_data = env.physics.render(width = 640, height = 480)
        img = Image.fromarray(image_data, 'RGB')
        img.save("frames_minimal_2/frame-%.10d.png" % time_step_counter)
        # img.show()
        # UPDATE TIME STEP
        time_step_counter += 1
        # LAUNCH WITH A FIXED POLICY (IF DEFINED)
        # viewer.launch(env)
        # viewer.launch(env, policy = fixed_policy)

    # PRINTING ALL REWARDS TOGETHER
    print("All Rewards")
    print(reward)

    # PLOT REWARDS 
    time_steps = [i for i in range(time_step_counter)]
    plt.plot(time_steps, reward)
    plt.show()
    plt.savefig('plot.jpg')
    # MAKE A VIDEO
    subprocess.call([
                    'ffmpeg', '-framerate', '50', '-y', '-i',
                    'frames_minimal_2/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p', 'video_cloth_minimum_2.mp4'
    ])

