# the following 3 lines are helpful if you have multiple GPUs and want to train
# agents on multiple GPUs. I do this frequently when testing.
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import dm_control
from dm_control import suite
from dm_control import viewer
from dm_control.suite.wrappers import pixels
from dm_env import specs
from PIL import Image
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect

from SAC.sac_torch import Agent
from SAC.utils import plot_learning_curve
import os
import torch as T

if __name__ == '__main__':
    
    # Load environment
    env_id = 'cloth_v0'
    env = suite.load(domain_name=env_id, task_name="easy")
    print("TEST")
    action_space = env.action_spec()
    action_shape = action_space.shape[0]
    observation_space = env.observation_spec()
    observation_shape = observation_space['position'].shape
    # SAC Agent 
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims= observation_shape, tau=0.005,
                env=env, batch_size=256, layer1_size=256, layer2_size=256,
                n_actions=action_shape)
    n_games = 50000

    # Define filename to store plots
    filename = env_id + '_'+ str(n_games) + 'games_scale' + str(agent.scale) + \
                    '_clamp_on_sigma.png'
    figure_file = 'SAC/plots/' + filename
    figure_file = os.path.join(os.getcwd(), figure_file)

    # reset frames folder
    subprocess.call([ 'rm', '-rf', '-frames' ])
    subprocess.call([ 'mkdir', '-p', 'frames' ])

    best_score = -1*float('inf') # lower reward range
    score_history = []
    
    load_checkpoint = False
    # display 
    if load_checkpoint:
        agent.load_models()
        viewer.launch(env)
    
    steps = 0        
    
    # reset frames folder
    subprocess.call([ 'rm', '-rf', '-frames' ])
    subprocess.call([ 'mkdir', '-p', 'frames' ])

    done = False
    score = 0

    action_spec = env.action_spec()
    time_step = env.reset()
    time_step_counter = 0
    observation = time_step.observation['position']
 
    while not time_step.last() and time_step_counter < n_games:
        
            print("TIME STEP")
            print(time_step)
            action = np.random.uniform(action_spec.minimum,
                                action_spec.maximum,
                                size = action_spec.shape)
            
            time_step = env.step(action)
           
            observation_ = time_step.observation['position']
            print(observation_)
            reward = time_step.reward

            image_data = env.physics.render(width = 64, height = 64, camera_id = 0)
            img = Image.fromarray(image_data, 'RGB')
            img.save("frames/frame-%.10d.png" % time_step_counter)
            time_step_counter += 1

            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            # Each call to an environment’s step() method returns a TimeStep namedtuple with
            # step_type, reward, discount and observation fields
            score += time_step.reward #reward
            observation = observation_ # update observation
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
            print('episode ', time_step_counter, 'score %.1f' % score,
                'trailing 100 games avg %.1f' % avg_score, 
                'steps %d' % steps, env_id, 
                ' scale ', agent.scale)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        print("SCORE")
        print(score_history)
        print(x)
        plot_learning_curve(x, score_history, figure_file)

    filename = str(n_games)
    subprocess.call([
                    'ffmpeg', '-framerate', '50', '-y', '-i',
                    'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p','video_sac.mp4' # TODO to update name of file
            ])

