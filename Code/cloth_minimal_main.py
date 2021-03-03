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
from PIL import Image
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect
import sys

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    # Load one task:
    print(inspect.getfile(dm_control))
    env = suite.load(domain_name="cloth_corner", task_name="easy") 
    # action 
    action_spec = env.action_spec()
    # cloth_corner - BoundedArray(shape=(3,), dtype=dtype('float64'), name=None, minimum=[-1. -1. -1.], maximum=[1. 1. 1.])
    # First two terms are (x,y) pick positions and the last term is the place movement
    
    # defining variables
    time_step = env.reset()
    time_step_counter = 0
       
    # reset frames folder
    subprocess.call([ 'rm', '-rf', '-frames' ])
    subprocess.call([ 'mkdir', '-p', 'frames' ])

    while not time_step.last() and time_step_counter < 50000:
        # np.random.uniform(low=action_spec.minimum,
        #                   high=action_spec.maximum,
        #                   size=action_spec.shape)
        action = np.array([0., 0., 1.])
        # action = np.array([0., 0.])
        # print(action)
        time_step = env.step(action)

        image_data = env.physics.render(width = 64, height = 64, camera_id = 0)
        img = Image.fromarray(image_data, 'RGB')
        # current_mask = np.any(image_data < 100 axis=-1).astype(int))

        img.save("frames/frame-%.10d.png" % time_step_counter)
        time_step_counter += 1
        # print(time_step)
        # plt.imshow(img, interpolation="none")
        # plt.show()
        # plt.imshow(current_mask, interpolation="none")
        # plt.show()
        
        # viewer.launch(env, policy = fixed_policy)
    subprocess.call([
                    'ffmpeg', '-framerate', '50', '-y', '-i',
                    'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p', 'video_cloth_minimum.mp4'
    ])

