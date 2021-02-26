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

if __name__ == '__main__':
    # Load one task:
    env = suite.load(domain_name="cloth_v0", task_name="easy") 
    # action 
    action_spec = env.action_spec()
    print('Printing location')
    print(inspect.getfile(dm_control))
    time_step = env.reset()

    time_step_counter = 0
    
    # reset frames folder
    subprocess.call([ 'rm', '-rf', '-frames' ])
    subprocess.call([ 'mkdir', '-p', 'frames' ])


    while not time_step.last() and time_step_counter < 5000000:
        action = np.random.uniform(action_spec.minimum, 
                                action_spec.maximum, 
                                size = action_spec.shape)
        print(action)
        time_step = env.step(action)

        image_data = env.physics.render(width = 64, height = 64, camera_id = 0)
        img = Image.fromarray(image_data, 'RGB')
        img.save("frames/frame-%.10d.png" % time_step_counter)
        time_step_counter += 1
        print(time_step)
        #plt.imshow(img, interpolation="none")
        #plt.show()
        #viewer.launch(env)
    subprocess.call([
                    'ffmpeg', '-framerate', '50', '-y', '-i',
                    'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p', 'video_name.mp4'
    ])
