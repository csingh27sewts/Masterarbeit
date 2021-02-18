from dm_control import suite
from dm_control import viewer
from dm_control.suite.wrappers import pixels
from dm_env import specs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load one task:
    env = suite.load(domain_name="cloth_v0", task_name="easy")
    print('Env',env,'\n')
    
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    
    print('Observation',observation_spec,'\n')
    print('Action' ,action_spec,'\n')
    
    pixels = env.physics.render(width = 64, height = 64, camera_id = 0)
    
    plt.imshow(pixels, interpolation="none")
    plt.show()
    
    #env.reset()
    #print(time_step)
    #done = False
    
    #while not done:
    #  action = np.random.uniform(action_spec.minimum,
    #                             action_spec.maximum,
    #                             size=action_spec.shape)
    #  env.step(action)
    #  viewer.launch(env)
    #  print(env.step.reward, env.step.discount, env.step.observation)
