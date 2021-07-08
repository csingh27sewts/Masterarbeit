# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,/
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
import numpy as np

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
# CORNER_INDEX_POSITION=[86,81,59,54]
CORNER_INDEX_ACTION=['B0_0','B0_2','B2_0','B2_2']
CORNER_INDEX_POSITION=['G0_0','G0_2','G2_0','G2_2']

SEAM_INDEX_ACTION=['B0_0','B0_1','B0_2']
SEAM_INDEX_POSITION=['G0_0','G0_1','G0_2']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""

  return common.read_model('cloth_sewts_minimal_raw.xml'),common.ASSETS


W=64

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cloth(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""



class Cloth(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, random_location=False,
               pixels_only=False, maxq=False):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._pixels_only = pixels_only

    self._maxq = maxq

    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    # one hot corner + action
    return specs.BoundedArray(
        shape=(6,), dtype=np.float, minimum=[-1.0] * 6, maximum=[1.0] * 6
        # probably the first 3 terms here now represent seam points using one hot encoding, i.e. [1,0,0,0] is for corner 1
        # last 3 terms are the same as above
      )

  def initialize_episode(self,physics):
    physics.named.data.xfrc_applied['B1_1', :3] = np.array([0,0,-2])
    # physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)

    for i in range(0,50):
        physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:2] = np.random.uniform(-.5,.5,size=2) * 5
        physics.step()
    super(Cloth, self).initialize_episode(physics)
  
  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
  #     # Support legacy internal code.

      physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))


      one_hot = action[:3] # selection of one of the 4 corners using one hot encoding, e.g. [0,1,0,0] for corner 2
      index = np.argmax(one_hot) # selects the corner from encoding, e.g. for [0,0,0,1], position 3 has max value and hence index for corner is 3, i.e. last corner
      action = action[3:] # ultimately action is just of size 3 , i.e. x,y,z for pick position

      goal_position = action * 0.05 * 0.5 # multiplying by 0.05 for right scaling perhaps
      # action is a numpy array of size 4 and the next three terms are just x,y,z position of the particle
      corner_action = SEAM_INDEX_ACTION[index]
      corner_geom = SEAM_INDEX_POSITION[index]

      # apply consecutive force to move the point to the target position
      position = goal_position + physics.named.data.geom_xpos[corner_geom]
      # geom_xpos gives the x, y, z position of the selected corner point
      # add the movement of the corner given by goal_position (scaled action) to the current corner position
      dist = position - physics.named.data.geom_xpos[corner_geom]
      # distance movement
      loop = 0
      # while Frobenius norm or 2-norm is greater than a certain value
      # Frobenius norm is just taking squares of each of the term, adding them and taking a square root
      # https://www.youtube.com/watch?v=yiSKsLcniGw
      # https://youtu.be/Gt56YxMBlVA
      # We are taking this because we want to ensure that the distance moved is significant, i.w. freater than a certain minimum amount
      # and we represent it using this term
      while np.linalg.norm(dist) > 0.025:
        loop += 1
        if loop > 40:
        # why loop ?
          break
        physics.named.data.xfrc_applied[corner_action, :3] = dist * 20
        # Force applied on the corner proportional to the distance to be moved, i.e. more force for more distance
        physics.step()
        # get observations after applying actions, new position perhaps
        self.after_step(physics)
        dist = position - physics.named.data.geom_xpos[corner_geom]


  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()  
    obs['position'] = [] 
    a = physics.named.data.geom_xpos['G0_0'] 
    b = physics.named.data.geom_xpos['G0_1'] 
    c = physics.named.data.geom_xpos['G0_2'] 
    obs_ = np.array ([a,b]) 
    obs['position'] = obs_.reshape(-1).astype('float32') 
    # print(obs) 
    return obs 

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    x_G00 = physics.named.data.geom_xpos['G0_0'][0]
    y_G00 = physics.named.data.geom_xpos['G0_0'][1]
    z_G00 = physics.named.data.geom_xpos['G0_0'][2] 

    x_G01 = physics.named.data.geom_xpos['G0_1'][0]
    y_G01 = physics.named.data.geom_xpos['G0_1'][1]
    z_G01 = physics.named.data.geom_xpos['G0_1'][2] 
 
    dist1 = (z_G00 - 0.017) **2
    dist2 = (z_G01 - 0.017) **2
    
    reward1 = - 1000 * np.sqrt(dist1)
    reward2 = - 1000 * np.sqrt(dist2)

    reward = reward1 + reward2

    return reward

