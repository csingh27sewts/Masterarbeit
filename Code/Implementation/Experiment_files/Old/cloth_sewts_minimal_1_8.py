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
from random import randrange
from dm_control import viewer

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
CORNER_INDEX_ACTION=['B0_0','B0_2','B2_0','B2_2']
CORNER_INDEX_POSITION=['G0_0','G0_2','G2_0','G2_2']
INDEX_ACTION=['B0_0','B0_1','B0_2','B1_0','B2_0','B1_1','B1_2','B2_1','B2_2']
INDEX_POSITION=['G0_0','G0_1','G0_2','G1_0','G2_0','G1_1','G1_2','G2_1','G2_2']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""

  return common.read_model('cloth_sewts_full.xml'),common.ASSETS

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

  def __init__(self, randomize_gains, random=None, random_location=True,
               pixels_only=False, maxq=False):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._random_location = random_location
    self._pixels_only = pixels_only

    self._maxq = maxq

    print('random_location', self._random_location, 'pixels_only', self._pixels_only, 'maxq', self._maxq)

    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    # LEFT CORNER ACTION [x,y,z]
    return specs.BoundedArray(
          shape=(2,), dtype=np.double, minimum=[-1.0] * 2, maximum=[1.0] * 2)

  def initialize_episode(self,physics):
    point = randrange(len(INDEX_ACTION))
    physics.named.data.xfrc_applied[INDEX_ACTION[point], :3] = np.array([0, 0, -2])
    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
  #     # Support legacy internal code.

      physics.named.data.xfrc_applied[:,:3]= np.zeros((3,))
      # physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3] = np.random.uniform(-.5,.5,size=3)

      index = 0
      print("action")
      print(action)
      goal_position = action * 0.05
      corner_action = INDEX_ACTION[index]
      corner_geom = INDEX_POSITION[index]

      # apply consecutive force to move the point to the target position
      position = goal_position + physics.named.data.geom_xpos[corner_geom][:2]
      dist = position - physics.named.data.geom_xpos[corner_geom][:2]

      loop = 0
      while np.linalg.norm(dist) > 0.025:
        loop += 1
        if loop > 40:
          break
        physics.named.data.xfrc_applied[corner_action, :2] = dist * 20
        physics.step()
        self.after_step(physics)
        dist = position - physics.named.data.geom_xpos[corner_geom][:2]

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict() 
    obs['position'] = []
    a = physics.named.data.geom_xpos['G0_0']
    b = physics.named.data.geom_xpos['G0_1']
    c = physics.named.data.geom_xpos['G0_2']
    d = physics.named.data.geom_xpos['G0_3']
    e = physics.named.data.geom_xpos['G0_4']
    obs_ = np.array ([a,b,c,d,e])
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

    x_G02 = physics.named.data.geom_xpos['G0_2'][0]
    y_G02 = physics.named.data.geom_xpos['G0_2'][1]
    z_G02 = physics.named.data.geom_xpos['G0_2'][2]

 
    x_G03 = physics.named.data.geom_xpos['G0_3'][0]
    y_G03 = physics.named.data.geom_xpos['G0_3'][1]
    z_G03 = physics.named.data.geom_xpos['G0_3'][2]

    x_G04 = physics.named.data.geom_xpos['G0_4'][0]
    y_G04 = physics.named.data.geom_xpos['G0_4'][1]
    z_G04 = physics.named.data.geom_xpos['G0_4'][2]

 
    dist1 = (z_G00 - 0.017) **2
    dist2 = (z_G01 - 0.017) **2
    dist3 = (z_G02 - 0.017) **2
    dist4 = (z_G03 - 0.017) **2
    dist5 = (z_G04 - 0.017) **2

    reward1 = - 1000 * np.sqrt(dist1)
    reward2 = - 1000 * np.sqrt(dist2)
    reward3 = - 1000 * np.sqrt(dist3)
    reward4 = - 1000 * np.sqrt(dist4)
    reward5 = - 1000 * np.sqrt(dist5)

    reward = reward1 + reward2 + reward3 + reward4 + reward5

