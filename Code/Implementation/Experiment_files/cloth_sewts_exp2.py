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

from dm_control import viewer


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
# CORNER_INDEX_POSITION=[86,81,59,54]
CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
CORNER_INDEX_POSITION=['G0_0','G0_8','G8_0','G8_8']
INDEX_ACTION=['B0_0','B0_1','B0_2','B0_3','B0_4','B0_5','B0_6','B0_7','B0_8','B1_0','B2_0','B3_0','B4_0','B5_0','B6_0','B7_0','B7_0','B1_1','B1_2','B1_3','B1_4','B1_5','B1_6','B1_7','B1_8','B2_1','B2_2','B2_3','B2_4','B2_5','B2_6','B2_7','B2_8','B3_1','B3_2','B3_3','B3_4','B3_5','B3_6','B3_7','B3_8','B4_1','B4_2','B4_3','B4_4','B4_5','B4_6','B4_7','B4_8','B5_1','B5_2','B5_3','B5_4','B5_5','B5_6','B5_7','B5_8','B6_1','B6_2','B6_3','B6_4','B6_5','B6_6','B6_7','B6_8','B7_1','B7_2','B7_3','B7_4','B7_5','B7_6','B7_7','B7_8','B8_1','B8_2','B8_3','B8_4','B8_5','B8_6','B8_7','B8_8',]
def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""

  return common.read_model('cloth_sewts.xml'),common.ASSETS



W=64

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cloth(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit,n_frame_skip=1,special_task=True, **environment_kwargs)

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
          shape=(3,), dtype=np.float, minimum=[-1.0] * 3, maximum=[1.0] * 3)
  def initialize_episode(self,physics):

    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)

    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
  #     # Support legacy internal code.

      physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))

      index = 0
      
      goal_position = action * 0.05
      corner_action = CORNER_INDEX_ACTION[index]
      corner_geom = CORNER_INDEX_POSITION[index]


      # apply consecutive force to move the point to the target position
      position = goal_position + physics.named.data.geom_xpos[corner_geom]
      dist = position - physics.named.data.geom_xpos[corner_geom]

      loop = 0
      while np.linalg.norm(dist) > 0.025:
        loop += 1
        if loop > 40:
          break
        physics.named.data.xfrc_applied[corner_action, :3] = dist * 20
        physics.step()
        self.after_step(physics)
        dist = position - physics.named.data.geom_xpos[corner_geom]


  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = []
    a = physics.named.data.geom_xpos['G0_0']
    b = physics.named.data.geom_xpos['G0_1']
    c = physics.named.data.geom_xpos['G0_2']
    d = physics.named.data.geom_xpos['G0_3']
    obs_ = np.array ([a,b,c,d])
    obs['position'] = obs_.reshape(-1).astype('float32')
    # obs.append(physics.named.data.geom_xpos['G0_0'].reshape(-1).astype('float32'))
    # print(obs)
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    # DEFINING THE CURRENT POSITIONS OF 4 POINTS, G0_0, G0_1, G0_2, G0_3
    current_pos_G0_0 = physics.named.data.geom_xpos['G0_0']
    current_pos_G0_1 = physics.named.data.geom_xpos['G0_1']
    current_pos_G0_2 = physics.named.data.geom_xpos['G0_2']
    current_pos_G0_3 = physics.named.data.geom_xpos['G0_3']

    # TAKING THE CURRENT x,y POSITIONS OF 4 POINTS NEXT TO A CORNER, INCLUDING THE CORNER
    # Taking Corner point x,y coordinates, G0_0
    x_G0_0 = current_pos_G0_0[0] 
    y_G0_0 = current_pos_G0_0[1]
    # Taking the point fourth to corner, G0_3
    x_G0_3 = current_pos_G0_3[0]
    y_G0_3 = current_pos_G0_3[1]
    # Taking the point next to corner, G0_1
    x_G0_1 = current_pos_G0_1[0]
    y_G0_1 = current_pos_G0_1[1]
    # Taking the point second to corner, G0_2 
    x_G0_2 = current_pos_G0_2[0]
    y_G0_2 = current_pos_G0_2[1]
    
    # DRAWING A VIRTUAL STRAIGHT LINE FOR MATCHING WITH THE
    # Join the corner point and fourth point and get the slope of the line joining the points
    slope = (y_G0_3 - y_G0_0)/ (x_G0_3 - x_G0_0)
    # Defining the radius
    radius = 0.03
    # Defining the virtual line points
    # Point #1
    X_G0_0 = x_G0_0
    Y_G0_0 = y_G0_0
    # Point #2 (Intersection of circle of radius 0.3 and line with the calculated slope)
    X_G0_1 = X_G0_0 + radius * 1/(np.sqrt(slope*slope+1))
    Y_G0_1 = Y_G0_0 + slope * (X_G0_1 - X_G0_0)
    # Point #3 (Intersection of circle of radius 0.6 and line with the calculated slope)
    X_G0_2 = X_G0_0 + 2*radius * 1/(np.sqrt(slope*slope+1))
    Y_G0_2 = Y_G0_0 + slope * (X_G0_2 - X_G0_0)
    # Point #4 (Intersection of circle of radius 0.9 and line with the calculated slope)
    X_G0_3 = X_G0_0 + 3*radius * 1/(np.sqrt(slope*slope+1))
    Y_G0_3 = Y_G0_0 + slope * (X_G0_3 - X_G0_0)
    reward = 0
    array = ([X_G0_0, Y_G0_0, 0.],[X_G0_1, Y_G0_1, 0.],[X_G0_2, Y_G0_2, 0.],[X_G0_3, Y_G0_3, 0.])
    dist_G0_0 = np.sqrt((current_pos_G0_0[0] - X_G0_0)**2 + (current_pos_G0_0[1] - Y_G0_0)**2 + (current_pos_G0_0[2] - 0.)**2) 
    dist_G0_1 = np.sqrt((current_pos_G0_1[0] - X_G0_1)**2 + (current_pos_G0_1[1] - Y_G0_1)**2 + (current_pos_G0_1[2] - 0.)**2) 
    dist_G0_2 = np.sqrt((current_pos_G0_2[0] - X_G0_2)**2 + (current_pos_G0_2[1] - Y_G0_2)**2 + (current_pos_G0_2[2] - 0.)**2)     
    dist_G0_3 = np.sqrt((current_pos_G0_3[0] - X_G0_3)**2 + (current_pos_G0_3[1] - Y_G0_3)**2 + (current_pos_G0_3[2] - 0.)**2)     
    reward = -1 * (dist_G0_0 + dist_G0_1 + dist_G0_2 + dist_G0_3)
    return reward

