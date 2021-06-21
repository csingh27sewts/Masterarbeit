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
CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
CORNER_INDEX_POSITION=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""

  return common.read_model('cloth_corner.xml'),common.ASSETS



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

    self._current_loc = self._generate_loc()
    self._maxq = maxq

    print('random_location', self._random_location, 'pixels_only', self._pixels_only, 'maxq', self._maxq)

    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    # one hot corner + action
    if self._random_location:
      return specs.BoundedArray(
          shape=(3,), dtype=np.float, minimum=[-1.0] * 3, maximum=[1.0] * 3)
          # shape of 3 represents pick point (x,y,z)
          # it is as specified in the paper
    else:
      return specs.BoundedArray(
        shape=(7,), dtype=np.float, minimum=[-1.0] * 7, maximum=[1.0] * 7
        # probably the first 4 terms here now represent corners using one hot encoding, i.e. [1,0,0,0] is for corner 1
        # last 3 terms are the same as above
      )

  def initialize_episode(self,physics):
    if self._random_location or self._maxq:
        self._current_loc = self._generate_loc()

    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])
    
    # B00, B08, B80. B88 Represent 4 corners
    # We have 0 to 8 are the particle name since we have 9 particles in each row and column as specified in xml file
    # B34 and B44 hence probably represent two intermediate particles and a force of -2 in Z direction is applied
    # This is to pinch these points initially
    # xfrc is force acting on the particle. Has 6 terms but only first 3 corresponding to Force (:3) are considered
    # Next 3 (3:6) point to torque

    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image = image
    self.mask = np.any(image < 100, axis=-1).astype(int)

    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)
    # Random forces are applied on all 4 corners in x,y,z direction after anchoring B3_4 and B4_4

    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
  #     # Support legacy internal code.

      physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))

      if self._random_location and not self._maxq:
        index = self._current_loc
        # One of the 4 corners is selected randomly
      else:
        one_hot = action[:4] # selection of one of the 4 corners using one hot encoding, e.g. [0,1,0,0] for corner 2
        index = np.argmax(one_hot) # selects the corner from encoding, e.g. for [0,0,0,1], position 3 has max value and hence index for corner is 3, i.e. last corner
        action = action[4:] # ultimately action is just of size 3 , i.e. x,y,z for pick position

      goal_position = action * 0.05 # multiplying by 0.05 for right scaling perhaps
      # action is a numpy array of size 4 and the next three terms are just x,y,z position of the particle
      corner_action = CORNER_INDEX_ACTION[index]
      corner_geom = CORNER_INDEX_POSITION[index]


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

       
      if self._random_location and not self._maxq:
        self._current_loc = self._generate_loc()



  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image=image

    if self._random_location or self._maxq:
      one_hot = np.zeros(4).astype('float32')
      one_hot[self._current_loc] = 1
      obs['location'] = np.tile(one_hot, 50).reshape(-1).astype('float32')

    if not self._pixels_only:
        obs['position'] = physics.data.geom_xpos[5:,:].reshape(-1).astype('float32')

    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    current_mask = np.any(self.image < 100, axis=-1).astype(int)
    area = np.sum(current_mask * self.mask)
    reward = area / np.sum(self.mask)
    # reward = np.sum(current_mask) # Max spread of the cloth
    # reward = 
    return reward

  def _generate_loc(self):
      return np.random.choice(4)
