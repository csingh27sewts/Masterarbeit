# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
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
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
import random
import mujoco_py

"""Input action and location, location maps [0,1] grid onto nearest joint on deformed cloth
   Allows diagonal or area reward"""

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
CORNER_INDEX_POSITION=[86,81,59,54]
CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
GEOM_INDEX=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cloth_v4.xml'),common.ASSETS

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cloth(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, n_frame_skip=1, special_task=True, **environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

class Cloth(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, pixel_size=64, camera_id=0,
               reward='diagonal', mode='corners', distance_weight=0.0, eval=False):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """

    assert reward in ['diagonal', 'area']
    self._randomize_gains = randomize_gains
    self.pixel_size = pixel_size
    self.camera_id = camera_id
    self.reward = reward
    self.distance_weight = distance_weight
    self.mode = mode
    print('pixel_size', self.pixel_size, 'camera_id', self.camera_id,
          'reward', self.reward, 'distance_weight', self.distance_weight,
          'mode', self.mode)
    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArray` matching the `physics` actuators."""
    return specs.BoundedArray(
        shape=(5,), dtype=np.float, minimum=[-1.0,-1.0,-1.0,-1.0,-1.0] , maximum=[1.0,1.0,1.0,1.0,1.0] )

  def initialize_episode(self,physics):
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)

    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
      physics.named.data.xfrc_applied[1:, :3] = np.zeros((3,))
      action_force = action[2:]
      action_position = (action[:2] * 0.5 + 0.5) * 8 # [-1, 1] -> [0, 1] -> [0, 8]

      assert len(action_force) == 3
      assert len(action_position) == 2

      action_position = action_position[None, :]
      grid_points = self._get_grid_points()
      distances = np.linalg.norm(grid_points - action_position, axis=-1)
      idx = np.argmin(distances)

      x, y = grid_points[idx]
      x, y = int(x), int(y)

      force_id = 'B{}_{}'.format(x, y)
      physics.named.data.xfrc_applied[force_id, :3] = 5 * action_force

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.data.geom_xpos[6:, :2].astype('float32').reshape(-1)
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    if self.reward == 'area':
        pixels = physics.render(width=self.pixel_size, height=self.pixel_size,
                                camera_id=self.camera_id)
        segmentation = (pixels < 100).any(axis=-1).astype('float32')
        reward = segmentation.mean()
        return reward, dict()
    elif self.reward == 'diagonal':
        pos_ll = physics.data.geom_xpos[86,:2]
        pos_lr = physics.data.geom_xpos[81,:2]
        pos_ul = physics.data.geom_xpos[59,:2]
        pos_ur = physics.data.geom_xpos[54,:2]

        diag_dist1 = np.linalg.norm(pos_ll - pos_ur)
        diag_dist2 = np.linalg.norm(pos_lr - pos_ul)

        diagonal_reward = diag_dist1 + diag_dist2
        distance_reward = -self.distance_weight * np.linalg.norm(physics.named.data.geom_xpos['G4_4', :2])
        reward = diagonal_reward + distance_reward
        return reward, dict()

    raise ValueError(self.reward)

  def _get_grid_points(self):
      if self.mode == 'corners':
          grid_points = np.array([[0, 0], [0, 8], [8, 0], [8, 8]], dtype='float32')
      elif self.mode == 'border':
          grid_points = np.mgrid[0:9, 0:9].reshape(2, 81).T.astype('float32')
          grid_points = grid_points[((grid_points == 0) | (grid_points == 8)).any(axis=-1)]
          assert len(grid_points) == 32
      elif self.mode == '3x3':
          grid_points = np.mgrid[0:9:4, 0:9:4].reshape(2, 9).T.astype('float32')
          assert len(grid_points) == 9
      elif self.mode == '5x5':
          grid_points = np.mgrid[0:9:2, 0:9:2].reshape(2, 25).T.astype('float32')
          assert len(grid_points) == 25
      elif self.mode == 'normal':
          grid_points = np.mgrid[0:9, 0:9].reshape(2, 81).T.astype('float32')
          assert len(grid_points) == 81
      elif self.mode == 'inner_border':
          grid_points = np.mgrid[1:8, 1:8].reshape(2, 49).T.astype('float32')
          grid_points = grid_points[((grid_points == 1) | (grid_points == 7)).any(axis=-1)]
          assert len(grid_points) == 24
      else:
          raise Exception(self.mode)
      return grid_points
