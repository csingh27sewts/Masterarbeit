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

"""Planar Stacker domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import mujoco
from dm_env import specs
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools
import os
from imageio import imsave
from PIL import Image,ImageColor
from lxml import etree
import numpy as np
import math
_TOL = 1e-13
_CLOSE = .01    # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .02  # (Seconds)
_TIME_LIMIT = 30  # (Seconds)



CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
CORNER_INDEX_GEOM=['G0_0','G0_8','G8_0','G8_8']



SUITE = containers.TaggedTasks()

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cloth_gripper.xml'), common.ASSETS

@SUITE.add('hard')
def easy(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns stacker task with 2 boxes."""

  physics=Physics.from_xml_string(*get_model_and_assets())

  task = Stack(randomize_gains=False,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP,special_task=True,time_limit=time_limit,
      **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics with additional features for the Planar Manipulator domain."""


class Stack(base.Task):
  """A Stack `Task`: stack the boxes."""

  def __init__(self, randomize_gains, random=None):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self.current_loc = np.zeros((2,))

    super(Stack, self).__init__(random=random)

  def initialize_episode(self, physics):
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = np.random.uniform(-.3, .3, size=3)
    super(Stack, self).initialize_episode(physics)


  def action_spec(self, physics):
    """Returns a `BoundedArray` matching the `physics` actuators."""

    return specs.BoundedArray(
      shape=(3,), dtype=np.float, minimum=[-1.0,-1.0,-1.0] , maximum=[1.0,1.0,1.0])

  def before_step(self, action, physics):

    """Sets the control signal for the actuators to values in `action`."""
    # Support legacy internal code.

    # clear previous xfrc_force
    physics.named.data.xfrc_applied[:, :3] = np.zeros((3,))

    # scale the position to be a normal range

    goal_position = action[:3]
    location = self.current_loc

    goal_position = goal_position * 0.1

    # computing the mapping from geom_xpos to location in image
    cam_fovy = physics.model.cam_fovy[0]
    f = 0.5 * 64 / math.tan(cam_fovy * math.pi / 360)
    cam_matrix = np.array([[f, 0, 64 / 2], [0, f, 64 / 2], [0, 0, 1]])
    cam_mat = physics.data.cam_xmat[0].reshape((3, 3))
    cam_pos = physics.data.cam_xpos[0].reshape((3, 1))
    cam = np.concatenate([cam_mat, cam_pos], axis=1)
    cam_pos_all = np.zeros((86, 3, 1))
    for i in range(86):
        geom_xpos_added = np.concatenate([physics.data.geom_xpos[i], np.array([1])]).reshape((4, 1))
        cam_pos_all[i] = cam_matrix.dot(cam.dot(geom_xpos_added)[:3])
    # position of each particle calculated with respect to camera by multuolying with camera matrix perhaps

    # cam_pos_xy=cam_pos_all[5:,:]
    cam_pos_xy=np.rint(cam_pos_all[:,:2].reshape((86,2))/cam_pos_all[:,2])
    cam_pos_xy=cam_pos_xy.astype(int)
    cam_pos_xy[:,1]=64-cam_pos_xy[:,1]

    # hyperparameter epsilon=3(selecting 3 nearest joint) and select the point
    epsilon=3
    possible_index=[]
    possible_z=[]
    for i in range(86):
        #flipping the x and y to make sure it corresponds to the real location
        if abs(cam_pos_xy[i][0]-location[0,1])<epsilon and abs(cam_pos_xy[i][1]-location[0,0])<epsilon and i>4:
            possible_index.append(i)
            possible_z.append(physics.data.geom_xpos[i,2])

    if possible_index != [] :
        index=possible_index[possible_z.index(max(possible_z))]

        corner_action = index-4
        corner_geom = index

        # apply consecutive force to move the point to the target position
        position=goal_position+physics.named.data.geom_xpos[corner_geom]
        dist = position-physics.named.data.geom_xpos[corner_geom]

        loop=0
        while np.linalg.norm(dist)>0.025:
          loop+=1
          if loop >40:
            break
          physics.named.data.xfrc_applied[corner_action,:3]=dist*20
          physics.step()
          self.after_step(physics)
          dist=position-physics.named.data.geom_xpos[corner_geom]

  def get_observation(self, physics):
    """Returns either features or only sensors (to be used with pixels)."""
    obs = collections.OrderedDict()
    self.current_loc = self.sample_location(physics)
    obs['force_location'] = np.tile(self.current_loc, 50).reshape(-1).astype('float32')
    return obs

  def sample_location(self,physics):
    # obs=self.get_observation(physics)
    render_kwargs={}
    render_kwargs['camera_id']=0
    render_kwargs['width'] = 64
    render_kwargs['height'] = 64
    image=physics.render(**render_kwargs)
    location_range = np.transpose(np.where(~np.all(image > 100, axis=2)))
    num_loc=np.shape(location_range)[0]
    index=np.random.randint(num_loc,size=1)
    location=location_range[index]

    return location

  def get_reward(self,physics):
      dist_sum=0
      for i in range(9):
          for j in range(9):
              index='G'+str(i)+'_'+str(j)
              geom_dist=np.sum(abs(physics.named.data.geom_xpos[index]-np.array([-0.09+0.03*i,-0.15+0.03*j,0])))
              # geom_dist = sum of x,y,z coordinates after subtracting a specific value from x,y,z
              # the amount to be subtracted is 0 for the element G35 as i =3 and j =5
              # for particles below term 35, overall sum is positive and it is negative for particle greater than 35
              dist_sum += geom_dist
      dist_sum = dist_sum/81
      # goal is to minimize this overall normalized sum
      # ideally this should be 0
      # taking the ideal goal state into account which is represented by a flat cloth
      # G00 is probably at -0.09,-0.15, 0 so its sum is 0
      # G88 is probably at 0.15, 0.09, 0
      # G35 is probably at 0, 0 ,0
      # It is measured from G35 because probably the camera is at G35
      return -dist_sum, dict()
