"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import numpy as np
import glfw

from abr_control.interfaces.mujoco import Mujoco
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.controllers import path_planners
from abr_control.utils import transformations
from SAC.networks import ActorNetwork
import torch 
from Packages.dm_control.dm_control import suite

# initialize our robot config for the jaco2
robot_config = arm("ur5", use_sim_state=False)


# Define model
action_space = env.action_spec()
action_shape = action_space.shape[0]

# Define observation space and observation shape
observation_space = env.observation_spec()
observation_shape = observation_space['position'].shape
print(observation_shape)
# Initialize SAC Agent 
agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
            input_dims= observation_shape, tau=0.005,
            env=env, batch_size=256, layer1_size=256, layer2_size=256,
            n_actions=action_shape)


observation = [0.]*12
model = Age
model.load_state_dict(torch.load('cloth_sewts_exp2_2_actor_sac.ckpt'))
model.eval()


# create our path planner
n_timesteps = 2000
path_planner = path_planners.InverseKinematics(robot_config)

# create our interface
dt = 0.001
interface = Mujoco(robot_config, dt=dt)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)
feedback = interface.get_feedback()
# feedback = [0.,0.57,-0.57,0.57,0.,0.]

try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")

    count = 0
    while 1:
        #if interface.viewer.exit:
        #    glfw.destroy_window(interface.viewer.window)
        #    break

        if count % n_timesteps == 0:
            feedback = interface.get_feedback()
            target_xyz = np.array(
                [
                    0.50,0.30,0.
                    #np.random.random() * 0.5 - 0.25,
                    #np.random.random() * 0.5 - 0.25,
                    #np.random.random() * 0.5 + 0.5,
                ]
            )
            target_ori = [0.,-1.57,0.]
            R = robot_config.R("EE", q=feedback["q"])
            target_orientation = transformations.euler_from_matrix(R, "sxyz")
            # update the position of the target
            interface.set_mocap_xyz("target", target_xyz)

            # can use 3 different methods to calculate inverse kinematics
            # see inverse_kinematics.py file for details
            path_planner.generate_path(
                position= np.array([0.,0.57,-0.57,0.57,0.,0.]), #feedback["q"],
                target_position=np.hstack([target_xyz, target_ori]),
                method=3,
                dt=0.005,
                n_timesteps=n_timesteps,
                plot=False,
            )

        # returns desired [position, velocity]
        target = path_planner.next()[0]

        # use position control
        print("target angles: ", target[: robot_config.N_JOINTS])
        interface.send_target_angles(target[: robot_config.N_JOINTS])
        interface.viewer.render()

        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")
