# Log for understanding rlpyt

Script : /home/chandandeep/Masterarbeit_ws/src/Masterarbeit/Packages/rlpyt/rlpyt/experiments/scripts/dm_control/qpg/sac/launch/launch_dm_control_sac_pixels_cloth_sim.py

Variables initialized - 

script = "rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac.py"
affinity_code = '20cpu_4gpu_4hto'
runs_per_setting = 2
default_config_key = "sac_pixels_cloth_sim"
experiment_title = "sac_dm_control_pixels_cloth_sim"
domains = ['cloth_sim']
tasks = ['easy']
values = [('cloth_sim', 'easy')]
dir_names = ['domain_cloth_sim_task_easy']
keys = [('env', 'domain'), ('env', 'task')]
variants = [{'env': {'domain': 'cloth_sim', 'task': 'easy'}}]
log_dirs = ['domain_cloth_sim_task_easy']
exp_dir = '/home/chandandeep/.mujoco/rlpyt/data/local/20210219/sac_dm_control_pixels_cloth_sim'

Calls function run_experiments in Script - /home/chandandeep/Masterarbeit_ws/src/Masterarbeit/Packages/rlpyt/rlpyt/utils/launching/exp_launcher.py

------------------------------------------

Script : /home/chandandeep/Masterarbeit_ws/src/Masterarbeit/Packages/rlpyt/rlpyt/utils/launching/exp_launcher.py

n_run_slots = 4 
exp_dir = /home/chandandeep/.mujoco/rlpyt/data/local/20210219/sac_dm_control_pixels_cloth_sim
procs = [None, None, None, None]
common_args = ('sac_pixels_cloth_sim',)
runs_args = [()]
num_launched = 0, 1
total = 2
variant = {'env': {'domain': 'cloth_sim', 'task': 'easy'}}
log_dir = '/home/chandandeep/.mujoco/rlpyt/data/local/20210219/sac_dm_control_pixels_cloth_sim/domain_cloth_sim_task_easy'
run_args = domain_cloth_sim_task_easy
run_slot = 0,1,2,3
args = (('sac_pixels_cloth_sim',),)
slot_affinity_code = '3slt_20cpu_4gpu_4hto'
cpus = '15,16,17,18,19,19,20,21,22,23'
run_ID = 0, 1

Call_list = ['taskset', '-c', '15,16,17,18,19,19,20,21,22,23', 'python', 'rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac.py', '3slt_20cpu_4gpu_4hto', '/home/chandandeep/.mujoco/rlpyt/data/local/20210219/sac_dm_control_pixels_cloth_sim/domain_cloth_sim_task_easy', '0', "('sac_pixels_cloth_sim',)"]

# Note that the script call_list runs the python file specified with the arguments given

p = subprocess.Popen(call_list) (This is the main launch command. p is returned after launching each experiment. )
taskset: failed to set pid 27589's affinity: Invalid argument

Calls script : rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac.py


---------------------------------------------

Script : rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac.py

affinity = {'all_cpus': (15, 16, 17, 18, 19, 19, 20, 21, 22, 23), 'master_cpus': (15, 16, 17, 18, 19, 19, 20, 21, 22, 23), 'workers_cpus': ((15,), (16,), (17,), (18,), (19,), (19,), (20,), (21,), (22,), (23,)), 'master_torch_threads': 5, 'worker_torch_threads': 1, 'alternating': True, 'set_affinity': True, 'cuda_idx': 3}

config = {'sac_module': 'sac_v2', 'sac_agent_module': 'sac_agent_v2', 'name': '', 'agent': {'ModelCls': 'PiConvModel', 'QModelCls': 'QofMuConvModel', 'q_model_kwargs': {'channels': (64, 64, 64), 'kernel_sizes': (3, 3, 3), 'strides': (2, 2, 2), 'hidden_sizes': [256, 256], 'n_tile': 20}, 'model_kwargs': {'channels': (64, 64, 64), 'kernel_sizes': (3, 3, 3), 'strides': (2, 2, 2), 'hidden_sizes': [256, 256]}}, 'algo': {'discount': 0.99, 'batch_size': 1024, 'target_update_tau': 0.005, 'target_update_interval': 1, 'learning_rate': 0.0006, 'reparameterize': True, 'policy_output_regularization': 0.0, 'reward_scale': 1, 'replay_ratio': 128}, 'model': {}, 'optim': {}, 'runner': {'n_steps': 200000.0, 'log_interval_steps': 10000.0}, 'sampler': {'is_pixel': True, 'batch_T': 1, 'batch_B': 16, 'max_decorrelation_steps': 0, 'eval_n_envs': 10, 'eval_max_steps': 20000, 'eval_max_trajectories': 50}, 'env': {'domain': 'cloth_v8', 'task': 'easy', 'max_path_length': 30, 'pixel_wrapper_kwargs': {'observation_key': 'pixels', 'pixels_only': True, 'render_kwargs': {'width': 64, 'height': 64}}}}


sac_module = 'rlpyt.algos.qpg.sac_v2'
sac_agent_module = 'rlpyt.agents.qpg.sac_agent_v2'

sac_module = importlib.import_module(sac_module)#
sac_agent_module = importlib.import_module(sac_agent_module)
# <module 'rlpyt.algos.qpg.sac_v2' from '/home/chandandeep/.mujoco/rlpyt/rlpyt/algos/qpg/sac_v2.py'>
# <module 'rlpyt.agents.qpg.sac_agent_v2' from '/home/chandandeep/.mujoco/rlpyt/rlpyt/agents/qpg/sac_agent_v2.py'>

SAC = sac_module.SAC
SacAgent = sac_agent_module.SacAgent
# <class 'rlpyt.algos.qpg.sac_v2.SAC'>
# <class 'rlpyt.agents.qpg.sac_agent_v2.SacAgent'>
agent = <rlpyt.agents.qpg.sac_agent_v2.SacAgent object at 0x7feb80a66e50>
algo = <rlpyt.algos.qpg.sac_v2.SAC object at 0x7feb34bd8810>
sampler = <rlpyt.samplers.parallel.cpu.sampler.CpuSampler object at 0x7feb05df6ed0>
runner = <rlpyt.runners.minibatch_rl.MinibatchRlEval object at 0x7feb34bae3d0>
name = 'sac_cloth_v8_easy'


variant = {'env': {'domain': 'cloth_sim', 'task': 'easy'}}

config_key = 'sac_pixels_cloth_sim'

Calls class MiniBarchRLEval from script : /home/chandandeep/.mujoco/rlpyt/rlpyt/runners/minibatch_rl.py

runner.train (train function in MinibatchRlEval)

----------------------------------------------

Script : /home/chandandeep/.mujoco/rlpyt/rlpyt/runners/minibatch_rl.py

NOTE : When line runners.train is launched in train file

1. Initialize 
a) Initialize sampler 
n_envs_list = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
n_workers = 10
batch_spec = BatchSpec(T=1, B=16)
B = 16
env_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
world_size = 1
rank = 0
eval_n_envs = 10
eval_n_envs_per = 1 
Define and initialize env space (observation space and action space) in DM Control and convert to rlpyt
Define and initialze Agent (Q model and Target Q model) for the given environment spaces
b) Initialize agent
Initialize parameters of the Q model
Define target entropy
Set replay buffer
2. Obtain samples
3. Train
4. Optimize Agent
samples_to_buffer stores samples_replay, action, observation, reward - Add to replay buffer

Calls class CpuSampler from script : /home/chandandeep/Masterarbeit_ws/src/Masterarbeit/Packages/rlpyt/rlpyt/samplers/parallel/cpu/sampler.py

----------------------------------------------

Script : /home/chandandeep/Masterarbeit_ws/src/Masterarbeit/Packages/rlpyt/rlpyt/samplers/parallel/cpu/sampler.py

Calls class ParallelSamplerBase from /home/chandandeep/.mujoco/rlpyt/rlpyt/samplers/parallel/base.py

----------------------------------------------

Script : /home/chandandeep/.mujoco/rlpyt/rlpyt/samplers/parallel/cpu/samplers/parallel/base.py

Calls class BaseSampler from /home/chandandeep/.mujoco/rlpyt/rlpyt/samplers/base.py

----------------------------------------------

Script : /home/chandandeep/.mujoco/rlpyt/rlpyt/samplers/base.py


----------------------------------------------

EnvCls=DMControlEnv
env_kwargs=config["env"]
CollectorCls=CpuResetCollector
eval_env_kwargs=config["env"]
config['sampler']

----------------------------------------------


Quick links : 

launch -> train -> MiniBatchRl
Samplers + Algo + Agent -> MiniBatchRl
main command is runners.train in train file (in MiniBatchRlEval class)


DmControlEnv -> Samplers

Action : [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] to [1,1,1,1,1,1,1,1,1,1,1,1] (size is given by the no. of actions)

Observation : 64*64 image captured from Mujoco environment (Top view)
Mujoco environment uses composite objects (set of particles) to simulate deformable objects

<composite type="grid" count="9 9 1" spacing="0.05" offset="0 0 1">
    <skin material="matcarpet" inflate="0.001" subgrid="3" texcoord="true"/>
    <geom size=".02"/>
    <pin coord="0 0"/>
    <pin coord="8 0"/>
</composite>

count="9 9 1" represents no. of particles in x, y, z direction , i.e. 9 particles in x and y and 1 in z 


<actuator>
	<motor ctrllimited="true" ctrlrange="-1 1" gear="5.0" joint="r1"/>
</actuator>

position actuators specify positions for joints. In this case the range of joints is between -5 to 5 (after multiplying by gear)

Each element in the action array corresponds to an actuator in MuJoCo. These are ordered the same as in the corresponding model XML. The action_spec of the environment is created here:

dm_control/dm_control/mujoco/engine.py

But how and why is the no. of actuators 12 in this case ?

ctrlrange clamps the control input

ctrlrange is in radians for revolute joints. For slide joints the joint range is in meters

cloths are simulated using one joint as floating joint and the rest joints as hinge joints

http://mujoco.org/book/modeling.html#CComposite

action_spec() and observation_spec() describe the actions accepted and the
observations returned by an Environment. For all the tasks in the suite, actions
are given as a single NumPy array. action_spec() returns an ArraySpec, with
attributes describing the shape, data type, and optional minimum and maximum
bounds for the action arrays. Observations consist of an OrderedDict containing one or more NumPy arrays. observation_spec() returns an OrderedDict of
ArraySpecs describing the shape and data type of each corresponding observation

https://arxiv.org/pdf/1801.00690.pdf

Reward : 
Environment : Mujoco domain and task
Agent : Q Model (input_dim = int(np.sum(observation_shape)) , output_size=action_size * 2,)


----------------------------------------------

Minimal implementation using DM Control Env

Agent
----------------------------------------------

Exploring different environment files to understand action_space and reward 

rope_v1.py, rope_v2.py, rope_sac.py : 
action - (2,) - [x,y] of pixel location for a pick point , place point is defined by moving a random distance within a small circle centred at the pick position
reward - rewarding the pixels of the bead to be in the middle. Check the paper

cloth_v8.py, :
(action conditional type ? Or actually when point is randomly selected than 3 DOF are sufficient to define action space)
action - (5,) {if self.eval - why ?? } or (3,) - (5,) = 3  [Fx, Fy, Fz] no. of force actions + 2 [x,y] pixel location
reward - maximizing the overall area with the goal state

cloth_v7.py : 
action - (5,) {if self.eval - why ?? } - (5,) = 3  [Fx, Fy, Fz] no. of force actions + 2 [x,y] pixel location
reward - 

cloth_v3.py : 
action - (12,) -  [Fx, Fy, Fz] no. of force actions + 2 [x, y] pixel location + 3*2 DOF for the 3 arm units + ???

cloth_v0.py : 
action - (12,) - 

cloth_two_hand.py : 
action - (10,) - first two [x,y] pixel locations for pick point + next 5 dont know ? + last three are movements in [x,y,z] from the pick point towards place point
 or (6,) - first xyz is left, second xyz is right

cloth_sim_state.py :
action - (3,) - position of a particle

cloth_point_state.py : 
action - (3,) or (5,) 

cloth_point.py : 
action - (5,) or (3,)

cloth_gripper.py : 
action - (5,)

cloth_corner.py : 
action - (3,) or (7,) - One hot + action

Expand on the pipeline to make it faster for real world use

Rewards : 
Area
Area Convex
Diagonal
Area Concave

