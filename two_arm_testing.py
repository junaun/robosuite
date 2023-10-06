import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from PIL import Image
from IPython.display import display
from matplotlib import pyplot as plt
import imageio


env = GymWrapper(suite.make(
    env_name="TwoArmPegInHole", # try with other tasks like "Stack" and "Door"
    robots=["UR5e","UR5e_custom"],  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=True,
    render_camera=None, 
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,
    camera_names="frontview",      # use "agentview" camera for observations
    camera_heights=512,                      # image height
    camera_widths=512,                       # image width
    reward_shaping=True,                    # use a dense reward signal for learning
    horizon = 500,
    control_freq=20,                        # control should happen fast enough so that simulation looks smooth
))

# writer = imageio.get_writer(f'video.mp4', fps=20)

for i in range(200):
    action = np.random.uniform(-1, 1,12)
    obs, reward, done,done, info = env.step(action)  # take action in the environment
    # writer.append_data(obs['frontview_image'][::-1])
    env.render()
# writer.close()
env.close()