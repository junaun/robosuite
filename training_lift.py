import argparse
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
import os
from stable_baselines3 import PPO
import imageio
from robosuite.controllers import load_controller_config

filename = 'tmp/gym/multiinput_simple'
cwd = os.getcwd()
new_folder = os.path.join(cwd, filename)
os.makedirs(new_folder, exist_ok=True)

parser = argparse.ArgumentParser(description='Training or testing mode')
parser.add_argument('mode', type=str, help='mode', nargs='?', default="training")
args = parser.parse_args()

mode = args.mode
controller_config = load_controller_config(default_controller='OSC_POSE')

if mode == 'test':
    env = GymWrapper(suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs = controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_object_obs=False,                   # don't provide object observations to agent
        use_camera_obs=True,
        camera_names="robot0_eye_in_hand",      # use "agentview" camera for observations
        camera_heights=84,                      # image height
        camera_widths=84,                       # image width
        reward_shaping=True,                    # use a dense reward signal for learning
        horizon = 500,
        control_freq=20,                        # control should happen fast enough so that simulation looks smooth
    ))
    model = PPO.load(f'{filename}/best_model.zip', env=env)
    obs = env.reset()[0]
    writer = imageio.get_writer(f'{filename}/video.mp4', fps=20)
    for i in range(500):
        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, done, info = env.step(action)
        frontview = env.sim.render(height=1024, width=1024, camera_name="frontview")[::-1]
        writer.append_data(frontview)
        env.render()
        if done:
            break
    writer.close()
    env.close()
else:
    class SaveOnBestTrainingRewardCallback(BaseCallback):
        def __init__(self, check_freq: int, log_dir: str, verbose=1):
            super().__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, "best_model")
            self.best_mean_reward = -np.inf

        def _init_callback(self) -> None:
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:

                # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(
                            f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                        )

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}.zip")
                        self.model.save(self.save_path)

            return True

    env = GymWrapper(suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=False,
        controller_configs = controller_config,
        use_object_obs=False,                   # don't provide object observations to agent
        use_camera_obs=False,
        reward_shaping=True,                    # use a dense reward signal for learning
        reward_scale=1.0,
        horizon = 500,
        control_freq=20,                        # control should happen fast enough so that simulation looks smooth
        ignore_done=True,
        hard_reset=False,
    ))
    env = Monitor(env, filename)
    # ), ['robot0_eye_in_hand_image'])
    policy_kwargs = dict(
        net_arch=[256, 256]
    )

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=filename)

    if os.path.exists(f'{filename}/best_model.zip'):
        # Do something if the folder exists
        print(f"The folder '{filename}' exists.")
        model = PPO.load(f'{filename}/best_model.zip', env=env)
        model.learn(total_timesteps=5e6, progress_bar=True, log_interval=20, callback=callback)
    else:
        # Do something else if the folder doesn't exist
        print(f"The folder '{filename}' does not exist.")
        model = PPO("MultiInputPolicy", env, verbose=1, batch_size=256, policy_kwargs=policy_kwargs)
        print('start learning')
        model.learn(total_timesteps=5e4, progress_bar=True, log_interval=10, callback=callback)
