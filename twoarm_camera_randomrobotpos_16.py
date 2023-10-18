import argparse
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import time
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3 import PPO
import imageio
from robosuite.controllers import load_controller_config

current_name = os.path.basename(__file__)
filename = f'tmp/gym/{current_name[:-3]}'
cwd = os.getcwd()
new_folder = os.path.join(cwd, filename)
os.makedirs(new_folder, exist_ok=True)

parser = argparse.ArgumentParser(description='Training or testing mode')
parser.add_argument('mode', type=str, help='mode', nargs='?', default="training")
args = parser.parse_args()

mode = args.mode
controller_config = load_controller_config(default_controller='OSC_POSE')

if mode == 'test':
    num_envs = 16
    env = GymWrapper(suite.make(
        env_name="TwoArmPegInHoleCustom", # try with other tasks like "Stack" and "Door"
        robots=["UR5e","UR5e_custom"],  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=None, 
        use_object_obs=False,                   # don't provide object observations to agent
        use_camera_obs=False,
        camera_names="robot0_eye_in_hand",      # use "agentview" camera for observations
        camera_heights=512,                      # image height
        camera_widths=512,                       # image width
        reward_shaping=True,                    # use a dense reward signal for learning
        reward_scale=1.0,
        horizon = 500,
        control_freq=20,                        # control should happen fast enough so that simulation looks smooth
        ignore_done=True,
        hard_reset=False,
    ))
    model = PPO.load(f'{filename}/best_model.zip', env=env)
    writer = imageio.get_writer(f'{filename}/video.mp4', fps=20)
    for i in range(num_envs):
        obs = env.reset()[0]
        for i in range(100):
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
    def make_env(i):
        env = GymWrapper(suite.make(
            env_name="TwoArmPegInHoleCustom", # try with other tasks like "Stack" and "Door"
            robots=["UR5e","UR5e_custom"],  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera=None, 
            use_object_obs=False,                   # don't provide object observations to agent
            use_camera_obs=False,
            camera_names="robot0_eye_in_hand",      # use "agentview" camera for observations
            camera_heights=84,                      # image height
            camera_widths=84,                       # image width
            reward_shaping=True,                    # use a dense reward signal for learning
            reward_scale=1.0,
            horizon = 500,
            control_freq=20,                        # control should happen fast enough so that simulation looks smooth
            ignore_done=False,
            hard_reset=False,
        ))
        return env

    if __name__ == '__main__':
        num_envs = 16
        envs = [lambda i=i: make_env(i) for i in range(num_envs)]
        env = SubprocVecEnv(envs)
        env = VecMonitor(env, filename=filename)

        policy_kwargs = dict(
            net_arch=[256, 256]
        )

        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=filename)

        # Do something else if the folder doesn't exist
        print(f"The folder '{filename}' does not exist.")
        model = PPO("MultiInputPolicy", env, verbose=1, batch_size=256, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=5e6, progress_bar=True, log_interval=10, callback=callback)
        # del model
        # # model.save(f'{filename}/best_model')
        # for i in range(10):
        #     model = PPO.load(f'{filename}/best_model.zip', env=env)
        #     model.learn(total_timesteps=5e5, progress_bar=True, log_interval=20, callback=callback)
