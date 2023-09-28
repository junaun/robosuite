"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env

from robosuite.wrappers import Wrapper


class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        # self.obs_dim = flat_ob.shape
        # high = np.inf * np.ones(self.obs_dim)
        # low = -high
        high = 255
        low = 0
        # robot_state = {
        #     'robot0_joint_pos_cos': spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        #     'robot0_joint_pos_sin':spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        #     'robot0_joint_vel':spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        #     'robot0_eef_pos':spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        #     'robot0_eef_quat':spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        #     'robot0_gripper_qpos':spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        #     'robot0_gripper_qvel':spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        #     'robot0_eye_in_hand_image':spaces.Box(low, high, shape=obs['robot0_eye_in_hand_image'].shape, dtype=np.uint8()),
        #     'robot0_proprio-state':spaces.Box(low=-1, high=1, shape=(37,), dtype=np.float32),
        #     'cube_pos':spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        #     'cube_quat':spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        #     'gripper_to_cube_pos':spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        #     'object-state':spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        # }
        robot_state = {}
        for key, value in obs.items():
            if key in self.keys:
                if 'image' in key:
                    robot_state[key] = gym.spaces.Box(low=0, high=255, shape=value.shape, dtype=np.uint8)
                else:
                    robot_state[key] = gym.spaces.Box(low=-1, high=1, shape=value.shape, dtype=np.float32)
        self.observation_space = spaces.Dict(robot_state)
        # self.observation_space = spaces.Box(low, high, shape=self.obs_dim, dtype=np.uint8())
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        robot_state = {}
        for key, value in obs_dict.items():
            if key in self.keys:
                robot_state[key] = value
        return robot_state

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict), {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
