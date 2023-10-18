import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

dir = 'demo_test/'
controller_config = load_controller_config(default_controller='JOINT_VELOCITY')
env = DataCollectionWrapper(suite.make(
    env_name="Lift_custom_fixboxpose", # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    #controller_configs = controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=False,
    reward_shaping=True,                    # use a dense reward signal for learning
    horizon = 500,
    control_freq=20,                        # control should happen fast enough so that simulation looks smooth
), dir, flush_freq=500)
env = GymWrapper(env)


model = PPO.load(f'best_model.zip', env=env)
for i in range(16): 
    obs = env.reset()[0]
    for i in range(500):
        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, done, info = env.step(action)
        env.render()
        if done:
            break
    env.close()

    config = {
        "env_name": 'Lift',
        "robots": 'UR5e',
        "controller_configs": controller_config,
    }
    env_info = json.dumps(config)
    os.makedirs('hdf5_test/', exist_ok=True)
    gather_demonstrations_as_hdf5('demo_test/', 'hdf5_test/', env_info)
