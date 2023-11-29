import torch
import argparse
import numpy as np

from collections import deque

from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry

from drl_motion_planning.utils import *
from drl_motion_planning.models import DQN

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_DIM = 18


def test_policy(args):
    env, env_cfg = task_registry.make_env(args.env_name, args)

    # Generating random actions
    command_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
    command_actions[:, 0] = 0.0
    env.reset()
    current_pos = np.array([0.0, 0.0, 0.0]) # Check how to get initial pos

    for i in range(0, 50000):
        mp = np.random.randint(1, ACTION_DIM)
        target_pos = get_action(current_pos, mp)
        command_actions[:, 1] = target_pos[0]
        command_actions[:, 2] = target_pos[1]
        command_actions[:, 3] = target_pos[2]
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)

        if i % 500 == 0:
            print("Resetting environment")
            print("Shape of observation tensor", obs.shape)
            print("Shape of reward tensor", rewards.shape)
            print("Shape of reset tensor", resets.shape)
            if priviliged_obs is None:
                print("Privileged observation is None")
            else:
                print("Shape of privileged observation tensor", priviliged_obs.shape)
            print("------------------")
            env.reset()


if __name__ == "__main__":
    args = get_args()
    test_policy(args)