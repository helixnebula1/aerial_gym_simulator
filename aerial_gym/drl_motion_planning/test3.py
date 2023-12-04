import argparse
import queue

import numpy as np

from collections import deque

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch

from utils import *
from models import DQN

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ActionObj:
    def __init__(self, env_cfg):
        # Initialize these
        self.commandActions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
        self.motionPrimitive = torch.zeros((env_cfg.env.num_envs, ACTION_DIM))
        self.currentPos = torch.zeros((env_cfg.env.num_envs, 3))



def test_policy(args):
    RANGE = 50000
    T_RANGE = 1

    # Make environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.reset()

    #
    actionObj = ActionObj(env_cfg)
    num_envs = env_cfg.env.num_envs

    # Lists to hold 8 last positions, and 3 last depth_images
    POS_Q_SIZE = 8
    DEPTH_Q_SIZE = 3
    pos_queue = queue.Queue(maxsize=POS_Q_SIZE)
    depth_image_queue = queue.Queue(maxsize=DEPTH_Q_SIZE)

    # Observe beginning state
    pos, depth_image, rewards, resets = observe_env(env, actionObj)
    print("main ", type(pos), pos.shape)

    # Fill the position and depth Qs for DQN to start with by repeating the first item
    while pos_queue.full() == False:
        pos_queue.put(pos)
    while depth_image_queue.full() == False:
        depth_image_queue.put(depth_image)

    # Obtain the DQN tensors (3 depth images, 8 positions)
    process_state_for_dqn(pos, depth_image, pos_queue, depth_image_queue)

    #print(pos_queue.queue[0])


    for i in range(0, 50000):

        take_random_action(env, actionObj)
        pos, depth_image, rewards, resets = observe_env(env, actionObj)
        process_state_for_dqn(pos, depth_image, pos_queue, depth_image_queue)

        if i % 500 == 0:
            print("Resetting command")
            current_pos = pos #obs[:, :3]

            print(i, " ------------------")
            env.reset()

# AJS : 2 questions - If there are >1 num_envs, is the start state random?  It doesn't look like it.  and it looks like all of the envs
# are getting the same random target position, but some are flying all over the place.  Or appear to.  But i'm not seeing where that
# randomness is comign from.  (Will we want different motions for all of them in the end?)
# Also, where is the env.step, env.reset defined.  I looked in utils directory and don't see it.

if __name__ == "__main__":
    args=get_args()
    test_policy(args)
