import argparse
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

ACTION_DIM = 18

def all_members(obj, obj2):
    with open('members.txt', 'a') as fp:
        dirobj = str((dir(obj)))
        dirobj2 = "\n" + str((dir(obj2)))
        fp.write(dirobj)
        #print(dir(env._create_envs))
        fp.write(dirobj2)
    return


class ActionObj:
    def __init__(self, env_cfg):
        # Initialize these
        self.commandActions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
        self.motionPrimitive = torch.zeros((env_cfg.env.num_envs, ACTION_DIM))
        self.currentPos = torch.zeros((env_cfg.env.num_envs, 3))

def get_depth_image(env):  # TODO no idea where this goes
    # python3 test3.py --num_envs 1 --task="quad_with_obstacles" to get camera_handles
    from isaacgym import gymapi
    IMAGE_HEIGHT = 270
    IMAGE_WIDTH = 480

    num_envs = env.num_envs
    depth_images = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, num_envs))

    #depth_image = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handles[0], gymapi.IMAGE_DEPTH)
    #print("type of depth image ", type(depth_image))
    #print("shape depth_image ", depth_image.shape)

    for ii in range(num_envs):

        depth_image = env.gym.get_camera_image(env.sim, env.envs[ii], env.camera_handles[ii], gymapi.IMAGE_DEPTH)

        # -inf implies no depth value, set it to zero. output will be black.
        depth_image[depth_image == -np.inf] = 0

        # clamp depth image to 10 meters to make output image human friendly
        depth_image[depth_image < -10] = -10

        # flip the direction so near-objects are light and far objects are dark
        normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))

        # Convert to a pillow image and write it to disk
        #normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
        #normalized_depth_image.save("graphics_images/depth_env%d_cam%d_frame%d.jpg" % (ii, jj, frame_count))
    print(depth_images)
    return


def take_random_action(env, actionObj):  # TODO: move this fn to utils.py

    # Generate a random action
    mp = np.random.randint(1, ACTION_DIM)
    actionObj.motionPrimitive[:, mp] = 1.0
    target_pos = get_action(actionObj.currentPos, actionObj.motionPrimitive)

    # Obtain command action given target position, using lee position controller
    actionObj.commandActions[:, 0] = target_pos[:,0] # x
    actionObj.commandActions[:, 1] = target_pos[:,1] # y
    actionObj.commandActions[:, 2] = target_pos[:,2] # z
    actionObj.commandActions[:, 3] = 0.0 # yaw

    # Step the environment and observe new state and reward.
    # If the same command_action is used 3x in a row, the observations are the same for each,
    # so it appears there's one step for each commanded action and you don't get observations in between
    obs, privileged_obs, rewards, resets, extras = env.step(actionObj.commandActions)
    #current_pos = obs[:, :3]

    return obs[:, :3], privileged_obs, rewards, resets, extras #current_pos_t0, current_pos_t1, current_pos_t2

def take_3_random_actions(env, actionObj):  # TODO: move this fn to utils.py
    # Generate 3 successive random action
    # Maybe for improvement, we add another, or velocity or something


    current_pos_t2, privileged_obs, rewards, resets, extras = take_random_action(env, actionObj)

    current_pos_t1, privileged_obs, rewards, resets, extras = take_random_action(env, actionObj)

    current_pos_t0 , privileged_obs, rewards, resets, extras = take_random_action(env, actionObj)

    # Concatenate the positions

    return #current_pos_t0, current_pos_t1, current_pos_t2



def test_policy(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.reset()

    actionObj = ActionObj(env_cfg)

    num_envs = env_cfg.env.num_envs
    get_depth_image(env)


    #for i in range(0, 50000):

        #obs, privileged_obs, rewards, resets, extras = take_random_action(env, actionObj)

        #if i % 500 == 0:
        #    print("Resetting command")
        #    current_pos = obs #obs[:, :3]

        #    print(i, " ------------------")
        #    env.reset()

# AJS : 2 questions - If there are >1 num_envs, is the start state random?  It doesn't look like it.  and it looks like all of the envs
# are getting the same random target position, but some are flying all over the place.  Or appear to.  But i'm not seeing where that
# randomness is comign from.  (Will we want different motions for all of them in the end?)
# Also, where is the env.step, env.reset defined.  I looked in utils directory and don't see it.

if __name__ == "__main__":
    args=get_args()
    test_policy(args)
