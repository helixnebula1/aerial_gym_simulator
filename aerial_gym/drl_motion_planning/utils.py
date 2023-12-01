import torch
import numpy as np
from collections import namedtuple, deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_DIM = 18

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_action(current_pos, motion_primitive):
    actions = torch.tensor([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [0, -1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                            [1, 1, 1],
                            [1, 1, 0],
                            [1, 1, -1],
                            [1, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, -1, 1]], dtype=torch.float32, device=DEVICE)

    motion_primitive = torch.tensor(motion_primitive, device=DEVICE)
    #return current_pos + torch.matmul(actions, motion_primitive)
    return torch.tensor(current_pos, device=DEVICE) + torch.matmul(motion_primitive, actions)

def get_depth_image(env):
    # python3 test3.py --num_envs 1 --task="quad_with_obstacles" to get camera_handles
    from isaacgym import gymapi
    IMAGE_HEIGHT = 270
    IMAGE_WIDTH = 480

    num_envs = env.num_envs
    depth_images = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, num_envs))

    # For each environment, get the depth image
    for ii in range(num_envs):

        depth_image = env.gym.get_camera_image(env.sim, env.envs[ii], env.camera_handles[ii], gymapi.IMAGE_DEPTH)

        # -inf implies no depth value, set it to zero. output will be black.
        depth_image[depth_image == -np.inf] = 0

        # clamp depth image to 10 meters to make output image human friendly
        depth_image[depth_image < -10] = -10

        # flip the direction so near-objects are light and far objects are dark
        # Can't say this looks very normalized.  but taken from graphics.py
        normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))

        depth_images[:,:,ii] = normalized_depth

        # Convert to a pillow image and write it to disk
        #normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
        #normalized_depth_image.save("graphics_images/depth_env%d_cam%d_frame%d.jpg" % (ii, jj, frame_count))

    depth_images = torch.tensor(depth_images, device=DEVICE)
    #print(depth_images.shape)
    return depth_images


def take_random_action(env, actionObj):

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

def take_3_random_actions(env, actionObj):
    # Generate 3 successive random action
    # Maybe for improvement, we add another, or velocity or something

    current_pos_t2, privileged_obs, rewards, resets, extras = take_random_action(env, actionObj)
    depth_image_t2 = get_depth_image(env)

    current_pos_t1, privileged_obs, rewards, resets, extras = take_random_action(env, actionObj)
    depth_image_t1 = get_depth_image(env)

    current_pos_t0 , privileged_obs, rewards, resets, extras = take_random_action(env, actionObj)
    depth_image_t0 = get_depth_image(env)

    # Concatenate the positions etc and depth images to return # TODO

    return current_pos_t0, rewards, resets
