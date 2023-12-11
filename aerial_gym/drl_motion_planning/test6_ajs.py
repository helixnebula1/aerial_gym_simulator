import argparse
import queue
from misc_utils import *

import numpy as np

from collections import deque

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch

from utils import *

# --task="quad_with_obstacles"
# python3 test3.py --num_envs 15 --task="quad_with_obstacles"
from models import DQN

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ActionObj:
    def __init__(self, env_cfg):
        # Initialize these
        self.commandActions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
        self.motionPrimitive = torch.zeros((env_cfg.env.num_envs, ACTION_DIM))
        self.currentPos = torch.zeros((env_cfg.env.num_envs, 3))


class StateQClass:
    # I have a feeling memory handling might be easier this way but not sure
    def __init__(self, POS_Q_SIZE, DEPTH_Q_SIZE):
        self.pos = queue.Queue(maxsize=POS_Q_SIZE)
        self.depth = queue.Queue(maxsize=DEPTH_Q_SIZE)

    def resetState(self, pos, depth):
        # Fill the position and depth Qs for DQN to start with by repeating the first item
        while self.pos.full() == False:
            self.pos.put(pos)
        while self.depth.full() == False:
            self.depth.put(depth)

    def addObservation(self, pos, depth):
        # Add new observation to the queue
        # Pop the queues of the oldest element if full
        if self.pos.full():
            self.pos.get()
        if self.depth.full():
            self.depth.get()

        # Push the latest to the queue
        self.pos.put(pos)
        self.depth.put(depth)

    def state(self):
        # Stack the most recent 3 depth images
        # Double check the order these go in the queue and are used in teh dqn # TODO
        rel_depth_images = torch.stack(
            [self.depth.queue[0], self.depth.queue[1], self.depth.queue[2]], dim=1)

        # Stack the most recent 8 positions
        rel_pos = torch.stack([self.pos.queue[0], self.pos.queue[1],
                               self.pos.queue[2], self.pos.queue[3],
                               self.pos.queue[4], self.pos.queue[5],
                               self.pos.queue[6], self.pos.queue[7]], dim=1)
        return rel_pos, rel_depth_images


#class StateClass:
#    def __init__(self,  POS_Q_SIZE, DEPTH_Q_SIZE):
#        self.pos = # Tensor size
#        self.depth = # Tensor size

#    def set(self, pos, depth):
#        self.pos = pos
#        self.depth = depth


class ExperienceClass:
    def __init__(self, pos, depth, action, nextPos, nextDepth, reward):
        # I think I should init these things with zeros, but not sure how big they are
        self.pos= pos  # This is a tensor
        self.depth= depth  # This is a tensor
        self.action = action    # This needs to be the output of the dqn type
        self.nextPos= nextPos  # This is a tensor
        self.nextDepth= nextDepth  # This is a tensor
        self.reward = reward


Transition = namedtuple('Transition',
                        ('pos', 'depth', 'action', 'nextPos', 'nextDepth', 'reward'))
class ReplayMemoryClass(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def test_policy(args):
    RANGE = 50000
    T_RANGE = 1

    # Make environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.reset()

    # Action object to hold current position, motion primitives and command actions
    actionObj = ActionObj(env_cfg)
    num_envs = env_cfg.env.num_envs

    # Lists to hold 8 last positions, and 3 last depth_images
    POS_Q_SIZE = 8
    DEPTH_Q_SIZE = 3


    # Instantiate state objects for current and next states (holds both position and depth
    # This hold 4 depth images, and 9 positions.  Because each state is 3 depth images and 8 positions,
    # and each nextState is also 3 depth images and 8 positions, with 7 overlaps of positions and 2
    # overlaps of depth images with state.
    stateQObj = StateQClass(POS_Q_SIZE, DEPTH_Q_SIZE)

    # Observe beginning state (reset state)
    pos, depth_image, rewards, resets = observe_env(env, actionObj)
    stateQObj.resetState(pos, depth_image)  # Use the beginning state to reset state queue

    # Set current state
    posT, depthT = stateQObj.state()        # Torch tensor

    # Set up memory replay queue
    # Instantiate Experience object to hold an experience for storage and replay
    # Initialize replay memory
    REPLAY_MEMORY_DEPTH = 10
    replayMemoryObj = ReplayMemoryClass(REPLAY_MEMORY_DEPTH)


    for i in range(0, 50000):

        # Take random action.  # We will have to replace this with something like: action = DQN(posT, depthT)
        take_random_action(env, actionObj)

        # Observe reward and environment
        pos_next, depth_image_next, rewards, resets = observe_env(env, actionObj)

        # Add observations to the queue since we need to put multiple observations through the DQN
        stateQObj.addObservation(pos_next, depth_image_next)

        # Get the next state (these are tensors obtained from the queue of states - pos, depth
        nextPosT, nextDepthT = stateQObj.state()        # This returns the latest 8 position observations, and latest 3 depth observations as tensors

        # I wanted to have a class with 2 tensors, one for position and one for depth - but is it necessary?
        #nextStateObj.set(nextPosT, nextDepthT)       # This was to have objects to push on experience replay

        # Push the transition into replay memory
        replayMemoryObj.push(posT, depthT, actionObj, nextPosT, nextDepthT, rewards)


        if rewards < 0:  # If we've hit a wall/obstacle or crashed
            break

        # Update next state
        posT, depthT = nextPosT, nextDepthT

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
