import argparse
import queue
from misc_utils import *
import numpy as np
from collections import deque
import math
import math

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch

from utils import *
from train_utils import *
from models import DQN1a
#from models import *


# Some code resue from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# --task="quad_with_obstacles"
# python3 test3.py --num_envs 15 --task="quad_with_obstacles"


torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBS_DIM = 270*480
ACTION_DIM = 18

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

    def state(self):   # Returns relative positions, and 3 depth images
        # Stack the most recent 3 depth images
        # Double check the order these go in the queue and are used in teh dqn # TODO
        rel_depth_images = torch.stack(
            [self.depth.queue[0], self.depth.queue[1], self.depth.queue[2]], dim=2)
        rel_depth_images = torch.squeeze(rel_depth_images)  # 270 x 480 x 3

        # Most recent position
        rel_pos = torch.tensor(self.pos.queue[0], dtype=torch.float)    # 1x3

        return rel_pos, rel_depth_images

def train_policy(args):
    RANGE = 50000
    T_RANGE = 1

    # Make environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.reset()

    # Action object to hold current position, motion primitives and command actions
    actionObj = ActionObj(env_cfg)
    num_envs = env_cfg.env.num_envs

    # Lists to hold 8 last positions, and 3 last depth_images
    POS_Q_SIZE = 1
    DEPTH_Q_SIZE = 3

    # Instantiate state objects for current and next states (holds both position and depth
    # This hold 4 depth images, and 9 positions.  Because each state is 3 depth images and 8 positions,
    # and each nextState is also 3 depth images and 8 positions, with 7 overlaps of positions and 2
    # overlaps of depth images with state.
    stateQObj = StateQClass(POS_Q_SIZE, DEPTH_Q_SIZE)

    # Observe beginning state (reset state)
    # pos: Tensor [1,3]
    # depth_image: Tensor [270, 480, 1]
    # rewards: Tensor [1]
    # resets: Tensor [1]
    pos, depth_image, rewards, resets = observe_env(env, actionObj)

    # Use the beginning state to reset state queue - repeat these in the queue
    stateQObj.resetState(pos, depth_image)

    # Set current state (1 position, 3 depth images)
    pos, depth = stateQObj.state()        # Torch tensors [1,3] and [270,480,3]

    # Set up memory replay queue
    # Instantiate Experience object to hold an experience for storage and replay
    # Initialize replay memory
    REPLAY_MEMORY_DEPTH = 100# 50
    REPLAY_BATCH_SIZE = 32
    replayMemoryObj = ReplayMemoryClass(REPLAY_MEMORY_DEPTH)

    # Target and policy networks
    TAU = 0.005
    LR = 1e-4
    policy_net = DQN1a(OBS_DIM, ACTION_DIM).to(DEVICE)
    #target_net = DQN(OBS_DIM, ACTION_DIM).to(DEVICE)
    #target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

    # Constants
    BATCH_SIZE = 128
    GAMMA = 0.99

    # Steps
    steps_done = 0



    def select_action(posT, depthT, steps_done):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

        if sample > 0: #eps_threshold:  # TODO convert this back to eps_threshold
            with torch.no_grad():
                # Return index of largest Q value, converted to int (rather than tensor)
                return int(policy_net(depthT, posT).argmax())  # Tensors [270,480,3], [1,3]
        else:
            return np.random.randint(ACTION_DIM)    # Return a number from 0-ACTION_DIM-1

    def optimize_model():
        if len(replayMemoryObj) < REPLAY_BATCH_SIZE:
            return
        transitions = replayMemoryObj.sample(REPLAY_BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask is a torch tensor of size 32
        non_final_mask = tuple(map(lambda s: s is not None,
                                               batch.nextPos))

        non_final_next_pos = [s for s in batch.nextPos
                                           if s is not None]

        #print("***", batch.pos.shape, type(batch.pos), batch.depth.shape, type(batch.depth))
        #print("types ", type(batch.pos))
        #pos_batch = torch.cat(batch.pos)
        #print("888", pos_batch.shape, type(pos_batch))
        #depth_batch = torch.cat(batch.depth)
        print("***lensbatch ",len(batch.pos), len(batch.depth[0]), len(batch.action), len(batch.reward))

        pos_batch = torch.cat(batch.pos)            # [32,3]
        depth_batch = torch.stack(batch.depth)      # [32, 270, 480, 3]
        action_batch = torch.tensor(batch.action)   # [32]
        reward_batch = torch.cat(batch.reward)      # [32]
        print("\n***types ",type(pos_batch), type(depth_batch), type(action_batch), type(reward_batch))
        print("***lens ",len(pos_batch), len(depth_batch), len(action_batch), len(reward_batch))
        print("***shape ",pos_batch.shape, depth_batch.shape, action_batch.shape, reward_batch.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = torch.zeros(REPLAY_BATCH_SIZE, device=DEVICE)
        for ii in range(REPLAY_BATCH_SIZE):
            p = torch.unsqueeze(pos_batch[ii,:], dim=0)
            state_action_values[ii] = policy_net(depth_batch[ii,:,:,:],p).argmax() #gather(1, action_batch) # TODO not sure this line is correct

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(args.batch_size, device=DEVICE)
        with torch.no_grad():
            for ?
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        return

    for i in range(0, 50000):

        # Select random action.
        action = select_action(pos, depth, steps_done)
        #print("type action", type(action), action.shape)

        # Take the action
        take_action(env, actionObj, action)

        # Observe reward and environment
        pos_next, depth_image_next, rewards, resets = observe_env(env, actionObj)
        #rewards = torch.tensor([rewards], device=DEVICE)


        # Add observations to the queue since we need to put multiple observations through the DQN
        stateQObj.addObservation(pos_next, depth_image_next)

        # Get the next state (these are tensors obtained from the queue of states - pos, depth
        nextPos, nextDepth = stateQObj.state()        # This returns the latest 8 position observations, and latest 3 depth observations as tensors
        #print("\n***types here", type(nextPos), type(nextDepth), nextPos.shape, nextDepth.shape)

        # Push the transition into replay memory
        replayMemoryObj.push(pos, depth, action, nextPos, nextDepth, rewards)
        #x = replayMemoryObj.sample(REPLAY_BATCH_SIZE)
        #print(type(x))
        #optimize_model()

        if rewards < 0:  # If we've hit a wall/obstacle or crashed
            break

        # Make next state the current state
        pos, depth = nextPos, nextDepth
        optimize_model()

        # Increment number of steps
        steps_done += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TODO: Define Additional Parameters
    additional_parameters = []

    parser.add_argument('--num_iterations', help='Number of Iterations/Experiments', type=int, default=30)
    parser.add_argument('--alpha', help='Step Size alpha', type=float, default=0.1)
    # parser.add_argument('-n', '--num_episodes', help='Number of Episodes', type=int, default=50)
    # parser.add_argument('-g', '--gamma', help='Discount Factor gamma', type=float, default=0.95)
    # parser.add_argument('-e', '--epsilon', help='Epsilon Greedy epsilon', type=float, default=0.1)
    # parser.add_argument('-p', '--num_planning_steps', help='Number of Planning Steps', type=int, default=50)

    args = get_args(additional_parameters)
    train_policy(args)