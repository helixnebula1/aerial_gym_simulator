import argparse
import queue
from misc_utils import *
import numpy as np
from collections import deque
import math

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch

from utils import *
from train_utils import *
#from models import *


# Some code resue from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# --task="quad_with_obstacles"
# python3 test3.py --num_envs 15 --task="quad_with_obstacles"
from models import DQN

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBS_DIM = 10
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
    REPLAY_MEMORY_DEPTH = 100# 50
    REPLAY_BATCH_SIZE = 32
    replayMemoryObj = ReplayMemoryClass(REPLAY_MEMORY_DEPTH)

    # Target and policy networks
    policy_net = DQN(OBS_DIM, ACTION_DIM).to(DEVICE)
    target_net = DQN(OBS_DIM, ACTION_DIM).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    # Constants
    BATCH_SIZE = 128
    GAMMA = 0.99

    TAU = 0.005
    LR = 1e-4

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)
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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                               batch.nextPos)), device=DEVICE, dtype=torch.bool)

        non_final_next_pos = torch.cat([s for s in batch.nextPos
                                           if s is not None])

        print("types ", type(batch.pos))
        pos_batch = torch.cat(batch.pos)
        depth_batch = torch.cat(batch.depth)
        action_batch = batch.action #torch.cat(batch.action)
        reward_batch = batch.reward #torch.cat(batch.reward)
        print(pos_batch.shape, depth_batch.shape, type(action_batch), type(reward_batch))


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # next_state_values = torch.zeros(args.batch_size, device=DEVICE)
        # with torch.no_grad():
        #    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        # expected_state_action_values = (next_state_values * args.gamma) + reward_batch
        return

    for i in range(0, 50000):

        # Take random action.  # We will have to replace this with something like: action = DQN(posT, depthT)
        take_random_action(env, actionObj)

        # Observe reward and environment
        pos_next, depth_image_next, rewards, resets = observe_env(env, actionObj)
        #rewards = torch.tensor([rewards], device=DEVICE)
        #print(rewards.shape)

        # Add observations to the queue since we need to put multiple observations through the DQN
        stateQObj.addObservation(pos_next, depth_image_next)

        # Get the next state (these are tensors obtained from the queue of states - pos, depth
        nextPosT, nextDepthT = stateQObj.state()        # This returns the latest 8 position observations, and latest 3 depth observations as tensors

        # Push the transition into replay memory
        replayMemoryObj.push(posT, depthT, actionObj, nextPosT, nextDepthT, rewards)
        #x = replayMemoryObj.sample(REPLAY_BATCH_SIZE)
        #print(type(x))
        #optimize_model()

        if rewards < 0:  # If we've hit a wall/obstacle or crashed
            break

        # Make next state the current state
        posT, depthT = nextPosT, nextDepthT
        optimize_model()


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