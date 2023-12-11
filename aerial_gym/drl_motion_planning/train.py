import torch
import torch.nn as nn
import argparse

from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry

from drl_motion_planning.utils import *
from drl_motion_planning.models import DQN

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBS_DIM = 10
ACTION_DIM = 18


def train_policy(args):

    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    policy_net = DQN(OBS_DIM, ACTION_DIM).to(DEVICE)
    target_net = DQN(OBS_DIM, ACTION_DIM).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    memory = ReplayMemory(10000)

    def select_action(state):
        # epsilon-greedy action selection #TODO
        pass

    def optimize_model():
        if len(memory) < args.batch_size:
            return
        transitions = memory.sample(args.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(args.batch_size, device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * args.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    env.reset()
    # observe initial states #TODO
    episode_durations = []

    for i in range(args.num_iterations):

        for t in range(args.num_timesteps):
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*args.tau + target_net_state_dict[key]*(1-args.tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break


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