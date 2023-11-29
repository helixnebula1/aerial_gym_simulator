import torch
import argparse

from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry

from drl_motion_planning.utils import *
from drl_motion_planning.models import DQN

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_policy(args):

    env, env_cfg = task_registry.make_env(name=args.task, args=args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TODO: Define Additional Parameters
    additional_parameters = []

    # parser.add_argument('-i', '--num_iterations', help='Number of Iterations/Experiments', type=int, default=30)
    # parser.add_argument('-n', '--num_episodes', help='Number of Episodes', type=int, default=50)
    # parser.add_argument('-g', '--gamma', help='Discount Factor gamma', type=float, default=0.95)
    # parser.add_argument('-a', '--alpha', help='Step Size alpha', type=float, default=0.1)
    # parser.add_argument('-e', '--epsilon', help='Epsilon Greedy epsilon', type=float, default=0.1)
    # parser.add_argument('-p', '--num_planning_steps', help='Number of Planning Steps', type=int, default=50)

    args = get_args(additional_parameters)
    train_policy(args)