import torch
import numpy as np
from collections import namedtuple, deque

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
                            [1, -1, 1]], dtype=torch.float32)

    return current_pos + torch.matmul(actions, motion_primitive)