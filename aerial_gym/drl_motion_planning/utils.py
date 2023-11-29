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
    action = np.array([0, 0, 0])

    if motion_primitive == 1: action = np.array([0, 0, 1])
    elif motion_primitive == 2: action = np.array([0, 1, 1])
    elif motion_primitive == 3: action = np.array([0, 1, 0])
    elif motion_primitive == 4: action = np.array([0, 1, -1])
    elif motion_primitive == 5: action = np.array([0, 0, -1])
    elif motion_primitive == 6: action = np.array([0, -1, -1])
    elif motion_primitive == 7: action = np.array([0, -1, 0])
    elif motion_primitive == 8: action = np.array([0, -1, 1])
    elif motion_primitive == 9: action = np.array([1, 0, 0])
    elif motion_primitive == 10: action = np.array([1, 0, 1])
    elif motion_primitive == 11: action = np.array([1, 1, 1])
    elif motion_primitive == 12: action = np.array([1, 1, 0])
    elif motion_primitive == 13: action = np.array([1, 1, -1])
    elif motion_primitive == 14: action = np.array([1, 0, -1])
    elif motion_primitive == 15: action = np.array([1, -1, -1])
    elif motion_primitive == 16: action = np.array([1, -1, 0])
    elif motion_primitive == 17: action = np.array([1, -1, 1])

    return current_pos + action