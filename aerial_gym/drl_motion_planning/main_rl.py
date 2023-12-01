
import torch
import torch.nn as nn
from helper import *
from ddns import *

if __name__ == '__main__':

    # Device indicates whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        NUM_EPISODES = 600
    else:
        NUM_EPISODES = 50
    NUM_EPISODES = 1    # TODO remove, change the above

    # Initialize replay memory
    REPLAY_DEPTH = 64 #?
    ReplayMemoryObj = ReplayMemory(REPLAY_DEPTH)

    # Initialize Q (DQN network)
    num_actions = 8 # Or however many we define.
    # Observations would be a depth image x 3 + probably last action? + position
    num_observations = 10 # should this be the shape of the flattened tensor? Placeholder

    QPolicyObj = DQN(num_observations, num_actions)
    QPTargetObj = DQN(num_observations, num_actions)

    # Do we need a make an aerial env here??

    for i_episode in range(NUM_EPISODES):
        # Initialize state, and preprocess it into a tensor that pytorch uses
        state = reset_env()                         # initialize a state
        state_p = process_state_for_dqn(state)      # Flatten state?

        T_COUNT = 100  # I think this shouldn't have a limit but for testing
        for t_timesteps in range(T_COUNT):
            action = select_action(state, i_episode, NUM_EPISODES)
            state_next, reward = observe_env(action)
            state_next_p = process_state_for_dqn(state)

            #save_replay_data(state, action, state_next, reward)
            save_replay_data(state_p, action, state_next_p, reward)

            if reward < 0:      # If we've hit a wall/obstacle or crashed
                break

            state = state_next
            process_state_for_dqn(state)
        # end for

        #if INTER_LIMIT:    # TODO uncomment when fixed
        #    run_experience_replay()



