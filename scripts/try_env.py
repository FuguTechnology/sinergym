import logging

import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)

# Logger
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='MAIN',
    level=logging.INFO
)

# Create environment and apply wrappers for normalization and logging
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

# Execute 1 episode
episodes = 1
print("start test")
for i in range(episodes):

    # Reset the environment to start a new episode
    obs, info = env.reset()

    rewards = []
    truncated = terminated = False
    current_month = 0

    while not (terminated or truncated):

        # Random action selection
        a = env.action_space.sample()
        print("action: ", a)

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        # Display results every simulated month
        if info['month'] != current_month:
            current_month = info['month']
            print('Reward: {}'.format(sum(rewards)))
            print('Info: {}'.format(info))

    print('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
                                                                              np.mean(rewards), sum(rewards)))
env.close()
