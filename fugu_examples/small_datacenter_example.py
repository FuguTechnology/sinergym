import logging

import gymnasium as gym
import numpy as np
import sys
import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)
from sinergym.utils.rewards import LinearReward

# Logger


if __name__ == "__main__":

    terminal_logger = TerminalLogger()
    logger = terminal_logger.getLogger(
        name='MAIN',
        level=logging.DEBUG
    )

    #设定步长为1分钟
    extra_params={
        'timesteps_per_hour':60
    }

    #使用预设好的深圳天气环境模型
    env = gym.make('Eplus-smalldatacenter-hot-continuous-stochastic-v1',config_params=extra_params,
                   reward=LinearReward,reward_kwargs={
    'temperature_variables': ['air_temperature'],
    'energy_variables': ['HVAC_electricity_demand_rate'],
    'range_comfort_winter': (25.0, 27.0),
    'range_comfort_summer': (25.0, 27.0),
    'energy_weight': 0.5,

    })

    env = NormalizeAction(env,normalize_range=(0,1))

    #可对观测值归一化
    # env = NormalizeObservation(env)
    # env = LoggerWrapper(env)
    # env = CSVLogger(env)

    print("test")
    obs, info = env.reset()

    rewards = []
    truncated = terminated = False
    current_month = 0
    print(f'start info{info}')
    print(f'obs length{obs.shape[0]}')

    while not (terminated or truncated):

        # Random action selection
        a = env.action_space.sample()

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        obs_dict = dict(zip(env.get_wrapper_attr(
            'observation_variables'), obs))
        print('Reward: {}'.format(sum(rewards)))
        print('Info: {}'.format(info))
        print('observation dic:{}'.format(obs_dict))
        # print('obs:{}'.format(obs))

    print('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(1,
                                                                              np.mean(rewards), sum(rewards)))
    env.close()