import logging

import gymnasium as gym
import numpy as np
import sys
import sinergym
from fugu_examples.rewards import ExpRewardV2
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

    # 设定步长为1分钟
    extra_params = {
        'timesteps_per_hour': 60
    }

    # 使用预设好的深圳天气环境模型
    env = gym.make('Eplus-smalldatacenter-hot-continuous-stochastic-v1', config_params=extra_params,
                   reward=ExpRewardV2, reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['cooling_coil_demand_rate', 'fan_demand_rate'],  # 制冷+风扇能耗和
            'range_comfort_winter': (25.0, 27.0),
            'range_comfort_summer': (25.0, 27.0),
            'energy_weight': 0.1,
            'lambda_energy': 0.01,
            'lambda_temperature': 0.1,
        })

    env = NormalizeAction(env, normalize_range=(0, 1))

    # 可对观测值归一化
    # env = NormalizeObservation(env)
    # env = LoggerWrapper(env)
    # env = CSVLogger(env)

    print("test")
    obs, info = env.reset()

    rewards = []
    truncated = terminated = False
    current_month = 0

    step = 0
    while not (terminated or truncated):
        # Random action selection
        a = env.action_space.sample()
        # Perform action and receive env information

        a = np.array([0.6], dtype=np.float32)  ## 制冷恒定设置25.6
        step += 1
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        obs_dict = dict(zip(env.get_wrapper_attr(
            'observation_variables'), obs))
        print('====================step{} data================'.format(step))
        info.update(obs_dict)
        print('observation dic:{}'.format(info))
        print('Reward: {}'.format(sum(rewards)))
        print(
            f'temp:{info['air_temperature']}, set temp:{info['action'][0]}, cooling demand:{info['cooling_coil_demand_rate']}, fan:{info['fan_demand_rate']}')

    print('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(1,
                                                                        np.mean(rewards), sum(rewards)))
    env.close()

##生成excel文件记录每一列的值
