import logging

import gymnasium as gym
import numpy as np
from fugu_examples.rewards import ExpRewardV2
from sinergym.utils.logger import TerminalLogger
from wrapper import PidActionWrapper
from pid import PID

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
    env = gym.make('Eplus-largedatacenterHVAC-hot-continuous-stochastic-v1', config_params=extra_params,
                   reward=ExpRewardV2, reward_kwargs={
            'temperature_variables': ['thermal_zone_1_air_temperature','thermal_zone_2_air_temperature','thermal_zone_3_air_temperature','thermal_zone_4_air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (24.0, 26.0),
            'range_comfort_summer': (24.0, 26.0),
            'energy_weight': 0.1,
            'lambda_energy': 0.01,
            'lambda_temperature': 0.1,
        })

    # env = NormalizeAction(env, normalize_range=(0, 1))

    # 可对观测值归一化
    env = PidActionWrapper(env,  normalize_range=(0, 1))
    # env = LoggerWrapper(env)
    # env = CSVLogger(env)

    obs, info = env.reset()
    print("init info: {}".format(info))
    obs_dict = dict(zip(env.get_wrapper_attr(
        'observation_variables'), obs))
    print("init observation {}".format(obs_dict))
    info.update(obs_dict)
    rewards = []
    truncated = terminated = False
    current_month = 0
    # pidControllers = []
    # for i in range(1, 5):
    #     pidControllers.append(PID(setpoint=25))

    step = 0
    while not (terminated or truncated):
        # Random action selection
        a = env.action_space.sample()
        # Perform action and receive env information
        # for i in range(1,5):
        #     output = pidControllers[i-1](info['thermal_zone_{}_air_temperature'.format(i)])
        #     a[i-1] = 0.6
        step += 1
        print("action: {}".format(a))
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        # obs_dict = dict(zip(env.get_wrapper_attr(
        #     'observation_variables'), obs))
        # print('====================step{} data================'.format(step))
        # info.update(obs_dict)
        print('observation dic:{}'.format(info))
        for i in range(1,5):
            print('zone{} air temperature:{}'.format(i, info['thermal_zone_{}_air_temperature'.format(i)]))
        print('Reward: {}'.format(sum(rewards)))
        # print(
        #     f'temp:{info['air_temperature']}, set temp:{info['action'][0]}, cooling demand:{info['cooling_coil_demand_rate']}, fan:{info['fan_demand_rate']}')

    print('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(1,
                                                                        np.mean(rewards), sum(rewards)))
    env.close()


