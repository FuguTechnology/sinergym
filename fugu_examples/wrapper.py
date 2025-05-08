import gymnasium as gym
import numpy as np
from gymnasium import Env
from typing import Any, Callable, Dict, List, Optional, Tuple
from pid import PID

from sinergym.utils.constants import LOG_WRAPPERS_LEVEL, YEAR
from sinergym.utils.logger import LoggerStorage, TerminalLogger
from copy import deepcopy


class PidActionWrapper(gym.ActionWrapper):
    """Wrapper to normalize action space.
    """

    logger = TerminalLogger().getLogger(name='WRAPPER NormalizeAction',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 normal_value_bias: Tuple[float, float] = (6, 0),
                 normalize_range: Tuple[float, float] = (0, 1.0)):
        """Wrapper to normalize action space in default continuous environment (not to combine with discrete environments). The action will be parsed to real action space before to send to the simulator (very useful ion DRL algorithms)

        Args:
            env (Env): Original environment.
            normalize_range (Tuple[float,float]): Range to normalize action variable values. Defaults to values between [-1.0,1.0].
        """
        super().__init__(env)

        # Checks
        if self.get_wrapper_attr('is_discrete'):
            self.logger.critical(
                'The original environment must be continuous instead of discrete')
            raise TypeError

        self.controllers = []
        self.temps = []

        # 出风温度设定转回风温度设定偏置值
        low_bias, upper_bias = normal_value_bias
        self.low_bias = low_bias
        self.upper_bias = upper_bias
        # Define real space for simulator
        self.real_space = deepcopy(self.action_space)



        # Define normalize space
        lower_norm_value, upper_norm_value = normalize_range



        self.normalized_space = gym.spaces.Box(
            low=np.array(
                np.repeat(
                    lower_norm_value,
                    env.get_wrapper_attr('action_space').shape[0]),
                dtype=np.float32),
            high=np.array(
                np.repeat(
                    upper_norm_value,
                    env.get_wrapper_attr('action_space').shape[0]),
                dtype=np.float32),
            dtype=env.get_wrapper_attr('action_space').dtype)

        # Updated action space to normalized space
        self.action_space = self.normalized_space
        self.action_dim = env.get_wrapper_attr('action_space').shape[0]

        self.logger.info(
            f'New normalized action Space: {self.action_space}')
        self.logger.info('Wrapper initialized')

    def reset(self):
        self.controllers = [PID() for i in range(self.action_dim)]
        self.temps = []
        obs, info = super().reset()
        obs_dict = dict(zip(self.get_wrapper_attr(
            'observation_variables'), obs))
        info.update(obs_dict)
        for i in range(1, self.action_dim + 1):
            self.temps.append(info['thermal_zone_{}_air_temperature'.format(i)])
        return obs,info

    def reverting_action(self,
                         action: Any):
        """ This method maps a normalized action in a real action space.

        Args:
            action (Any): Normalize action received in environment

        Returns:
            np.array: Action transformed in simulator real action space.
        """

        ##todo  添加一层pid转换

        # Convert action to the original action space
        # if self.normalize:
        action_ = (action - self.normalized_space.low) * (self.real_space.high + self.upper_bias - self.real_space.low - self.low_bias) / \
                  (self.normalized_space.high - self.normalized_space.low) + self.real_space.low + self.low_bias
        # else:
        #     action_ = action
        for i in range(self.action_dim):

            setpoint = action_[i]
            action_[i] = self.controllers[i](self.temps[i], setpoint=setpoint)
            # print(f'i{i},setpoint{setpoint},action{action_[i]}')

        return action_

    def action(self, action: Any):
        return self.get_wrapper_attr('reverting_action')(action)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs_dict = dict(zip(self.get_wrapper_attr(
            'observation_variables'), obs))
        info.update(obs_dict)
        for i in range(1, self.action_dim + 1):
            self.temps[i-1] = info['thermal_zone_{}_air_temperature'.format(i)]
        return obs, reward, terminated, truncated, info
