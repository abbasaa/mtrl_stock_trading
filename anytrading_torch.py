import gym
import gym_anytrading
import math
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import torch

class anytrading_torch():
    def __init__(self, device, anytrading_env, stock_data, frame, window_size):
        self.device = device
        self.env = gym.make(anytrading_env, df = stock_data, frame_bound=frame, window_size=window_size)
        self.action_space = self.env.action_space

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = torch.tensor([observation], dtype=torch.float, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float, device=self.device)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = torch.tensor([observation], dtype=torch.float, device=self.device)
        return observation

    def render_all(self):
        self.env.render_all()
