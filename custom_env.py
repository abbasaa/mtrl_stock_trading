import gym
import os
import pandas as pd
import gym_anytrading
from anytrading_torch import anytrading_torch
import matplotlib.pyplot as plt
import torch

TICKER = 'AAPL'
DATA_DIR = 'Tech'
WINDOW = 250
END_TIME = 754

data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
stock_prices = pd.read_csv(data_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = anytrading_torch(device, 'stocks-v0', stock_prices, (WINDOW, END_TIME), WINDOW)

for i in range(100):
    observation = env.reset()
    t = 0
    while True:
        t += 1
        if t == 503:
            action = 1
        elif t == 504:
            action = 0
        else:
            action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("info:", info)
            break

plt.cla()
env.render_all()
plt.show()