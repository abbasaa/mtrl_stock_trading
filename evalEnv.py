import math
from anytrading_torch import anytrading_torch
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from DQN import DQN
import os
from ReplayMemory import ReplayMemory, Transition
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

start = time.perf_counter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TICKER = 'GOOGL'

WINDOW = 250
END_TIME = 700

evalEnv = anytrading_torch(device, 'stocks-v0', STOCKS_GOOGL, (END_TIME - 100, END_TIME), WINDOW)

N_ACTIONS = evalEnv.action_space.n
HIDDEN_DIM = 5
N_HISTORIC_PRICES = 1

PolicyNet = DQN(N_HISTORIC_PRICES+2, HIDDEN_DIM, N_ACTIONS, TICKER, device)
TargetNet = DQN(N_HISTORIC_PRICES+2, HIDDEN_DIM, N_ACTIONS, TICKER, device)
TargetNet = TargetNet.to(device)
PolicyNet = PolicyNet.to(device)

TargetNet.load_state_dict(PolicyNet.state_dict())
TargetNet.eval()
PolicyNet.eval()


def select_action(position, time_idx, last_price):
    with torch.no_grad():
        return PolicyNet(position, time_idx, last_price).max(1)[1].view(1, 1).float()


best_reward = float('-inf')
best_profit = float('-inf')

observation = evalEnv.reset()
position = torch.zeros((1, 1),  dtype=torch.float, device=device)
t = 0
while True:
    print(t)
    t += 1
    # select and perform action
    action = select_action(position, [t], observation[:, -1, 0])
    next_position = action
    next_observation, reward, done, info = evalEnv.step(action)
    position = next_position
    observation = next_observation

    if done:
        evalEnv.render_all()
        print("info:", info, " action:", action)
        if info['total_reward'] > best_reward:
            best_reward = info['total_reward']
            print('Saving Best Reward Model for Episode ...')
            torch.save({
                'dqn_state_dict': PolicyNet.state_dict(),
            }, os.path.join(os.curdir, 'models', f'dqn_reward_{TICKER}.pth'))
            plt.savefig(os.path.join(os.curdir, 'models', f'dqn_reward_{TICKER}.png'))
        if info['total_profit'] > best_profit:
            best_profit = info['total_profit']
            print('Saving Best Profit Model for Episode ...')
            torch.save({
                'dqn_state_dict': PolicyNet.state_dict(),
            }, os.path.join(os.curdir, 'models', f'dqn_profit_{TICKER}.pth'))
            plt.savefig(os.path.join(os.curdir, 'models', f'dqn_profit_{TICKER}.png'))
        break

plt.show()