import math
import os
import random
import time
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from anytrading_torch import anytrading_torch
from DQN import DQN
from preprocess import import_stock_to_env, getimfs
from ReplayMemory import ReplayMemory, Transition

start = time.perf_counter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TICKER = sys.argv[1]
DATA_DIR = 'Tech'
WINDOW = 250
END_TIME = 755

# CHECK DIR FOR FILE IF NOT THROW ERROR/RUN PREPROCESS
imf_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_IMF.npy')
denorm_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_denorm.npy')
data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
stock_prices = pd.read_csv(data_file)


if not os.path.isfile(imf_filename) or not os.path.isfile(denorm_filename):
    print(f'IMF or denorm file missing for stock: {TICKER}')
    getimfs(stock_prices, WINDOW, data_file[:-4])
    print(f'Preprocessing for stock: {TICKER} complete ...')
# TODO: should we read in window and end time for preprocess ?

# Prepare Training and Evaluation Environments
env = anytrading_torch(device, 'stocks-v0', stock_prices, (WINDOW, END_TIME), WINDOW)
K_folds = 5
eval_envs = []
fold_length = (END_TIME - WINDOW) // K_folds
for i in range(K_folds):
    eval_envs.append(anytrading_torch(device, 'stocks-v0', stock_prices,
                                      (WINDOW + i*fold_length, WINDOW + (i+1)*fold_length), WINDOW))

# Hyperparameters
REPLAY_SIZE = 512
BATCH_SIZE = 128
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.05
EPS_DELAY = 2000
EPS_DECAY = .99975
TARGET_UPDATE = 10
EVAL = 10

# Initialize Networks, Memory and Optimizer
N_ACTIONS = env.action_space.n
HIDDEN_DIM = 5
N_HISTORIC_PRICES = 1
PolicyNet = DQN(N_HISTORIC_PRICES+2, HIDDEN_DIM, N_ACTIONS, TICKER, device)
TargetNet = DQN(N_HISTORIC_PRICES+2, HIDDEN_DIM, N_ACTIONS, TICKER, device)
TargetNet = TargetNet.to(device)
PolicyNet = PolicyNet.to(device)
optimizer = optim.Adam(PolicyNet.parameters())
memory = ReplayMemory(REPLAY_SIZE)

# load checkpoint if possible
EPISODE_START = 0
steps_done = 0
if not os.path.isdir(os.path.join(os.curdir, TICKER)):
    os.mkdir(os.path.join(os.curdir, 'checkpoints', TICKER))
while True:
    checkpoint_path = os.path.join(os.curdir, 'checkpoints', TICKER, f'dqn_{EPISODE_START}.pth')
    if not os.path.isfile(checkpoint_path):
        EPISODE_START = max(0, EPISODE_START-1)
        break
    EPISODE_START += 1

if EPISODE_START != 0:
    print(f'Loading model from checkpoint at episode: {EPISODE_START}')
    checkpoint_path = os.path.join(os.curdir, 'checkpoints', f'dqn_{EPISODE_START}.pth')
    checkpoint = torch.load(checkpoint_path)
    PolicyNet.load_state_dict(checkpoint['dqn_state_dict'])
    TargetNet.load_state_dict(checkpoint['dqn_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    steps_done = EPISODE_START * (END_TIME - WINDOW)
else:
    TargetNet.load_state_dict(PolicyNet.state_dict())
TargetNet.eval()

# Data Tracking
exploration = []
intentional_reward = []
train_reward = []
eval_reward = []
train_profit = []
eval_profit = []
highest_reward = float('-inf')
highest_profit = float('-inf')


def select_action(positions, time_idx, last_price):
    global steps_done
    sample = random.random()
    decay = 1
    if steps_done > EPS_DELAY:
        decay = math.pow(EPS_DECAY, steps_done)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * decay
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return PolicyNet(positions, time_idx, last_price).max(1)[1].view(1, 1).float(), True
    else:
        exploration[-1] += 1
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.float), False


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

    # Next State
    non_final_next_positions = torch.cat([s[0] for s in batch.next_state if s is not None])
    non_final_next_times = [s[1] for s in batch.next_state if s is not None]
    non_final_next_last_prices = torch.cat([s[2] for s in batch.next_state if s is not None])

    # State
    state_batch = list(zip(*batch.state))
    position_batch = torch.cat(state_batch[0])
    times_batch = list(state_batch[1])
    last_price_batch = torch.cat(state_batch[2])

    # Action & Reward
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Calculate State Values
    state_action_values = PolicyNet(position_batch, times_batch,
                                    last_price_batch).gather(1, action_batch.long())
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = TargetNet(non_final_next_positions, non_final_next_times,
                                                  non_final_next_last_prices).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Loss and Back Prop
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in PolicyNet.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()


def eval_model():
    mean_reward = 0
    mean_profit = 0
    for environment in eval_envs:
        obs = environment.reset()
        pos = torch.zeros((1, 1),  dtype=torch.float, device=device)
        t_step = -1
        while True:
            t_step += 1
            with torch.no_grad():
                act = PolicyNet(pos, [t_step], obs[:, -1, 0]).max(1)[1].view(1, 1).float()
            obs, _, is_done, inf = environment.step(act)
            pos = act
            if is_done:
                mean_reward += inf['total_reward']
                mean_profit += inf['total_profit']
                break
    mean_reward /= K_folds
    mean_profit /= K_folds
    eval_reward.append(mean_reward)
    eval_profit.append(mean_profit)
    global highest_reward
    global highest_profit
    if mean_reward > highest_reward:
        highest_reward = mean_reward
        print('Saving Best Reward Model for Episode ...')
        torch.save({
            'dqn_state_dict': PolicyNet.state_dict(),
        }, os.path.join(os.curdir, 'models', f'dqn_reward_{TICKER}.pth'))
    if mean_profit >= highest_profit:
        highest_profit = mean_profit
        print('Saving Best Profit Model for Episode ...')
        torch.save({
            'dqn_state_dict': PolicyNet.state_dict(),
        }, os.path.join(os.curdir, 'models', f'dqn_profit_{TICKER}.pth'))


NUM_EPISODES = 300
for i_episode in range(EPISODE_START, NUM_EPISODES):
    print("EPISODE: ", i_episode)
    # Initialize the environment and state
    exploration.append(0)
    observation = env.reset()
    position = torch.zeros((1, 1),  dtype=torch.float, device=device)
    t = -1

    while True:
        t += 1
        action, exploit = select_action(position, [t], observation[:, -1, 0])
        next_position = action
        next_observation, reward, done, info = env.step(action)

        memory.push((position, t, observation[:, -1, 0]), action, (next_position, t+1, next_observation[:, -1, 0]), reward)

        if steps_done % 16 == 0:
            optimize_model()

        position = next_position
        observation = next_observation

        if t % 100 == 0:
            print(f"[{t}]", end='', flush=True)
        if t % 11 == 0:
            print("=", end='', flush=True)

        if exploit and reward != 0:
            intentional_reward.append(reward[0].item())

        if done:
            print()
            print(i_episode, " info:", info, " action:", action)
            train_profit.append(info['total_profit'])
            train_reward.append(info['total_reward'])
            break

    if i_episode % TARGET_UPDATE == 0:
        TargetNet.load_state_dict(PolicyNet.state_dict())
    if i_episode % EVAL == 0:
        eval_model()

    # save checkpoint
    print(f'Saving checkpoint for Episode {i_episode} ...')
    if not os.path.isdir(os.path.join(os.curdir, TICKER)):
        os.mkdir(os.path.join(os.curdir, 'checkpoints', TICKER))
    torch.save({
        'dqn_state_dict': PolicyNet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(os.curdir, 'checkpoints', TICKER, f'dqn_{i_episode}.pth'))

stop = time.perf_counter()
print(f"Completed execution in: {stop - start:0.4f} seconds")

fig, ax = plt.subplots()
exploration = [e / (END_TIME - WINDOW) for e in exploration]
ax.plot([e for e in range(len(exploration))], exploration)
ax.set_title("Exploration vs episodes")
fig.savefig(f'Exploration_{TICKER}.png')

fig2, ax2 = plt.subplots()
ax2.plot([i for i in range(len(intentional_reward))], intentional_reward)
ax2.set_title("Intentional Reward vs Time")
fig2.savefig(f'Intentional_Reward_{TICKER}.png')

fig3, ax3 = plt.subplots()
ax3.plot([r for r in range(len(train_reward))], train_reward, 'r', label="train")
ax3.plot([r*EVAL for r in range(len(eval_reward))], eval_reward, 'b', label="eval")
ax3.set_title("Total Reward vs Episodes")
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels)
fig3.savefig(f'Reward_{TICKER}.png')

fig4, ax4 = plt.subplots()
ax4.plot([p for p in range(len(train_profit))], train_reward, 'r', label="train")
ax4.plot([p*EVAL for p in range(len(eval_profit))], eval_reward, 'b', label="eval")
ax4.set_title("Total Reward vs Episodes")
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels)
fig4.savefig(f'Profit_{TICKER}.png')

env.render_all()
plt.title(f"DQN After {NUM_EPISODES} Episodes")
plt.savefig('Environment.png')
