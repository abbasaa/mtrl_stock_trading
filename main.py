import math
import os
import random
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim


from anytrading_torch import anytrading_torch
from DQN import DQN
from preprocess import getimfs
from ReplayMemory import ReplayMemory, Transition

start = time.perf_counter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TICKER = sys.argv[1]
DATA_DIR = 'Tech'
WINDOW = 250
END_TIME = 754

# CHECK DIR FOR FILE IF NOT THROW ERROR/RUN PREPROCESS
imf_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_IMF.npy')
denorm_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_denorm.npy')
data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
stock_prices = pd.read_csv(data_file)


if not os.path.isfile(imf_filename) or not os.path.isfile(denorm_filename):
    print(f'IMF or denorm file missing for stock: {TICKER}')
    getimfs(stock_prices, WINDOW, data_file[:-4])
    print(f'Preprocessing for stock: {TICKER} complete ...')

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
GAMMA = 0.1
EPS_START = 0.9
EPS_END = 0.1
EPS_DELAY = 2*(END_TIME - WINDOW) # Increase?
EPS_DECAY = .99995 # Decrease?
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
optimizer = optim.RMSprop(PolicyNet.parameters())
memory = ReplayMemory(REPLAY_SIZE)
Actions = np.zeros((1, 2))

# make model folders
models_dir = os.path.join(os.curdir, 'models')
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)
models_dir = os.path.join(models_dir, TICKER)
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

# load checkpoint if possible
EPISODE_START = 0
steps_done = 0
checkpoints_dir = os.path.join(os.curdir, 'checkpoints')
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoints_dir = os.path.join(checkpoints_dir, 'dqn')
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoints_dir = os.path.join(checkpoints_dir, TICKER)
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoint_files = [f for f in os.listdir(checkpoints_dir) if os.path.isfile(os.path.join(checkpoints_dir, f))]
if len(checkpoint_files) != 0:
    max_eps = -2 ** 32
    for file in checkpoint_files:
        cur_eps = int(file.split('.')[1])
        if cur_eps > max_eps:
            max_eps = cur_eps
    EPISODE_START = max_eps
    print(f'Loading model from checkpoint at episode: {EPISODE_START}')
    checkpoint_file = os.path.join(checkpoints_dir, f'dqn.{EPISODE_START}.pth')
    checkpoint = torch.load(checkpoint_file)
    PolicyNet.load_state_dict(checkpoint['dqn_state_dict'])
    TargetNet.load_state_dict(checkpoint['dqn_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    Actions = checkpoint['actions']
    steps_done = EPISODE_START * (END_TIME - WINDOW)
    if len(checkpoint_files) > 1:
        print('Removing older checkpoint files')
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if (os.path.isfile(os.path.join(checkpoints_dir, f))
                                                                       and f != f'dqn.{EPISODE_START}.pth')]
        for f in checkpoint_files:
            os.remove(os.path.join(checkpoints_dir, f))

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
    # sample = random.random()
    # decay = 1
    # if steps_done > EPS_DELAY:
         #decay = math.pow(EPS_DECAY, steps_done - EPS_DELAY)
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * decay
    steps_done += 1
    # if sample > eps_threshold:
    #     with torch.no_grad():
    #         return PolicyNet(positions, time_idx, last_price).max(1)[1].view(1, 1).float(), True
    # else:
    #     exploration[-1] += 1
    #     return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.float), False
    if steps_done < EPS_DELAY:
        a = random.randrange(N_ACTIONS)
        is_exploit = False
    else:
        u_t = np.sqrt((2 * np.log(steps_done)) / Actions)
        q_t = PolicyNet(positions, time_idx, last_price).detach().cpu().numpy()
        a = np.argmax(u_t + q_t)
        is_exploit = bool(q_t[:, 0] - q_t[:, 1] > u_t[:, 0] - u_t[:, 1])
    Actions[:, a] += 1
    return torch.tensor([[a]], device=device, dtype=torch.float), is_exploit


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
    # TODO ADD penalization for all sell//all buy
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Loss and Back Prop
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in PolicyNet.parameters():
        if param.requires_grad:
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
        }, os.path.join(models_dir, f'dqn_reward.{GAMMA}-{TICKER}.pth'))
    if mean_profit >= highest_profit:
        highest_profit = mean_profit
        print('Saving Best Profit Model for Episode ...')
        torch.save({
            'dqn_state_dict': PolicyNet.state_dict(),
        }, os.path.join(models_dir, f'dqn_profit.{GAMMA}-{TICKER}.pth'))


NUM_EPISODES = 100
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
    torch.save({
        'dqn_state_dict': PolicyNet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'actions': Actions,
    }, os.path.join(checkpoints_dir, f'dqn.{i_episode}.pth'))

    # remove last checkpoint
    if i_episode-1 >= 0 and os.path.isfile(os.path.join(checkpoints_dir, f'dqn.{i_episode-1}.pth')):
        print(f'Removing old checkpoint file for episode {i_episode-1}')
        os.remove(os.path.join(checkpoints_dir, f'dqn.{i_episode-1}.pth'))

stop = time.perf_counter()
print(f"Completed execution in: {stop - start:0.4f} seconds")


def smooth(x, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    conv_x = np.convolve(x, kernel, mode='same')
    return conv_x

fig, ax = plt.subplots()
exploration = [e / (END_TIME - WINDOW) for e in exploration]
ax.plot([e for e in range(len(exploration))], exploration)
ax.set_title("Exploration vs episodes")
ax.set_xlabel('Episodes')
ax.set_ylabel('Exploration')
fig.savefig(os.path.join(models_dir, f'Exploration.{NUM_EPISODES}-{GAMMA}-{TICKER}.png'))

fig2, ax2 = plt.subplots()
ax2.plot(smooth(intentional_reward, kernel_size=40))
ax2.set_title("Intentional Reward vs Time")
ax2.set_xlabel('Reward')
ax2.set_ylabel('Time')
fig2.savefig(os.path.join(models_dir, f'Intentional_Reward.{NUM_EPISODES}-{GAMMA}-{TICKER}.png'))

fig3, ax3 = plt.subplots()
ax3.plot([r for r in range(len(train_reward))], smooth(train_reward), 'r', label="train")
ax3.plot([r*EVAL for r in range(len(eval_reward))], eval_reward, 'b', label="eval")
ax3.set_title("Rolling Average Total Reward vs Episodes")
ax3.set_xlabel('Total Reward')
ax3.set_ylabel('Episodes')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels)
fig3.savefig(os.path.join(models_dir, f'Reward.{NUM_EPISODES}-{GAMMA}-{TICKER}.png'))

fig4, ax4 = plt.subplots()
ax4.plot([p for p in range(len(train_profit))], smooth(train_profit), 'r', label="train")
ax4.plot([p*EVAL for p in range(len(eval_profit))], eval_profit, 'b', label="eval")
ax4.set_title("Rolling Average Total Profit vs Episodes")
ax4.set_xlabel('Total Profit')
ax4.set_ylabel('Episodes')
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels)
fig4.savefig(os.path.join(models_dir, f'Profit.{NUM_EPISODES}-{GAMMA}-{TICKER}.png'))
plt.cla()

PolicyNet.load_state_dict(torch.load(os.path.join(models_dir, f'dqn_profit.{GAMMA}-{TICKER}.pth'))['dqn_state_dict'])
obs = env.reset()
pos = torch.zeros((1, 1), dtype=torch.float, device=device)
t_step = -1
while True:
    t_step += 1
    with torch.no_grad():
        act = PolicyNet(pos, [t_step], obs[:, -1, 0]).max(1)[1].view(1, 1).float()
    obs, _, is_done, inf = env.step(act)
    pos = act
    if is_done:
        break
env.render_all()
plt.title(f"DQN After {NUM_EPISODES} Episodes")
plt.set_xlabel('Time (Days)')
plt.set_ylabel('Prices')
plt.savefig(os.path.join(models_dir, f'Environment.{NUM_EPISODES}-{GAMMA}-{TICKER}.png'))
