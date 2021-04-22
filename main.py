import math
import os
import random
import time

# from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from anytrading_torch import anytrading_torch
from DQN import DQN
from preprocess import import_stock_to_env, getimfs
from ReplayMemory import ReplayMemory, Transition

start = time.perf_counter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TICKER = 'GOOGL'  # TODO: make this an argument
DATA_DIR = 'Tech'
WINDOW = 250
END_TIME = 755

# CHECK DIR FOR FILE IF NOT THROW ERROR/RUN PREPROCESS
imf_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_IMF.npy')
denorm_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_denorm.npy')
data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
env = import_stock_to_env(data_file)
stock_prices = env.env.prices

if not os.path.isfile(imf_filename) or not os.path.isfile(denorm_filename):
    print(f'IMF or denorm file missing for stock: {TICKER}')
    getimfs(stock_prices, WINDOW, data_file[:-4])
    print(f'Preprocessing for stock: {TICKER} complete ...')
# TODO: should we read in window and end time for preprocess ?

env = anytrading_torch(device, 'stocks-v0', stock_prices, (WINDOW, END_TIME), WINDOW)

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.05
EPS_DELAY = 2000
EPS_DECAY = .99975
TARGET_UPDATE = 10

N_ACTIONS = env.action_space.n
HIDDEN_DIM = 5
N_HISTORIC_PRICES = 1

PolicyNet = DQN(N_HISTORIC_PRICES+2, HIDDEN_DIM, N_ACTIONS, TICKER, device)
TargetNet = DQN(N_HISTORIC_PRICES+2, HIDDEN_DIM, N_ACTIONS, TICKER, device)
TargetNet = TargetNet.to(device)
PolicyNet = PolicyNet.to(device)
optimizer = optim.Adam(PolicyNet.parameters())
memory = ReplayMemory(256)

# load checkpoint if possible
EPISODE_START = 0
while True:
    checkpoint_path = os.path.join(os.curdir, 'checkpoints', f'dqn_{EPISODE_START}.pth')
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
else:
    TargetNet.load_state_dict(PolicyNet.state_dict())

TargetNet.eval()
exploration = []
intentional_reward = []
episode_durations = []
steps_done = 0


def select_action(position, time_idx, last_price):
    global steps_done
    sample = random.random()
    decay = 1
    if steps_done > EPS_DELAY:
        decay = math.pow(EPS_DECAY, steps_done)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * decay
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return PolicyNet(position, time_idx, last_price).max(1)[1].view(1, 1).float(), True
    else:
        exploration[-1] += 1
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.float), False


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)

# State: (position, time, last_price)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))


    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_positions = torch.cat([s[0] for s in batch.next_state if s is not None])
    non_final_next_times = [s[1] for s in batch.next_state if s is not None]
    non_final_next_last_prices = torch.cat([s[2] for s in batch.next_state if s is not None])

    state_batch = list(zip(*batch.state))
    position_batch = torch.cat(state_batch[0])
    times_batch = list(state_batch[1])
    last_price_batch = torch.cat(state_batch[2])

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = PolicyNet(position_batch, times_batch,
                                    last_price_batch).gather(1, action_batch.long())
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = TargetNet(non_final_next_positions, non_final_next_times,
                                                  non_final_next_last_prices).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in PolicyNet.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()


NUM_EPISODES = 300
for i_episode in range(EPISODE_START, NUM_EPISODES):
    print("EPISODE: ", i_episode)
    # Initialize the environment and state
    exploration.append(0)
    observation = env.reset()
    position = torch.zeros((1, 1),  dtype=torch.float, device=device)
    print("[0]", end='', flush=True)
    t = 0
    while True:
        t += 1
        # select and perform action
        action, exploit = select_action(position, [t], observation[:, -1, 0])
        next_position = action
        next_observation, reward, done, info = env.step(action)

        memory.push((position, t, observation[:, -1, 0]), action, (next_position, t+1, next_observation[:, -1, 0]), reward)

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
            # episode_durations.append(t + 1)
            # plot_durations()
            print(i_episode, " info:", info, " action:", action)
            break
    if i_episode % TARGET_UPDATE == 0:
        TargetNet.load_state_dict(PolicyNet.state_dict())

    # save checkpoint
    print(f'Saving checkpoint for Episode {i_episode} ...')
    torch.save({
        'dqn_state_dict': PolicyNet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(os.curdir, 'checkpoints', f'dqn_{i_episode}.pth'))

stop = time.perf_counter()
print(f"Completed execution in: {stop - start:0.4f} seconds")

fig, ax = plt.subplots()
exploration = [e / (END_TIME - WINDOW) for e in exploration]
ax.plot(list(range(NUM_EPISODES)), exploration)
ax.set_title("Exploration vs episodes")
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(list(range(len(intentional_reward))), intentional_reward)
ax2.set_title("Intentional reward over time")
plt.show()

env.render_all()
plt.title("DQN After 300 Episodes")
plt.show()