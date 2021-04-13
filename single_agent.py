import math
from anytrading_torch import anytrading_torch
import matplotlib.pyplot as plt
from DQN import DQN
import torch.optim as optim
from ReplayMemory import ReplayMemory, Transition
import random
import torch
import torch.nn.functional as F


device = torch.device("cuda")
WINDOW = 10
END_TIME = 200


BATCH_SIZE = 128
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.05
EPS_DELAY = 2000
EPS_DECAY = .99975
TARGET_UPDATE = 10




steps_done = 0

def select_action(pos, obs):
    global steps_done
    sample = random.random()
    decay = 1
    if steps_done > EPS_DELAY:
        decay = math.pow(EPS_DECAY, steps_done)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * decay
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(pos, obs).max(1)[1].view(1, 1).float(), True
    else:
        exploration[-1] += 1
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.float), False


episode_durations = []


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


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_positions = torch.cat([s[0] for s in batch.next_state if s is not None])
    non_final_next_observations = torch.cat([s[1] for s in batch.next_state if s is not None])
    state_batch = list(zip(*batch.state))
    position_batch = torch.cat(state_batch[0])
    observation_batch = torch.cat(state_batch[1])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(position_batch,
                                     observation_batch).gather(1, action_batch.long())


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_positions,
                                                   non_final_next_observations).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()

def learn_policy(env, num_episodes):
    
    
    global n_actions
    n_actions = env.action_space.n
    global observation_dim
    observation_dim = 2

    global policy_net
    policy_net = DQN(WINDOW, observation_dim, WINDOW//2, WINDOW//4, n_actions)
    global target_net 
    target_net = DQN(WINDOW, observation_dim, WINDOW//2, WINDOW//4, n_actions)
    target_net = target_net.to(device)
    policy_net = policy_net.to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    global optimizer
    optimizer = optim.RMSprop(policy_net.parameters())
    global memory
    memory = ReplayMemory(256)


    global exploration
    exploration = []
    global intentional_reward
    intentional_reward = []
    
    
    
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        exploration.append(0)
        observation = env.reset()
        position = torch.zeros((1, 1),  dtype=torch.float, device=device)
        t = 0
        while True:
            t += 1
            # select and perform action
            action, exploit = select_action(position, observation)
            next_position = action
            next_observation, reward, done, info = env.step(action)

            memory.push((position, observation), action, (next_position, next_observation), reward)

            optimize_model()

            if exploit and reward != 0:
                intentional_reward.append(reward[0].item())

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                print(i_episode, " info:", info, " action:", action)
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    return policy_net