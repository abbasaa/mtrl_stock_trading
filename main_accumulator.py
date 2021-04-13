#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
from anytrading_torch import anytrading_torch
import matplotlib.pyplot as plt
from DQN import DQN
import torch.optim as optim
from ReplayMemory import ReplayMemory, Transition
import random
import torch
import torch.nn.functional as F
from single_agent import *
from accumulator import *


device = torch.device("cuda")
num_stocks = 5
WINDOW = 10
END_TIME = 200
num_episodes = 5
memory = ReplayMemory(256)



# weird bug with start time -- some times these values work and sometimes they cause an error and has to be re-run
start_time = np.random.randint(10, 1500, size=5)
#start_time = [50, 120, 210, 310, 600]
print('Start times for stocks: ',start_time)
stock_prices = np.zeros((num_stocks,200))
sector = np.zeros((1,200))
env = anytrading_torch(device, 'stocks-v0', (10, 200), 10)
sector[0,:] = env.env.prices
values  = np.zeros((190,2,num_agents))

for i in range(num_stocks):
    print('Training Agent ',i+1)
    env.reset()
    env = anytrading_torch(device, 'stocks-v0', (start_time[i], start_time[i] + 190), WINDOW)
    stock_prices[i,:] = env.env.prices
    policy_net = learn_policy(env, num_episodes)
    policy_net = policy_net.to(device)
    env.reset()
    env = anytrading_torch(device, 'stocks-v0', (10, 200), 10)
    observation = env.reset()
    position = torch.zeros((1, 1),  dtype=torch.float, device=device)
    for j in range(190):
        # select and perform action
        action_new, value = step_SA(policy_net, position, observation)
        values[j,:,i] = value
        next_position = action_new
        next_observation, reward, done, info = env.step(action_new)
        position = next_position
        observation = next_observation


HP_A = accumulator(sector, stock_prices, values)
intentional_reward = []
reward_total = 0
env = anytrading_torch(device, 'stocks-v0', (10, 200), 10)
env.reset()
for i in range(190):
    # select and perform action
    if np.argmax(HP_A[i,:]) == 0:
        action = torch.zeros((1,1),  dtype=torch.float, device=device)
    else:
        action = torch.ones((1,1),  dtype=torch.float, device=device)
    print('action: ',action)
    next_position = action
    next_observation, reward, done, info = env.step(action)
    print(info)
    reward_total = reward + reward_total
    if reward != 0:
        intentional_reward.append(reward[0].item())


print('total reward: ',reward_total)
#fig, ax = plt.subplots()
#exploration = [e / (END_TIME - WINDOW) for e in exploration]
#ax.plot(list(range(num_episodes)), exploration)
#ax.set_title("Exploration vs episodes")
#plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(list(range(len(intentional_reward))), intentional_reward)
ax2.set_title("Intentional reward over time")
plt.show()

env.render_all()
plt.title("DQN After 300 Episodes")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




