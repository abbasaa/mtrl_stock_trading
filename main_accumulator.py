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
import pandas as pd


device = torch.device("cuda")
num_agents = 5
WINDOW = 10
END_TIME = 200
num_episodes = 5

# list file names of agent stocks and test stocks
file_names_training = ['TEST_STOCKS/AMAT.csv', 'TEST_STOCKS/AMD.csv', 'TEST_STOCKS/MCHP.csv', "TEST_STOCKS/NVDA.csv", "TEST_STOCKS/XLNX.csv"]
file_names_test = ["TEST_STOCKS/AAPL.csv","TEST_STOCKS/ADI.csv"]

# import stock CSV files to panda dataframe and create environment
def import_stock_to_env(file_path):
    stock_data = pd.read_csv(file_path) 
    env = anytrading_torch(device, 'stocks-v0', stock_data, (WINDOW, END_TIME), WINDOW)
    return env

# train single agents with 1 training stock each
# save each agent policy net
def train_agents(num_agents, file_names_training):
    stock_prices = np.zeros((num_agents, END_TIME))
    policy_nets = []
    for i in range(num_agents):
        print('Training Agent ',i+1)
        env = import_stock_to_env(file_names_training[i])
        stock_prices[i,:] = env.env.prices
        policy_net = learn_policy(env, num_episodes)
        policy_nets.append(policy_net)
        env.reset()
    return stock_prices, policy_nets

# from the policy nets of the trained agents, run each test stock
# to get action values
def run_agents(num_agents, policy_nets, file_names_test):
    test_stock_prices = np.zeros((len(file_names_test), END_TIME))
    values  = np.zeros((END_TIME - WINDOW,2,num_agents,len(file_names_test)))
    for i in range(len(file_names_test)):
        env = import_stock_to_env(file_names_test[i])
        test_stock_prices[i,:] = env.env.prices
        for j in range(num_agents):
            observation = env.reset()
            position = torch.zeros((1, 1),  dtype=torch.float, device=device)
            for k in range(END_TIME - WINDOW - 1):
                action_new, value = step_SA(policy_nets[j], position, observation)
                values[k,:,j,i] = value.cpu()
                next_position = action_new
                next_observation, reward, done, info = env.step(action_new)
                position = next_position
                observation = next_observation
        env.reset()
    return values, test_stock_prices

# accumulator takes prices of stocks and sector to calculate correlation coefficients,
# gets weighted average of action values for each test stock saved in HP_A,
# uses action values to calculate accumulator actions for each test stock
def run_accumulator(sector_price, stock_prices, test_stock_prices, values, file_names_test):
    HP_A = accumulator(sector_price, stock_prices, test_stock_prices, values)
    num_tests = len(file_names_test)
    for i in range(num_tests):
        intentional_reward = []
        reward_total = 0
        env = import_stock_to_env(file_names_test[i])
        env.reset()
        for j in range(END_TIME - WINDOW - 1):
            # select action
            if np.argmax(HP_A[j,:,i]) == 0:
                action = torch.zeros((1,1),  dtype=torch.float, device=device)
            else:
                action = torch.ones((1,1),  dtype=torch.float, device=device)
            print(action)
            next_position = action
            next_observation, reward, done, info = env.step(action)
            print(info)
            reward_total = reward + reward_total
            if reward != 0:
                intentional_reward.append(reward[0].item())
        print('total reward for test ',i,': ',reward_total)
        fig2, ax2 = plt.subplots()
        ax2.plot(list(range(len(intentional_reward))), intentional_reward)
        ax2.set_title("Intentional reward over time for test stock %d" %i)
        plt.show()

        env.render_all()
        plt.title("DQN After %d Episodes" % num_episodes)
        plt.show()
        env.reset()

# train agents and run on test stocks
stock_prices, policy_net = train_agents(num_agents, file_names_training)
values, test_stock_prices = run_agents(num_agents, policy_net, file_names_test)

# may need to change -- using sum of stock prices as "sector" price
sector = np.append(stock_prices,test_stock_prices,0)
sector_price = np.sum(sector,0)

# run accumulator on test stocks
run_accumulator(sector_price, stock_prices, test_stock_prices, values, file_names_test)




