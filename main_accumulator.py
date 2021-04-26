#!/usr/bin/env python
# coding: utf-8

# In[4]:

import os
import sys
import math
import time
from anytrading_torch import anytrading_torch
import matplotlib.pyplot as plt
from DQN import DQN
import torch.optim as optim
from ReplayMemory import ReplayMemory, Transition
import random
import torch
import torch.nn.functional as F
from accumulator import *
import pandas as pd


device = torch.device("cpu")
start = time.perf_counter()

WINDOW = 250
END_TIME = 754
HIDDEN_DIM = 5
N_HISTORIC_PRICES = 1
TICKER = "INTC"

policy_net = DQN(N_HISTORIC_PRICES + 2, HIDDEN_DIM, 2, TICKER, device)
policy_net = policy_net.to(device)


# import stock CSV files to panda dataframe and create environment
def import_stock_to_env(filename: str):
    stock_data = pd.read_csv(filename)
    env = anytrading_torch(device, 'stocks-v0', stock_data, (WINDOW, END_TIME), WINDOW)
    return env


DATA_DIR = "trained_models"
TEST_DIR = "Tech"
TRAINED_DIR = "Tech/TRAINED"


# from the policy nets of the trained agents, run each test stock
# to get action values
def run_agents():
    num_agents = 0
    num_tests = 0
    for file in os.listdir(TRAINED_DIR):
        num_agents = num_agents + 1
    for file in os.listdir(TEST_DIR):
        num_tests = num_tests + 1
    test_stock_prices = np.zeros((num_tests, END_TIME))
    values  = np.zeros((END_TIME - WINDOW,2,num_agents,num_tests))
    fnum = 0

    for testfile in os.listdir(TEST_DIR):
        env_filname = os.path.join(os.curdir, TEST_DIR, testfile)
        env = import_stock_to_env(env_filname)
        test_stock_prices[fnum,:] = env.env.prices
        j = 0
        for filetrained in os.listdir(DATA_DIR):
            filename = os.path.join(os.curdir, DATA_DIR, filetrained)
            trained_agent = torch.load(filename, map_location=torch.device('cpu'))
            policy_net.load_state_dict(trained_agent['dqn_state_dict'])
            observation = env.reset()
            position = torch.zeros((1, 1),  dtype=torch.float, device=device)
            t_step = -1
            print("Running agent %d on test stock %d" %(j,fnum))
            for k in range(END_TIME - WINDOW - 1):
                t_step += 1
                action_new, value = step_SA(policy_net, position, observation, t_step)
                values[k,:,j,fnum] = value
                next_position = action_new
                next_observation, reward, done, info = env.step(action_new)
                position = next_position
                observation = next_observation
            j = j + 1
        fnum = fnum + 1
        env.reset()
    return values, test_stock_prices, num_agents

# accumulator takes prices of stocks and sector to calculate correlation coefficients,
# gets weighted average of action values for each test stock saved in HP_A,
# uses action values to calculate accumulator actions for each test stock
def run_accumulator(num_agents, test_stock_prices, values):
    stock_prices = np.zeros((num_agents, END_TIME))
    num_file = 0
    for file in os.listdir(TRAINED_DIR):
        env_filname = os.path.join(os.curdir, TRAINED_DIR, file)
        env = import_stock_to_env(env_filname)
        stock_prices[num_file,:] = env.env.prices
        num_file = num_file + 1
    
    sector = np.append(stock_prices,test_stock_prices,0)
    sector_price = np.sum(sector,0)

    HP_A = accumulator(sector, stock_prices, test_stock_prices, values)
    i = 0
    for testfile in os.listdir(TEST_DIR):
        test_filename = os.path.join(os.curdir, TEST_DIR, testfile)
        intentional_reward = []
        reward_total = 0
        env = import_stock_to_env(test_filename)
        env.reset()
        print("Deciding actions for test stock ", i)
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
        plt.title("Actions on stock %s" %testfile )
        plt.show()
        env.reset()
        i = i + 1

# train agents and run on test stocks
values, test_stock_prices, num_agents = run_agents()

# may need to change -- using sum of stock prices as "sector" price

# run accumulator on test stocks
run_accumulator(num_agents, test_stock_prices, values)




