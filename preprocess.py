import numpy as np
import gym
import gym_anytrading
#from PyEMD import CEEMDAN
import pandas as pd
import os
import torch
from anytrading_torch import anytrading_torch

device = torch.device("cuda")



# import stock CSV files to panda dataframe and create environment
def import_stock_to_env(file_path):
    stock_data = pd.read_csv(file_path) 
    env = anytrading_torch(device, 'stocks-v0', stock_data, (WINDOW, END_TIME), WINDOW)
    return env


#CEEMD = CEEMDAN(DTYPE=np.float16, trials=20)
NUM_IMF = 5


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


def getImfs(prices, window, file_name):
    print("Computing IMFs")
    # Prices: price_count,
    end = len(prices)
    # Steps = END
    IMFs = []
    denorm = []
    for i in range(end - window):
        print("IMF #", i)
        ceemd_out = CEEMD(prices[i:i+window], max_imf=NUM_IMF-1)
        denorm.append(np.array([(min(i), max(i)) for i in ceemd_out]))
        IMFs.append(np.array(list(map(normalize, ceemd_out))))
    IMFs = np.stack(IMFs)
    denorm = np.stack(denorm)
    # IMFs: Steps x Num_imfs x window,
    # Denorm: Steps x Num_imfs x (min, max)
    file_path = 'IMF/'+file_name+'_IMF.npy'
    np.save(file_path, IMFs)
    file_path = 'IMF/'+file_name+'_denorm.npy'
    np.save(file_path, denorm)


WINDOW = 250
END_TIME = 700
path = "TEST_STOCKS"
cwd = os.getcwd()
os.chdir(path)
for file in os.listdir():
    file_path = f"{path}\{file}"
    os.chdir(cwd)
    env = import_stock_to_env(file_path)
    stock_prices = env.env.prices
    getImfs(stock_prices, WINDOW,file[:-4])
    os.chdir(path)
