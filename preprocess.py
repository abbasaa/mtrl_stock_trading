import os

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
import torch

from anytrading_torch import anytrading_torch

device = torch.device("cuda")
CEEMD = CEEMDAN(DTYPE=np.float16, trials=20)
NUM_IMF = 5
WINDOW = 250
END_TIME = 755
DATA_DIR = "Tech"


# import stock CSV files to panda dataframe and create environment
def import_stock_to_env(filename: str):
    stock_data = pd.read_csv(filename)
    env = anytrading_torch(device, 'stocks-v0', stock_data, (WINDOW, END_TIME), WINDOW)
    return env


def normalize(x):
    # return (x - min(x)) / (max(x) - min(x))
    return np.nan_to_num((x - np.amin(x, axis=1)[:, np.newaxis]) / ((np.amin(x, axis=1) - np.amax(x, axis=1))[:, np.newaxis]))


def getimfs(prices, window, filename):
    print(f'Computing IMFs for {filename}')
    # Prices: price_count,
    end = len(prices)
    # Steps = END
    IMFs = []
    denorm = []
    for i in range(end - window):
        print(f'IMF #{i}')
        ceemd_out = CEEMD(prices[i:i + window], max_imf=NUM_IMF - 1)
        if ceemd_out.shape[0] < NUM_IMF:
            padding = np.zeros((NUM_IMF-ceemd_out.shape[0], ceemd_out.shape[1]))
            ceemd_out = np.concatenate((padding, ceemd_out), axis=0)

        denorm.append(np.array([(min(i), max(i)) for i in ceemd_out]))
        # IMFs.append(np.array(list(map(normalize, ceemd_out))))
        IMFs.append(normalize(ceemd_out))
    IMFs = np.stack(IMFs)
    denorm = np.stack(denorm)
    # IMFs: Steps x Num_imfs x window,
    # Denorm: Steps x Num_imfs x (min, max)
    file_path = os.path.join(os.path.curdir, 'IMF', f'{filename}_IMF.npy')
    np.save(file_path, IMFs)
    file_path = os.path.join(os.path.curdir, 'IMF', f'{filename}_denorm.npy')
    np.save(file_path, denorm)


def main():
    for file in os.listdir(DATA_DIR):
        filename = os.path.join(os.curdir, DATA_DIR, file)
        env = import_stock_to_env(filename)
        stock_prices = env.env.prices
        getimfs(stock_prices, WINDOW, file[:-4])


if __name__ == '__main__':
    main()
