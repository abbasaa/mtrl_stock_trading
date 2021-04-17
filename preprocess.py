import numpy as np
import gym
import gym_anytrading
from PyEMD import CEEMDAN

CEEMD = CEEMDAN(DTYPE=np.float16, trials=20)
NUM_IMF = 5


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


def getImfs(prices, window):
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
    np.save('IMF/GOOGL_IMF.npy', IMFs)
    np.save('IMF/GOOGL_denorm.npy', denorm)


WINDOW = 250
END_TIME = 700
env = gym.make('stocks-v0', frame_bound=(WINDOW, END_TIME), window_size=WINDOW)
google_prices = env.prices

getImfs(google_prices, WINDOW)
