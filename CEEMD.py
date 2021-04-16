from PyEMD import EMD, CEEMDAN, EEMD
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_anytrading
from IMFNet import PricingNet, normalize, denormalize
from gym_anytrading.datasets import STOCKS_GOOGL
import torch

END = 250

custom_env = gym.make('stocks-v0',
                      df=STOCKS_GOOGL,
                      window_size=10,
                      frame_bound=(10, END))

s = custom_env.prices
c = CEEMDAN()
IMFs = c(s)

t = np.arange(0, END)
i = len(IMFs)

if i < 5:
    IMFs = np.concatenate((np.zeros((5-i, END)), IMFs))
elif i > 5:
    residual = np.sum(IMFs[4:,:], axis=0)
    IMFs[4, :] = residual
    IMFs = IMFs[:5, :]

print("Imfs: ", i)
"""
fig, axs = plt.subplots(6, 1)
axs[0].plot(t, s)
for j in range(5):
    axs[j+1].plot(t, IMFs[j])
fig.tight_layout()
plt.show()
"""

first = PricingNet()
y, min_x, max_x = normalize(IMFs[0])

input = torch.tensor(IMFs[0][np.newaxis, np.newaxis, :], dtype=torch.float)
testpredict, h = first(input, first.h0)

x = denormalize(testpredict, min_x, max_x)

print(testpredict)
