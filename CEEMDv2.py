import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_anytrading
import emd
from gym_anytrading.datasets import STOCKS_GOOGL

custom_env = gym.make('stocks-v0',
                      df=STOCKS_GOOGL,
                      window_size=10,
                      frame_bound=(10, 300))

s = custom_env.prices

t = np.arange(0, 300)
imfs = emd.sift.complete_ensemble_sift(s, max_imfs=4)
imfs = imfs[0].T
i = len(imfs)
print("Imfs: ", i)
fig, axs = plt.subplots(i + 1, 1)
axs[0].plot(t, s)
for j in range(i):
    axs[j+1].plot(t, imfs[j])
fig.tight_layout()
plt.show()