import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from IMFNet import IMFNet

from PyEMD import CEEMDAN

import time

NUM_FILTERS = 512
KERNEL_SIZE = 2
MAX_POOL_SIZE = 2
LSTM_UNITS = 200
N_LAYERS = 1

NUM_IMF = 5

CEEMD = CEEMDAN(DTYPE=np.float16, trials=20)


class PricingNet(nn.Module):

    def __init__(self, prices, window):
        super(PricingNet, self).__init__()
        self.imfNets = []*NUM_IMF
        self.layer = nn.Linear(NUM_IMF, 1)
        self.hn = []*NUM_IMF
        self.cn = []*NUM_IMF
        # TODO init hn cn
        for i in range(NUM_IMF):
            self.imfNets[i] = IMFNet()
        # END_TIME - Window x Num_imfs x window
        self.imfs, self.denorm = getImfs(prices, window)


    def forward(self, prices):
        # prices: N x 1
        input1 = self.getBatch(prices)
        output1 = []*NUM_IMF
        for i in range(NUM_IMF):
            # N x 1
            imf_prediction = self.imfNets[i](input1[i], self.hn[i], self.cn[i])
            for j in range(len(imf_prediction)):
                imf_prediction[j] = denormalize(imf_prediction[j], self.denorm[prices[j]][0], self.denorm[prices[j]][1])
            output1[i] = imf_prediction
        # Num IMFs x N
        output1 = torch.cat(output1)
        prediction = self.layer(output1)
        return prediction

    def h0(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(N_LAYERS, 1, LSTM_UNITS).zero_(),
                  weight.new(N_LAYERS, 1, LSTM_UNITS).zero_())
        return hidden

    def getBatch(self, prices):
        Batch = []*len(prices)
        for i, p in enumerate(prices):
            Batch[i] = self.imfs[p]


        # Batch N x Num_imfs x window
        Batch = np.stack(Batch)
        Batch = np.swapaxes(Batch, 0, 1)
        Batch = Batch[:, :, np.newaxis, :]
        # Num_imfs x N x 1 x window
        return torch.tensor(Batch)


def getImfs(prices, window):
    # Prices: 1 x price_count
    END_TIME = len(prices)
    # Steps = END
    IMFs = [] * (END_TIME - window)
    denorm = [] * (END_TIME - window)
    for i in range(len(IMFs)):
        ceemd_out = CEEMD(prices[i:i+window], max_imf=NUM_IMF-1)
        denorm[i] = list(zip(list(map(min, ceemd_out)), list(map(max, ceemd_out))))
        IMFs[i] = np.array(list(map(normalize, ceemd_out)))
    # IMFs: Steps x Num_imfs x window,
    # Denorm: Steps x Num_imfs x (min, max)
    return IMFs, denorm


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


def denormalize(y, min_x, max_x):
    return y*(max_x - min_x) + min_x


