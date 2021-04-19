import torch
import torch.nn as nn
import numpy as np
from IMFNet import IMFNet


NUM_FILTERS = 512
KERNEL_SIZE = 2
MAX_POOL_SIZE = 2
LSTM_UNITS = 200
N_LAYERS = 1

NUM_IMF = 5


class PricingNet(nn.Module):

    def __init__(self, ticker, batch_size):
        super(PricingNet, self).__init__()
        self.layer = nn.Linear(NUM_IMF, 1)
        self.batch_size = batch_size
        self.ticker = ticker
        imfList = []
        for i in range(NUM_IMF):
            imfList.append(IMFNet())
        self.imfNets = nn.ModuleList(imfList)
        # END_TIME - Window x Num_imfs x window
        self.imfs = np.load(f'IMF/{ticker}_IMF.npy')
        self.denorm = np.load(f'IMF/{ticker}_denorm.npy')

    def forward(self, prices):
        # prices: N x 1
        input1 = self.getBatch(prices)
        output1 = []
        for i in range(NUM_IMF):
            # N x 1
            imf_prediction = self.imfNets[i](input1[i])
            # TODO speed up
            for j in range(len(imf_prediction)):
                 imf_prediction[j] = denormalize(imf_prediction[j], self.denorm[prices[j]][i][0], self.denorm[prices[j]][i][1])
            output1.append(imf_prediction)
        # Num IMFs x N
        output1 = torch.stack(output1).squeeze()
        output1 = torch.transpose(output1, 0, 1)
        prediction = self.layer(output1)
        return prediction

    def getBatch(self, prices):
        Batch = []
        for p in prices:
            Batch.append(self.imfs[p])
        # Batch: N x Num_imfs x window
        Batch = np.stack(Batch)
        Batch = np.swapaxes(Batch, 0, 1)
        Batch = Batch[:, :, np.newaxis, :]
        # Num_imfs x N x 1 x window
        return torch.tensor(Batch, dtype=torch.float)


def denormalize(y, min_x, max_x):
    return y*(max_x - min_x) + min_x


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

