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

    def __init__(self, ticker, device):
        super(PricingNet, self).__init__()
        self.layer = nn.Linear(NUM_IMF, 1)
        self.ticker = ticker
        self.device = device
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
            imf_prediction = self.imfNets[i](input1[i]).squeeze(dim=1)
            output1.append(self.denormalize(imf_prediction, prices, i))
        # Num IMFs x N
        output1 = torch.stack(output1)
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
        return torch.tensor(Batch, dtype=torch.float, device=self.device)

    def denormalize(self, output, start_times, imf):
        mins = self.denorm[start_times, imf, 0]
        differences = self.denorm[start_times, imf, 1] - mins
        return output * torch.tensor(differences, dtype=torch.float, device=self.device) + torch.tensor(mins, dtype=torch.float, device=self.device)


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

