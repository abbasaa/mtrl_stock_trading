import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_FILTERS = 512
KERNEL_SIZE = 2
MAX_POOL_SIZE = 2
LSTM_UNITS = 200
N_LAYERS = 1


class IMFNet(nn.Module):

    def __init__(self):
        super(IMFNet, self).__init__()
        self.conv = nn.Conv1d(1, NUM_FILTERS, KERNEL_SIZE)
        self.pool = nn.MaxPool1d(MAX_POOL_SIZE)
        self.lstm = nn.LSTM(NUM_FILTERS, LSTM_UNITS, N_LAYERS)
        self.layer = nn.Linear(in_features=LSTM_UNITS, out_features=1)

    def forward(self, imf):
        # N x 1 x Window -> N * 512 * Window -1
        # kernel size = 2
        out1 = F.relu(self.conv(imf))
        # N x 512 x Window -> N x 512 x (Window-1)/2
        out2 = self.pool(out1)
        out2 = out2.permute(2, 0, 1)
        out3, (_, _) = self.lstm(out2, self.h0(imf.shape[0]))
        # (Window - 1) / 2 x N x 512 -> (Window - 1) / 2 x N x 200
        prediction = self.layer(out3[-1])
        # Pred: 1d
        return prediction

    def h0(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(N_LAYERS, batch_size, LSTM_UNITS).zero_(),
                  weight.new(N_LAYERS, batch_size, LSTM_UNITS).zero_())
        return hidden
