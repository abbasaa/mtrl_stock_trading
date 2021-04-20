import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PricingNet import PricingNet


class DQN(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size, ticker):
        super(DQN, self).__init__()
        self.pricing = PricingNet(ticker)
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        # linear on pricing net and position
        self.layer1 = nn.Linear(input_size, hidden_dim)
        # hidden on linear output
        self.layer2 = nn.Linear(hidden_dim, output_size)
        # output size: action space

    def forward(self, position, time_idx, last_price):
        prediction = self.pricing(time_idx).squeeze(dim=1)
        input1 = torch.stack((position.squeeze(dim=-1), last_price, prediction))
        input1 = torch.transpose(input1, 0, 1)
        out1 = F.relu(self.layer1(input1))
        qvals = self.layer2(out1)
        return qvals

