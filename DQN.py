import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PricingNet import PricingNet


class DQN(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size, ticker, device):
        super(DQN, self).__init__()
        self.pricing = PricingNet(ticker, device)
        for param in self.pricing.parameters():
            param.requires_grad = False
        # load in pretrained pricingnet
        models_dir = os.path.join(os.curdir, 'models', ticker)
        if not os.path.isdir(models_dir):
            raise Exception('No models dir to load pretrained pricingnet')
        if not os.path.isfile(os.path.join(models_dir, 'pricingnet.pth')):
            raise Exception('Pretrained pricingnet.pth does not exist')
        print(f'Loading pricingnet from models dir')
        model_file = os.path.join(models_dir, 'pricingnet.pth')
        checkpoint = torch.load(model_file)
        self.pricing.load_state_dict(checkpoint['pricingnet_state_dict'])

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        # linear on pricing net and position
        self.layer1 = nn.Linear(input_size, hidden_dim)
        # hidden on linear output
        self.layer2 = nn.Linear(hidden_dim, output_size)
        # output size: action space

    def forward(self, position, time_idx, last_price):
        prediction = self.pricing(time_idx)
        input1 = torch.cat((position.transpose(0, 1), last_price.transpose(0, 1), prediction.transpose(0, 1)))
        input1 = torch.transpose(input1, 0, 1)
        out1 = F.relu(self.layer1(input1))
        qvals = self.layer2(out1)
        return qvals

