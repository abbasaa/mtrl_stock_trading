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
        checkpoints_dir = os.path.join(os.curdir, 'checkpoints', 'pricingnet', ticker)
        if not os.path.isdir(checkpoints_dir):
            raise Exception('No checkpoints dir to load pretrained pricingnet')
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if os.path.isfile(os.path.join(checkpoints_dir, f))]
        if len(checkpoint_files) == 0:
            raise Exception('No checkpoint files to load pretrained pricingnet')

        max_epoch = -2**32
        for file in checkpoint_files:
            cur_epoch = int(file.split('.')[1])
            if cur_epoch > max_epoch:
                max_epoch = cur_epoch

        print(f'Loading pricingnet from checkpoint at epoch: {max_epoch}')
        checkpoint_file = os.path.join(checkpoints_dir, f'pricingnet.{max_epoch}.pth')
        checkpoint = torch.load(checkpoint_file)
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
        prediction = self.pricing(time_idx).squeeze(dim=1)
        input1 = torch.stack((position.squeeze(dim=-1), last_price, prediction))
        input1 = torch.transpose(input1, 0, 1)
        out1 = F.relu(self.layer1(input1))
        qvals = self.layer2(out1)
        return qvals

