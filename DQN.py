
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from IMFNet import PricingNet

class DQN(nn.Module):

    def __init__(self, input, hidden, output):
        super(DQN, self).__init__()
        self.pricing = PricingNet()
        # linear on observations
        self.layer1 = nn.Linear(input, hidden)
        # hidden on linear output
        self.layer2 = nn.Linear(hidden, output)
        # output size 2

        # init pricing net for each imf
        # init h0's
        # add dense layer at end to accumulate their outputs
        # self.imfnet1 = PricingNet()
        # self.h0_1 = self.imfnet1.h0()
        # self.imfnet2 = PricingNet()
        # self.h0_2 = self.imfnet2.h0()
        # self.imfnet3 = PricingNet()
        # self.h0_3 = self.imfnet3.h0()
        # self.imfnet4 = PricingNet()
        # self.h0_4 = self.imfnet4.h0()
        # self.imfnet5 = PricingNet()
        # self.h0_5 = self.imfnet5.h0()
        # self.layer = nn.Linear()  # fill in feature size



    def forward(self, position, prices):
        prediction = self.pricing(prices)
        input1 = torch.cat((position, prices[-1], prediction))
        out1 = F.relu(self.layer1(input1))
        qvals = self.layer2(out1)
        return qvals

        # position is N x 1
        # want obs N x obs_dim x obs_window
        # obs = obs.permute(0,2,1)
        # out1 = F.relu(self.layer1(obs))
        # out1 N x obs_dim x h1
        # out1 = out1.flatten(start_dim=1)
        # out1 N x (obs_dim * h1)
        # out2 = F.relu(self.layer2(out1))
        # out2 N x h2
        # x = torch.cat((position, out2), 1)
        # qvals = self.layer3(x)
        # return qvals

        # generating imfs in forward pass or in preprocessing ?
        # out1 = self.imfnet1.forward(imfs[0], self.h0_1)
        # out2 = self.imfnet2.forward(imfs[1], self.h0_2)
        # out3 = self.imfnet3.forward(imfs[2], self.h0_3)
        # out4 = self.imfnet4.forward(imfs[3], self.h0_4)
        # out5 = self.imfnet5.forward(imfs[4], self.h0_5)
