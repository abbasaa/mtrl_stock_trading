
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):

    def __init__(self, observation_window, observation_dim, h1, h2, output):
        super(DQN, self).__init__()
        # linear on observations
        self.layer1 = nn.Linear(observation_window, h1)
        # hidden on linear output
        self.layer2 = nn.Linear(observation_dim * h1, h2)
        # hidden on h1 output + position
        self.layer3 = nn.Linear(h2 + 1, output)
        # output size 1

    def forward(self, position, obs):
        # position is N x 1
        # want obs N x obs_dim x obs_window
        obs = obs.permute(0,2,1)
        out1 = F.relu(self.layer1(obs))
        # out1 N x obs_dim x h1
        out1 = out1.flatten(start_dim=1)
        # out1 N x (obs_dim * h1)
        out2 = F.relu(self.layer2(out1))
        # out2 N x h2
        x = torch.cat((position, out2), 1)
        qvals = self.layer3(x)
        return qvals
