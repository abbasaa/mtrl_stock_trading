from DQN import DQN
from gym_anytrading.datasets import STOCKS_GOOGL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

prices = STOCKS_GOOGL.loc[:, 'Close'].to_numpy()


def Batch(size):
    time_indices = np.random.randint(0, high=449, size=size)
    last_prices = torch.tensor(prices[time_indices + 1], dtype=torch.float)
    positions = torch.randint(low=0, high=2, size=(size, ), dtype=torch.float)
    return positions, time_indices, last_prices


# pos, last_price, prediction
def Label(net_input):
    label = []
    for i in range(net_input.shape[0]):
        # short, long
        diff = net_input[i, 2] - net_input[i, 1]
        label.append(torch.tensor([-diff, diff], dtype=torch.float))
    return torch.stack(label)


model = DQN(3, 5, 2, 'GOOGL')
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

Batches = 1000

losses = []
model.train()
for i in range(Batches):
    print("Batch: ", i)
    optimizer.zero_grad()
    pos, t_inds, last_p = Batch(32)
    action_v, inputs = model(pos, t_inds, last_p)
    lbls = Label(inputs)
    loss = criterion(action_v, lbls)
    losses.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()

plt.plot([i for i in range(len(losses))], losses)
plt.show()
