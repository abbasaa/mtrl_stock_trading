from anytrading_torch import anytrading_torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PricingNet import PricingNet
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 250
END_TIME = 700
env = anytrading_torch(device, 'stocks-v0', (WINDOW, END_TIME), WINDOW)
prices = env.prices()

EPOCHS = 20
BATCH = 32
BATCH_NUM = (END_TIME - WINDOW - 1)//BATCH


def Batch():
    prices_idx = np.random.randint(0, high=(END_TIME-WINDOW-1), size=BATCH)
    labels = []
    for p in prices_idx:
        labels.append(prices[p+WINDOW+1])
    return prices_idx, labels


PricingNet = PricingNet("GOOGL", BATCH)
PricingNet.to(device)

optimizer = optim.RMSprop(PricingNet.parameters())
criterion = nn.MSELoss()

PricingNet.train()
for j in range(EPOCHS):
    for k in range(BATCH_NUM):
        optimizer.zero_grad()
        inputs, labels = Batch()
        PricingNet.zero_grad()
        output = PricingNet(inputs)
        loss = criterion(output.squeeze(), torch.tensor(labels, dtype=torch.float, device=device))
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Batch: ", k)
    print("Epoch: ", j)

PricingNet.eval()

x = [j for j in range(END_TIME-WINDOW-1)]
predicted = PricingNet(x)
#ERROR HERE
plt.plot(x, predicted.detach().numpy(), 'r')
plt.plot(x, prices[WINDOW+1:], 'b')
plt.show()