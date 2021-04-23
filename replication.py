from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from PricingNet import PricingNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 250
END_TIME = 600
EVAL_END = 700
prices = STOCKS_GOOGL.loc[:, 'Low'].to_numpy()[:EVAL_END]

EPOCHS = 40
BATCH = 64
BATCH_NUM = (END_TIME - WINDOW - 1)//BATCH


training_loss = []
eval_loss = []

def eval_model():
    pe, le = Batch(True)
    PricingNet.eval()
    PricingNet.zero_grad()
    out = PricingNet(pe)
    eloss = criterion(out.squeeze(), torch.tensor(le, dtype=torch.float, device=device)).detach().cpu().numpy()
    eval_loss.append(eloss)
    PricingNet.train()


def Batch(evaluate):
    if evaluate:
        prices_idx = np.arange(END_TIME-WINDOW, EVAL_END-WINDOW-1)
    else:
        prices_idx = np.random.randint(0, high=(END_TIME-WINDOW-1), size=BATCH)
    labels = []
    for p in prices_idx:
        labels.append(prices[p+WINDOW+1])
    return prices_idx, labels


PricingNet = PricingNet("GOOGL", device)
PricingNet.to(device)

optimizer = optim.Adam(PricingNet.parameters())
criterion = nn.MSELoss()

# load checkpoint if possible
PATH = os.path.join(os.curdir, "checkpoints")
epoch_start = 0
while True:
    CURPATH = os.path.join(PATH, f"pricenet_{epoch_start}.pth")
    if not os.path.exists(CURPATH):
        epoch_start = max(0, epoch_start-1)
        break
    epoch_start += 1

if epoch_start != 0:
    print(f"Loading model from checkpoint at epoch: {epoch_start}")
    CURPATH = os.path.join(PATH, f"pricenet_{epoch_start}.pth")
    checkpoint = torch.load(CURPATH)
    PricingNet.load_state_dict(checkpoint['pricingnet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

PricingNet.train()
for j in range(epoch_start, EPOCHS):
    for k in range(BATCH_NUM):
        optimizer.zero_grad()
        inputs, labels = Batch(False)
        PricingNet.zero_grad()
        output = PricingNet(inputs)
        loss = criterion(output.squeeze(), torch.tensor(labels, dtype=torch.float, device=device))
        training_loss.append(loss.detach().cpu().numpy())
        loss.backward(retain_graph=True)
        for param in PricingNet.parameters():
            param.grad.data.clamp(-1, 1)
        optimizer.step()
        print("Batch: ", k)
    print("Epoch: ", j)
    eval_model()

    print(f"Saving checkpoint for Epoch {j} ...")
    torch.save({
        'pricingnet_state_dict': PricingNet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(os.curdir, "checkpoints", f"pricenet_{j}.pth"))

PricingNet.eval()

fig, ax = plt.subplots()
ax.plot(np.arange(len(training_loss)), training_loss, 'r', label='train')
ax.plot(np.arange(len(training_loss), step=BATCH_NUM), eval_loss, 'b', label='eval')
ax.legend()
fig.savefig('TrainEvalPNet.png')

x = [j for j in range(EVAL_END-WINDOW-1)]
with torch.no_grad():
    predicted = PricingNet(x).detach().cpu().numpy()
fig, ax = plt.subplots()
ax.plot(x, predicted, 'r')
ax.plot(x, prices[WINDOW+1:], 'b')
fig.savefig('Prediction.png')