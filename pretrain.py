import time
import sys
import os

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import getimfs
from PricingNet import PricingNet

start = time.perf_counter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TICKER = sys.argv[1]
DATA_DIR = 'Tech'
WINDOW = 250
END_TIME = 754

model = PricingNet(TICKER, device)
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# CHECK DIR FOR FILE IF NOT THROW ERROR/RUN PREPROCESS
imf_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_IMF.npy')
denorm_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_denorm.npy')
data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
stock_prices = pd.read_csv(data_file).loc[:, 'Close'].to_numpy()

if not os.path.isfile(imf_filename) or not os.path.isfile(denorm_filename):
    print(f'IMF or denorm file missing for stock: {TICKER}')
    getimfs(stock_prices, WINDOW, data_file[:-4])
    print(f'Preprocessing for stock: {TICKER} complete ...')

EPOCH_START = 0
if not os.path.isdir(os.path.join(os.curdir, 'checkpoints', TICKER)):
    os.mkdir(os.path.join(os.curdir, 'checkpoints', TICKER))
if not os.path.isdir(os.path.join(os.curdir, 'models', TICKER)):
    os.mkdir(os.path.join(os.curdir, 'models', TICKER))
while True:
    checkpoint_path = os.path.join(os.curdir, 'checkpoints', TICKER, f'pricingnet_{EPOCH_START}.pth')
    if not os.path.isfile(checkpoint_path):
        EPOCH_START = max(0, EPOCH_START-1)
        break
    EPOCH_START += 1

if EPOCH_START != 0:
    print(f'Loading model from checkpoint at episode: {EPOCH_START}')
    checkpoint_path = os.path.join(os.curdir, 'checkpoints', TICKER, f'pricingnet_{EPOCH_START}.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['pricing_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

BATCH = 64
K_folds = 5
EPOCHS = 41
BATCH_NUM = (END_TIME - WINDOW - 1)//BATCH
EVAL_SIZE = (END_TIME - WINDOW - 1)//K_folds
EVAL_INTERVAL = 4

training_loss = []
eval_loss = []
lowest_loss = float('inf')


def get_labels(prices_idx):
    actual_prices = []
    for p in prices_idx:
        actual_prices.append(stock_prices[p + WINDOW + 1])
    return actual_prices


def eval_model():
    eval_arr = np.arange(END_TIME - WINDOW - 1)
    np.random.shuffle(eval_arr)
    model.eval()
    model.zero_grad()
    eloss = 0
    for k in range(K_folds):
        pe = eval_arr[k*EVAL_SIZE:(k+1)*EVAL_SIZE]
        le = get_labels(pe)
        with torch.no_grad():
            out = model(pe)
        eloss += criterion(out.squeeze(), torch.tensor(le, dtype=torch.float, device=device)).detach().cpu().numpy()
    eloss /= K_folds
    eval_loss.append(eloss)
    model.train()
    if eloss < lowest_loss:
        print('Saving Best Model ...')
        torch.save({
            'pricing_state_dict': model.state_dict(),
        }, os.path.join(os.curdir, 'models', TICKER, f'pricingnet.pth'))


model.train()
for i in range(EPOCH_START, EPOCHS):
    print("Epoch ", i, ": [", end='')
    batch = np.arange(END_TIME - WINDOW - 1)
    for j in range(BATCH_NUM):
        optimizer.zero_grad()
        inputs = batch[j*BATCH:(j+1)*BATCH]
        labels = get_labels(inputs)
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), torch.tensor(labels, dtype=torch.float, device=device))
        training_loss.append(loss.detach().cpu().numpy())
        loss.backward(retain_graph=True)
        print("=", end='', flush=True)
    print("]")
    if i % EVAL_INTERVAL == 0:
        eval_model()

    print(f"Saving checkpoint for Epoch {i} ...")
    torch.save({
        'pricing_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(os.curdir, "checkpoints",  TICKER, f"pricingnet_{i}.pth"))

best_model_path = os.path.join(os.curdir, 'models', TICKER, f'pricingnet.pth')
model.load_state_dict(torch.load(best_model_path)['pricing_state_dict'])
model.eval()

# Plot Loss
fig, ax = plt.subplots()
ax.plot(np.arange(len(training_loss)), training_loss, 'r', label='train')
ax.plot(np.arange(len(training_loss), step=BATCH_NUM*EVAL_INTERVAL), eval_loss, 'b', label='eval')
ax.legend()
fig.savefig(f'models/{TICKER}/PricingLoss.png')

# Plot Pricing
x = [j for j in range(END_TIME-WINDOW)]
with torch.no_grad():
    predicted = model(x).detach().cpu().numpy()
fig, ax = plt.subplots()
ax.plot(x, predicted, 'r')
ax.plot(x, stock_prices[WINDOW+1:], 'b')
fig.savefig(f'models/{TICKER}/PricingPrediction.png')
