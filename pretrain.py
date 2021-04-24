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

from gym_anytrading.datasets import STOCKS_GOOGL

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

# make model folders
models_dir = os.path.join(os.curdir, 'models')
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)
models_dir = os.path.join(models_dir, TICKER)
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

EPOCH_START = 0
# makes checkpoint folders
checkpoints_dir = os.path.join(os.curdir, 'checkpoints')
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoints_dir = os.path.join(checkpoints_dir, 'pricingnet')
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoints_dir = os.path.join(checkpoints_dir, TICKER)
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoint_files = [f for f in os.listdir(checkpoints_dir) if os.path.isfile(os.path.join(checkpoints_dir, f))]
if len(checkpoint_files) != 0:
    max_epoch = -2 ** 32
    for file in checkpoint_files:
        cur_epoch = int(file.split('.')[1])
        if cur_epoch > max_epoch:
            max_epoch = cur_epoch
    EPOCH_START = max_epoch
    print(f'Loading pricingnet from checkpoint at epoch: {max_epoch}')
    checkpoint_file = os.path.join(checkpoints_dir, f'pricingnet.{max_epoch}.pth')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['pricingnet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if len(checkpoint_files) > 1:
        print('Removing older checkpoint files')
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if (os.path.isfile(os.path.join(checkpoints_dir, f))
                                                                       and f != f'pricingnet.{max_epoch}.pth')]
        for f in checkpoint_files:
            os.remove(os.path.join(checkpoints_dir, f))

BATCH = 64
K_folds = 5
EPOCHS = 200
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


model.train()
for i in range(EPOCH_START, EPOCHS):
    print("Epoch ", i, ": [", end='')
    batch = np.arange(END_TIME - WINDOW - 1)
    np.random.shuffle(batch)
    t_loss = 0
    for j in range(BATCH_NUM-1):
        optimizer.zero_grad()
        inputs = batch[j*BATCH:(j+1)*BATCH]
        labels = get_labels(inputs)
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), torch.tensor(labels, dtype=torch.float, device=device))
        t_loss += loss.detach().cpu().numpy()
        loss.backward(retain_graph=True)
        for param in model.parameters():
            param.grad.data.clamp(-1, 1)
        optimizer.step()
        print("=", end='', flush=True)
    t_loss /= BATCH_NUM - 1
    training_loss.append(t_loss)
    print("]")


    # Eval
    model.eval()
    model.zero_grad()
    optimizer.zero_grad()
    inputs = batch[(BATCH_NUM - 1)*BATCH:]
    labels = get_labels(inputs)
    with torch.no_grad():
        output = model(inputs)
    eloss = criterion(output.squeeze(), torch.tensor(labels, dtype=torch.float, device=device)).detach().cpu().numpy()
    eval_loss.append(eloss)

    #inputs = np.arange(END_TIME - WINDOW - 1)
    #labels = get_labels(inputs)
    #with torch.no_grad():
    #    output = model(inputs)
    #total_loss = criterion(output.squeeze(), torch.tensor(labels, dtype=torch.float, device=device)).detach().cpu().numpy()
    if eloss < lowest_loss:
        lowest_loss = eloss
        print('Saving Best Model ...')
        torch.save({
            'pricingnet_state_dict': model.state_dict(),
        }, os.path.join(models_dir, f'pricingnet.pth'))
    model.train()

    # save checkpoint
    print(f"Saving checkpoint for Epoch {i} ...")
    torch.save({
        'pricingnet_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoints_dir, f"pricingnet.{i}.pth"))

    # remove last checkpoint
    if i-1 >= 0 and os.path.isfile(os.path.join(checkpoints_dir, f'pricingnet.{i-1}.pth')):
        print(f'Removing old checkpoint file for epoch {i-1}')
        os.remove(os.path.join(checkpoints_dir, f'pricingnet.{i-1}.pth'))

best_model_path = os.path.join(os.curdir, 'models', TICKER, f'pricingnet.pth')
model.load_state_dict(torch.load(best_model_path)['pricingnet_state_dict'])
model.eval()

# Plot Loss
fig, ax = plt.subplots()
ax.plot(np.arange(len(training_loss)), training_loss, 'r', label='train')
ax.plot(np.arange(len(eval_loss)), eval_loss, 'b', label='eval')
ax.set_title(f'Train and Eval Loss vs Epoch of {TICKER} with {EPOCHS} epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig(f'models/{TICKER}/PricingLoss.png')

# Plot Pricing
x = [j for j in range(END_TIME-WINDOW)]
with torch.no_grad():
    predicted = model(x).detach().cpu().numpy()
fig, ax = plt.subplots()
ax.plot(x, predicted, 'r', label='predicted')
ax.plot(x, stock_prices[WINDOW+1:], 'b', label='actual')
ax.set_title(f'Predicted vs Actual Stock Price of {TICKER} with {EPOCHS} epochs')
ax.set_ylabel('Price (Dollars)')
ax.set_xlabel('Time (Days)')
ax.legend()
fig.savefig(f'models/{TICKER}/PricingPrediction.png')
