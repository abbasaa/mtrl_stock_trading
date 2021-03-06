import os
import time
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch


from anytrading_torch import anytrading_torch
from PricingNet import PricingNet
from preprocess import getimfs

start = time.perf_counter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TICKER = sys.argv[1]
DATA_DIR = 'Tech'
WINDOW = 250
END_TIME = 754

# CHECK DIR FOR FILE IF NOT THROW ERROR/RUN PREPROCESS
imf_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_IMF.npy')
denorm_filename = os.path.join(os.curdir, 'IMF', f'{TICKER}_denorm.npy')
data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
stock_prices = pd.read_csv(data_file)

if not os.path.isfile(imf_filename) or not os.path.isfile(denorm_filename):
    print(f'IMF or denorm file missing for stock: {TICKER}')
    getimfs(stock_prices, WINDOW, data_file[:-4])
    print(f'Preprocessing for stock: {TICKER} complete ...')

# Prepare Training and Evaluation Environments
env = anytrading_torch(device, 'stocks-v0', stock_prices, (WINDOW, END_TIME), WINDOW)
model = PricingNet(TICKER, device)
model = model.to(device)
model.eval()


def select_action(position, time_idx, prices):
    with torch.no_grad():
        predicted = model(time_idx)
    if predicted >= max(prices):
        return 1
    elif predicted <= min(prices):
        return 0
    else:
        return position


model.load_state_dict(torch.load(os.path.join('models', TICKER, 'pricingnet.pth'))['pricingnet_state_dict'])
obs = env.reset()
pos = torch.zeros((1, 1), dtype=torch.float, device=device)
t_step = 0
while True:
    t_step += 1
    with torch.no_grad():
        act = select_action(pos, [t_step], obs[:, -20:, 0])
    obs, _, is_done, inf = env.step(act)
    pos = act
    if is_done:
        break
env.render_all()
plt.title(f"ENV")
plt.xlabel('Time (Days)')
plt.ylabel('Prices')
plt.show()
