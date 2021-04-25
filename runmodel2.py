
from DQN import DQN
from anytrading_torch import anytrading_torch
import torch
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TICKER = 'AAPL'
GAMMA = 0.1
TYPE = None

# try:
#     TICKER = sys.argv[1]
#     GAMMA = sys.argv[2]
#     TYPE = 'reward' if sys.argv[3] == 1 else 'profit'
# except:
#     print("Usage: python runmodel.py [TICKER] [GAMMA] [1 for reward | 0 for profit]")
#     exit(-1)

DATA_DIR = 'Tech'
WINDOW = 250
END_TIME = 754

data_file = os.path.join(os.curdir, DATA_DIR, f'{TICKER}.csv')
stock_prices = pd.read_csv(data_file)


env = anytrading_torch(device, 'stocks-v0', stock_prices, (WINDOW, END_TIME), WINDOW)

N_ACTIONS = env.action_space.n
HIDDEN_DIM = 5
N_HISTORIC_PRICES = 1
PolicyNet = DQN(N_HISTORIC_PRICES+2, HIDDEN_DIM, N_ACTIONS, TICKER, device)
PolicyNet = PolicyNet.to(device)

PolicyNet.load_state_dict(torch.load(os.path.join('checkpoints', 'dqn','INTC', CHANGE THISSSSSS))['dqn_state_dict'])
models_dir = os.path.join(os.curdir, 'models', TICKER)
if not os.path.isdir(models_dir):
    raise Exception('No models dir to load pretrained pricingnet')
if not os.path.isfile(os.path.join(models_dir, 'pricingnet.pth')):
    raise Exception('Pretrained pricingnet.pth does not exist')
print(f'Loading pricingnet from models dir')
model_file = os.path.join(models_dir, 'pricingnet.pth')
checkpoint = torch.load(model_file)
PolicyNet.pricing.load_state_dict(checkpoint['pricingnet_state_dict'])


obs = env.reset()
pos = torch.zeros((1, 1), dtype=torch.float, device=device)
t_step = -1
while True:
    t_step += 1
    with torch.no_grad():
        act = PolicyNet(pos, [t_step], obs[:, -1, 0]).max(1)[1].view(1, 1).float()
    obs, _, is_done, inf = env.step(act)
    pos = act
    if is_done:
        break
env.render_all()
plt.title(f"DQN")
plt.savefig(os.path.join('models', f'{TICKER}\CUSTOMEnvironment_{GAMMA}.{TICKER}.png'))
