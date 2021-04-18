from anytrading_torch import anytrading_torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from IMFNet import IMFNet
import numpy as np

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW = 250
END_TIME = 700

EPOCHS = 20
BATCH = 32
BATCH_NUM = (END_TIME - WINDOW - 1)//BATCH

imfs = np.load(f'IMF/GOOGL_IMF.npy')
denorm = np.load(f'IMF/GOOGL_denorm.npy')


def Batch():
    prices_idx = np.random.randint(0, high=(END_TIME-WINDOW-1), size=BATCH)
    inputs = [None]*BATCH
    labels = []
    for i, p in enumerate(prices_idx):
        inputs[i] = imfs[p][0]
        labels.append(imfs[p+1][0][-1])
    inputs = np.stack(inputs)
    inputs = inputs[:, np.newaxis, :]
    inputs = torch.tensor(inputs, dtype=torch.float, device=device)
    labels = torch.tensor(labels, dtype=torch.float, device=device)
    return inputs, labels, prices_idx


def denormalize(output, start_times):
    mins = denorm[start_times, 0, 0]
    differences = denorm[start_times, 0, 1] - mins
    return output*torch.tensor(differences, dtype=torch.float) + torch.tensor(mins, dtype=torch.float)


losses = []
model = IMFNet()
model.to(device)

optimizer = optim.RMSprop(model.parameters())
criterion = nn.MSELoss()

x = imfs[:END_TIME - WINDOW - 1, 0]
x = x[:, np.newaxis, :]
x = torch.tensor(x, device=device, dtype=torch.float)
correct = torch.tensor([imfs[p + 1][0][-1] for p in range(END_TIME - WINDOW - 1)], dtype=torch.float, device=device)


def eval(doplot):
    predicted = model(x).squeeze()
    predicted = denormalize(predicted, [j for j in range(END_TIME - WINDOW - 1)])
    l = criterion(predicted, correct).detach().numpy()
    losses.append(l)
    if doplot:
        plt.plot([j for j in range(END_TIME - WINDOW - 1)], predicted.detach().numpy(), 'r')
        plt.plot([j for j in range(END_TIME - WINDOW - 1)], correct.detach().numpy(), 'b')
        plt.show()


model.train()
for i in range(EPOCHS):
    for j in range(BATCH_NUM):
        optimizer.zero_grad()
        inp, lab, price_idx = Batch()
        model.zero_grad()
        out = model(inp).squeeze()
        predictions = denormalize(out, price_idx)
        loss = criterion(predictions, lab)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Batch: ", j)
    print("Epoch: ", i)
    model.eval()
    eval(i + 1 == EPOCHS)
    model.train()


plt.clf()
plt.plot([i for i in range(len(losses))], losses)
plt.show()

