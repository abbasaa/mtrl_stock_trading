import numpy as np
import torch
import torch.nn.functional as F



def step_SA(policy_net, pos, obs):
    with torch.no_grad():
        return policy_net(pos, obs).max(1)[1].view(1, 1).float()


def accumulator(sector, stock_prices, actions):
    # stock prices as MxN (M = # of single agent stocks)
    # sector prices as 1xN
    # HP_SA as MxX
    M = stock_prices.shape[0]
    N = stock_prices.shape[1]

    # calculate correlation coefficients
    corr_coef = np.zeros((M,1))
    for i in range(M):
        corr_coef[i,0] = np.corrcoef(stock_prices[i,:], sector)[0,1]
    
    #print(corr_coef)
    # calculate actions
    actions_np = actions.numpy()
    #print(actions_np)
    N_actions = actions_np.shape[1]
    corr_sum = np.zeros((1,N_actions))
    for i in range(M):
        #print('current corr_sum')
        #print(corr_sum)
        corr_sum = actions_np[i,:] * corr_coef[i,0] + corr_sum
        #print('actions*corr_coef')
        #print(actions_np[i,:] * corr_coef[i,0])
        #print('new corr_sum')
        #print(corr_sum)
        
    
    HP_A = corr_sum/(np.sum(corr_coef,0))
    return HP_A
