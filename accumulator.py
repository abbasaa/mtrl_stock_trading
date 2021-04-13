import numpy as np
import torch
import torch.nn.functional as F



def step_SA(policy_net, pos, obs):
    with torch.no_grad():
        return policy_net(pos, obs).max(1)[1].view(1, 1).float(), policy_net(pos, obs)


def accumulator(sector, stock_prices, values):
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
    # calculate average action values
    N_values = values.shape[0]
    corr_sum = np.zeros((N_values,2))
    for i in range(M):
        corr_sum = values[:,:,i] * corr_coef[i,0] + corr_sum
        
    
    HP_A = corr_sum/(np.sum(corr_coef,0))
    return HP_A
