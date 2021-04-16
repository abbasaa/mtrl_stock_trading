import numpy as np
import torch
import torch.nn.functional as F



def step_SA(policy_net, pos, obs):
    with torch.no_grad():
        return policy_net(pos, obs).max(1)[1].view(1, 1).float(), policy_net(pos, obs)


def accumulator(sector, stock_prices, test_stock_prices, values):
    # stock prices as MxN (M = # of single agent stocks)
    # test stock prices as KxN (K = # of test stocks)
    # sector prices as 1xN
    # HP_SA as Nx2xK
    M = stock_prices.shape[0]
    N = stock_prices.shape[1]
    K = test_stock_prices.shape[0]

    # calculate correlation coefficients for agents
    corr_coef = np.zeros((M,1))
    for i in range(M):
        corr_coef[i,0] = np.corrcoef(stock_prices[i,:], sector)[0,1]
    
    # calculate correlation coefficients for test stocks
    corr_coef_test = np.zeros((K,1))
    for i in range(K):
        corr_coef_test[i,0] = np.corrcoef(test_stock_prices[i,:], sector)[0,1]
    
    # calculate action values for test stocks
    N_values = values.shape[0]
    print(N_values)
    corr_sum = np.zeros((N_values,2,K))
    for i in range(M):
        corr_sum = values[:,:,i,:] * corr_coef[i,0] + corr_sum
        
    HP_A = np.zeros((N_values,2,K))
    for i in range(K):
        HP_A[:,:,i] = corr_coef_test[i,0] * corr_sum[:,:,i]/(np.sum(corr_coef,0))
    return HP_A
