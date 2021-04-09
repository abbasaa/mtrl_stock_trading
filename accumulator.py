#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accumulator(sector, stock_prices, HP_SA):
    # stock prices as MxN (M = # of single agent stocks)
    # sector prices as 1xN
    # HP_SA as MxX
    M = stock_prices.shape(0)
    N = stock_prices.shape(1)

    # calculate correlation coefficients
    corr_coef = np.zeros((M,1))
    for i in range(M):
        corr_coef[i,0] = np.corrcoef(stock_prices[i,:], sector)
    
    # calculate hyperparameters
    HP_A = np.dot(corr_coef.T, HP_SA)/(N * np.sum(corr_coef))
    
    return HP_A

