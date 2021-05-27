# -*- coding: utf-8 -*-
"""
Created on Sun May 23 09:18:25 2021

@author: klowe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns

def main():
    
    # Set constants
    BUYER_WEIBULL = (1.5, 5, 10) # (shape, scale, max)
    SELLER_WEIBULL = (5, 7, 10) # (shape, scale, max)
    
    FEE_LEVELS_B = np.arange(0,10,.2)
    FEE_LEVELS_S = np.arange(0,12,.2)
    
    # Get the falloff function for the various fee levels
    buyer_mean_by_fee = GetFalloff(FEE_LEVELS_B, BUYER_WEIBULL[0], BUYER_WEIBULL[1], (0, BUYER_WEIBULL[2]))
    seller_mean_by_fee = GetFalloff(FEE_LEVELS_S, SELLER_WEIBULL[0], SELLER_WEIBULL[1], (0, SELLER_WEIBULL[2]))

    # Now we're going to do a double loop
    total_ev = [[] for i in range(len(FEE_LEVELS_B))]
    for ib in range(len(FEE_LEVELS_B)):
        buy_ev = GetSales(buyer_mean_by_fee[ib], FEE_LEVELS_B[ib])
        for ii in range(len(FEE_LEVELS_S)):
            sell_ev = GetSales(seller_mean_by_fee[ii], FEE_LEVELS_S[ii])
            total_ev[ib].append(buy_ev+sell_ev)
    total_ev = np.array(total_ev)
    
    plt.figure()
    plt.plot(FEE_LEVELS_B, buyer_mean_by_fee,label='Buyer',color=(.2, .2, .8))
    plt.plot(FEE_LEVELS_S, seller_mean_by_fee,label='Seller',color=(.8, .2, .2))
    plt.xlabel('Fee amount')
    plt.ylabel('Purchases per user')
    plt.legend()
    plt.show()
        
    plt.figure()
    sns.heatmap(total_ev, \
                yticklabels=['%.2f' % FEE_LEVELS_B[i] if FEE_LEVELS_B[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_B))], \
                    xticklabels=['%.2f' % FEE_LEVELS_S[i] if FEE_LEVELS_S[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_S))], \
                        cbar_kws={'label': 'Expected revenue per buyer/seller pair'})
    plt.xlabel('Seller Fee')
    plt.ylabel('Buyer Fee')
    plt.show()
    

def GetFalloff(vals, shape, scale, quant_range=(0,1)):
    # Define the weibull    
    weib_fun = lambda x, k, l: 1 - np.exp(-1*((x/l)**k))
    weib_vals = weib_fun(vals, shape, scale)
    
    # Scale the values to cover the range
    weib_vals = weib_vals * (quant_range[1]-quant_range[0])
    
    # And subtract from max to get falloff
    weib_out = quant_range[1]-weib_vals
    
    return weib_out


def GetSales(poiss_mean, this_fee):
    
    SALES_CHECK = np.arange(0,20)
    
    # Buyer and seller means will index into Poisson PMF at SALES_CHECK numbers of sales
    n_sales = poisson.pmf(SALES_CHECK,poiss_mean)
    ev_sales = sum(n_sales*(SALES_CHECK*this_fee))

    return ev_sales

        
if __name__=='__main__':
    main()    

