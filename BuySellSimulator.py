import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool
from scipy.stats import poisson


def main():
    
    # Set globals
    global BUYER_RETAIN, SELLER_RETAIN, BUYER_PBUY, SELLER_PBUY
    
    # Set a couple flags for different variants
    DO_IMBALANCED = True
    DO_SAME_WEIB = False
    SAVE_FIG = False
    
    # Set parameters for Weibull function to illustrate falloff with increased fees.
    # We'll use two sets: one for "retention" (i.e., how many transactions does an individual make) and "P(transact)", how likely an inidvidual is to make a given transaction
    # Here is the retention parameterization
    BUYER_RETAIN = (1.5, 7, (0,10)) # (shape, scale, max)
    SELLER_RETAIN = (5, 7, (0,10)) # (shape, scale, max)
    
    # For simplicity on this pass, assume same shape for P(buy)
    if DO_SAME_WEIB:
        BUYER_PBUY = (1.5, 7, (0, 1)) # 7, 10
        SELLER_PBUY = (5, 7, (0, 1)) # 3, 10
    else:
        BUYER_PBUY = (7, 10, (.5, 1)) # 7, 10
        SELLER_PBUY = (3, 10, (.25, 1)) # 3, 10
    
    # Set values for fee testing
    FEE_STEP = .2
    B_FEE_MIN = 0
    B_FEE_MAX = 10
    S_FEE_MIN = 0
    S_FEE_MAX = 12
    
    # Number of buyers/sellers to simulate
    if DO_IMBALANCED:
        N_BUYERS = 100
        N_SELLERS = 100
    else:
        N_BUYERS = 1000 
        N_SELLERS = 1000
    
    # And how many transactions to simulate 
    N_TRANSACT = 1000
    
    # Set range of fee levels
    FEE_LEVELS_B = np.arange(B_FEE_MIN,B_FEE_MAX+FEE_STEP,FEE_STEP)
    FEE_LEVELS_S = np.arange(S_FEE_MIN,S_FEE_MAX+FEE_STEP,FEE_STEP)
    
    # Set random seeds
    random.seed(0)
    np.random.seed(0)
    
    # Parallelization parameters
    CORE_PROP = .9
    DO_PARALLEL = True
    
    if DO_PARALLEL:
        # Get number of cores
        try:
            n_cores = len(os.sched_getaffinity(0))
        except:
            import psutil
            n_cores = psutil.cpu_count()
        with Pool(int(n_cores*CORE_PROP)) as p:
            par_vals = p.map(PoolHelper, [[vb, N_TRANSACT, N_BUYERS, N_SELLERS, FEE_LEVELS_S] for ib, vb in enumerate(FEE_LEVELS_B)])
            total_fees = [par_vals[i][0] for i in range(len(par_vals))]
            perc_accepted = [par_vals[i][1] for i in range(len(par_vals))]
    else:
        # Initialize output matrix and start fee loops
        total_fees = [[] for i in range(len(FEE_LEVELS_B))]
        perc_accepted = [[] for i in range(len(FEE_LEVELS_B))]
        for ib, vb in enumerate(FEE_LEVELS_B):
            for ii, vs in enumerate(FEE_LEVELS_S):
                if ii % 5 == 0:
                    print('Currently working on buyer fee {} ({} of {}), seller fee {} ({} of {})'.format(vb, ib, len(FEE_LEVELS_B), vs, ii, len(FEE_LEVELS_S)))
                
                [fees_collect, p_accepted] = GetFees(FEE_LEVELS_B[ib], FEE_LEVELS_S[ii], N_TRANSACT, N_BUYERS, N_SELLERS)    
                total_fees[ib].append(fees_collect/N_TRANSACT)
                perc_accepted[ib].append(p_accepted)
    
    # Plot retention and P(transact) as a function of fee level
    fig, axs = plt.subplots(2,2)
    fig.tight_layout(h_pad=2, w_pad=2)
    
    axs[0][0].plot(FEE_LEVELS_B, GetFalloff(FEE_LEVELS_B, BUYER_RETAIN),color=(.2, .2, .8), label='Buyer')
    axs[0][0].plot(FEE_LEVELS_S, GetFalloff(FEE_LEVELS_S, SELLER_RETAIN),color=(.8, .2, .2), label='Seller')
    axs[0][0].set_ylabel('# Transactions')
    axs[0][0].set_xlabel('Fee Amount')
    axs[0][0].legend()
    
    axs[0][1].plot(FEE_LEVELS_B, GetFalloff(FEE_LEVELS_B, BUYER_PBUY),color=(.2, .2, .8))
    axs[0][1].plot(FEE_LEVELS_S, GetFalloff(FEE_LEVELS_S, SELLER_PBUY),color=(.8, .2, .2))
    axs[0][1].set_xlabel('Fee Amount')
    axs[0][1].set_ylabel('P(Transact | Fee)')
    axs[0][1].set_ylim(-0.1, 1.1)
    
    # Plot a heatmap of the fees collected    
    sns.heatmap(total_fees, \
                yticklabels=['%.0f' % FEE_LEVELS_B[i] if FEE_LEVELS_B[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_B))], \
                    xticklabels=['%.0f' % FEE_LEVELS_S[i] if FEE_LEVELS_S[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_S))], \
                        cbar_kws={'label': 'Fees per transaction'}, ax=axs[1][0])
    axs[1][0].set_xlabel('Seller Fee')
    axs[1][0].set_ylabel('Buyer Fee')
    
    sns.heatmap(perc_accepted, \
                yticklabels=['%.0f' % FEE_LEVELS_B[i] if FEE_LEVELS_B[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_B))], \
                    xticklabels=['%.0f' % FEE_LEVELS_S[i] if FEE_LEVELS_S[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_S))], \
                        cbar_kws={'label': 'Percent accepted transactions'}, ax=axs[1][1])
    axs[1][1].set_xlabel('Seller Fee')
    axs[1][1].set_ylabel('Buyer Fee')
    
    if DO_IMBALANCED:
        plt.suptitle('Transactions >> Users')
    else:
        plt.suptitle('Transactions ~= Users')
    
    fig.subplots_adjust(top=0.9)
    
    if DO_IMBALANCED:
        imbal_str = 'NeedsRepeatUsers'
    else:
        imbal_str = 'OneShotUsersOK'
        
    if DO_SAME_WEIB:
        weib_str = 'SameFeeCurve'
    else:
        weib_str = 'SeparateFeeCurves'
        
    if SAVE_FIG:
        fig.savefig('FeeCalculation_{}_{}.png'.format(imbal_str,weib_str))
    
    plt.show()
    

def GetFees(buyer_fee, seller_fee, n_transact, n_buyers, n_sellers):
    global BUYER_RETAIN, SELLER_RETAIN, BUYER_PBUY, SELLER_PBUY
    
    
    # Get a list of buyers and the number of transactions they will make
    buyer_list = [NumberFromMean(GetFalloff(buyer_fee, BUYER_RETAIN)) for i in range(n_buyers)]
    seller_list = [NumberFromMean(GetFalloff(seller_fee, SELLER_RETAIN)) for i in range(n_sellers)]
    
    # Start the transaction loop
    fees_collect = 0
    n_accepted = 0
    for it in range(n_transact):
        
        # Grab a buyer and seller
        pos_inds_b = [i for i in range(len(buyer_list)) if buyer_list[i] > 0]
        pos_inds_s = [i for i in range(len(seller_list)) if seller_list[i] > 0]
        
        if (len(pos_inds_b) == 0) or (len(pos_inds_s) == 0):
            # No more transactions to occur. Break out of loop
            break
        
        # But if there are transactions to be done, let's select a buyer and seller
        buyer_ind = pos_inds_b[random.randint(0,len(pos_inds_b)-1)]
        seller_ind = pos_inds_s[random.randint(0,len(pos_inds_s)-1)]
        
        # Now check whether the buyer/seller "want to" make the transaction given the PBUY Weibull
        buyer_ok = np.random.binomial(n=1,p=GetFalloff(buyer_fee, BUYER_PBUY))==1
        seller_ok = np.random.binomial(n=1,p=GetFalloff(seller_fee, SELLER_PBUY))==1
        
        # If both do want to transact, collect fees and subtract 
        if buyer_ok and seller_ok:
            fees_collect += buyer_fee
            fees_collect += seller_fee
            n_accepted += 1
            
            buyer_list[buyer_ind] -= 1
            seller_list[seller_ind] -= 1
    
    perc_accepted = n_accepted / n_transact
            
    return fees_collect, perc_accepted


def GetFalloff(vals, w_params=(1,1,(0,1))):
    # Unpack w_params
    shape, scale, quant_range = w_params
    
    # Define the weibull    
    weib_fun = lambda x, k, l: 1 - np.exp(-1*((x/l)**k))
    weib_vals = weib_fun(vals, shape, scale)
    
    # Scale the values to cover the range
    weib_vals = weib_vals * (quant_range[1]-quant_range[0])
    
    # And subtract from max to get falloff
    weib_out = quant_range[1]-weib_vals
    
    return weib_out


def NumberFromMean(poiss_mean):
    pmf_potential = poisson.pmf(np.arange(0,30), poiss_mean)
    cmf_potential = np.cumsum(pmf_potential)
    
    rand_val = random.uniform(0,1)
    n = next((x for x in range(len(cmf_potential)) if cmf_potential[x] > rand_val),0)
    
    return n


def GetSales(poiss_mean, this_fee):
    
    SALES_CHECK = np.arange(0,20)
    
    # Buyer and seller means will index into Poisson PMF at SALES_CHECK numbers of sales
    n_sales = poisson.pmf(SALES_CHECK,poiss_mean)
    ev_sales = sum(n_sales*(SALES_CHECK*this_fee))

    return ev_sales

        
def PoolHelper(in_list):
    B_FEE, N_TRANSACT, N_BUYERS, N_SELLERS, FEE_LEVELS_S = in_list
    these_fees = []
    perc_accepted = []
    for ii, vs in enumerate(FEE_LEVELS_S):
        [fees_collect, p_accepted] = GetFees(B_FEE, FEE_LEVELS_S[ii], N_TRANSACT, N_BUYERS, N_SELLERS)    
        these_fees.append(fees_collect/N_TRANSACT)
        perc_accepted.append(p_accepted)
        
    return these_fees, perc_accepted


if __name__=='__main__':
    main()    

