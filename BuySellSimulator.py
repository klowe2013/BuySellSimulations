import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns
import random
from IPython.display import clear_output
import os
from multiprocessing import Pool

def main():
    
    # Set globals
    global BUYER_RETAIN, SELLER_RETAIN, BUYER_PBUY, SELLER_PBUY
    
    # Set constants
    
    # Set parameters for Weibull function to illustrate falloff with increased fees.
    # We'll use two sets: one for "retention" (i.e., how many transactions does an individual make) and "P(transact)", how likely an inidvidual is to make a given transaction
    # Here is the retention parameterization
    BUYER_RETAIN = (1.5, 7, 10) # (shape, scale, max)
    SELLER_RETAIN = (5, 7, 10) # (shape, scale, max)
    
    # For simplicity on this pass, assume same shape for P(buy)
    BUYER_PBUY = (1.5, 7, 1)
    SELLER_PBUY = (5, 7, 1)
    
    # Set range of fee levels
    FEE_LEVELS_B = np.arange(0,10,.2)
    FEE_LEVELS_S = np.arange(0,12,.2)
    
    # Number of buyers/sellers to simulate
    N_BUYERS = 1000
    N_SELLERS = 1000
    
    # And how many transactions to simulate 
    N_TRANSACT = 1000
    
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
            total_fees = p.map(PoolHelper, [[vb, N_TRANSACT, N_BUYERS, N_SELLERS, FEE_LEVELS_S] for ib, vb in enumerate(FEE_LEVELS_B)])
            
    else:
        # Initialize output matrix and start fee loops
        total_fees = [[] for i in range(len(FEE_LEVELS_B))]
        for ib, vb in enumerate(FEE_LEVELS_B):
            for ii, vs in enumerate(FEE_LEVELS_S):
                clear_output(wait=True)
                if ii % 5 == 0:
                    print('Currently working on buyer fee {} ({} of {}), seller fee {} ({} of {})'.format(vb, ib, len(FEE_LEVELS_B), vs, ii, len(FEE_LEVELS_S)))
                
                fees_collect = GetFees(FEE_LEVELS_B[ib], FEE_LEVELS_S[ii], N_TRANSACT, N_BUYERS, N_SELLERS)    
                total_fees[ib].append(fees_collect/N_TRANSACT)
            
            
    # Plot a heatmap of the fees collected    
    plt.figure()
    sns.heatmap(total_fees, \
                yticklabels=['%.2f' % FEE_LEVELS_B[i] if FEE_LEVELS_B[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_B))], \
                    xticklabels=['%.2f' % FEE_LEVELS_S[i] if FEE_LEVELS_S[i] % 1 == 0 else '' for i in range(len(FEE_LEVELS_S))], \
                        cbar_kws={'label': 'Expected revenue per buyer/seller pair'})
    plt.xlabel('Seller Fee')
    plt.ylabel('Buyer Fee')
    plt.show()
    

def GetFees(buyer_fee, seller_fee, n_transact, n_buyers, n_sellers):
    global BUYER_RETAIN, SELLER_RETAIN, BUYER_PBUY, SELLER_PBUY
    
    
    # Get a list of buyers and the number of transactions they will make
    buyer_list = [NumberFromMean(GetFalloff(buyer_fee, BUYER_RETAIN[0], BUYER_RETAIN[1], (0, BUYER_RETAIN[2]))) for i in range(n_buyers)]
    seller_list = [NumberFromMean(GetFalloff(seller_fee, SELLER_RETAIN[0], SELLER_RETAIN[1], (0, SELLER_RETAIN[2]))) for i in range(n_sellers)]
    
    # Start the transaction loop
    fees_collect = 0
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
        buyer_ok = np.random.binomial(n=1,p=GetFalloff(buyer_fee, BUYER_PBUY[0], BUYER_PBUY[1], (0, BUYER_PBUY[2])))==1
        seller_ok = np.random.binomial(n=1,p=GetFalloff(seller_fee, SELLER_PBUY[0], SELLER_PBUY[1], (0, SELLER_PBUY[2])))==1
        
        # If both do want to transact, collect fees and subtract 
        if buyer_ok and seller_ok:
            fees_collect += buyer_fee
            fees_collect += seller_fee
            
            buyer_list[buyer_ind] -= 1
            seller_list[seller_ind] -= 1
            
    return fees_collect


def GetFalloff(vals, shape, scale, quant_range=(0,1)):
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

        
def GetSparkContext(core_prop = 1):
    import pyspark as ps
    import os
    
    # Get number of cores
    try:
        n_cores = len(os.sched_getaffinity(0))
    except:
        import psutil
        n_cores = psutil.cpu_count()
        
    n_str = 'local[' + str(int(n_cores*core_prop)) + ']'
    
    spark = ps.sql.SparkSession.builder \
    .master(n_str) \
        .appName('spark-ml') \
            .getOrCreate()
    sc = spark.sparkContext

    return sc


def PoolHelper(in_list):
    B_FEE, N_TRANSACT, N_BUYERS, N_SELLERS, FEE_LEVELS_S = in_list
    these_fees = []
    for ii, vs in enumerate(FEE_LEVELS_S):
        fees_collect = GetFees(B_FEE, FEE_LEVELS_S[ii], N_TRANSACT, N_BUYERS, N_SELLERS)    
        these_fees.append(fees_collect/N_TRANSACT)
    
    return these_fees


if __name__=='__main__':
    main()    

