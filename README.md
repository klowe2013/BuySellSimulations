# BuySellSimulations

This Python code simulates the fee collection for simulated transactions, and incorporates the tradeoff wherein increased fees bring in more revenue, but reduce customer retention and reduce probability of completing a particular transaction. Both average # transactions per user and P(transact | fee) decrease with increased fees, with separate curves for buyers and sellers.

BuySellSimulator_Basic.py is a version that only incorporates average transactions per user.

BuySellSimulator.py adds additional functionality, including a separate calculation for P(transact | fee) and makes the occurence of each transaction more explicit. It has flags for saving figures, for forcing the same relationship of fees to both retention and p(buy) or having these relationships be separate, and for having an imbalanced number of transactions and users (i.e., users ~= transactions vs. users << transactions). 

In all cases, fee relationships are modeled as a Weibull function decreasing from an arbitrary max value. The parameters of the Weibull are manually assigned, but try to capture a higher accepted fee for sellers with a higher sensitivity to change (i.e., steeper slope) than for buyers. Users are randomly assigned a number of transactions from a Poisson distribution with the mean determined by the Weibull function. Then, when buyers and sellers are selected with at least one transaction remaining, the probability of transacting is calculated from a binomial distribution with p=p(transact | fee), and if both accept then the transaction occurs and fees are collected (and number of transactions remaining are decremented for both that buyer and seller). This process is repeated for N_TRANSACT transactions. The resulting plots show the fee relationships to average # of transactions and p(transact | fee) on the top row, and heatmaps for (1) average fees collected per transaction and (2) % accepted transactions as a function of both buyer and seller fees. Hotspots in (1) show the optimization of fee schedules.