"""utility functions for sampling distributions"""
import random
import numpy as np



def sample_1d(minVal=0.0,maxVal=1.0):
    rangeVal = maxVal-minVal
    return minVal + (rangeVal)*(random.random())

def sample_1d_log(minVal=np.log(1e-3),maxVal=np.log(1e3)):
    return np.exp(sample_1d(minVal,maxVal))

def rejection_sample_generic(prob_dist,sample_limits,prob_dist_max,**kwargs):
    """
    A generic routine for rejection sampling (currently 1D). 
    Returns sampled value.
    
    Parameters:
    prob_dist -- PDF function to sample from. Ideally other arguments should be wrapped up already from functools.partial

    sample_limits -- limits
    prob_dist_max -- maximum value of the PDF function

    Returns:
    sample_val -- sampled value
    """
    niter_max=1e6
    niter = 0
    while(1):
        test_sample = sample_1d(*sample_limits)
        dist_sample = prob_dist(test_sample,**kwargs)
        rand_num = random.random()
        niter+=1
        if(dist_sample > prob_dist_max):
            print(f"error! f(sample) > f(maximum). {dist_sample} > {prob_dist_max}");
        if(dist_sample/prob_dist_max > rand_num):
            break
        if(niter>niter_max):
            print(f"sampler rejected {niter_max:.2e} times, exiting")
            return
    return test_sample,niter

