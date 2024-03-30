"""utility functions for sampling distributions"""
import random
import numpy as np



def sample_1d(minVal=0.0,maxVal=1.0):
    rangeVal = maxVal-minVal
    return minVal + (rangeVal)*(random.random())

def sample_1d_log(minVal=np.log(1e-3),maxVal=np.log(1e3)):
    return np.exp(sample_1d(minVal,maxVal))