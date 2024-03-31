""" module to load and precompute the 'hot cross section' for a given distribution function"""
from scipy import integrate
import numpy as np
from utils import constants

def sigma_thomson():
    """return thomson cross section"""
    return constants['sigma_thomspon']

def sigma_kn():
    """return the klein nishina cross section"""

def get_sigma(**kwargs):
    """find cross section sigma in hot cross section integral.
    Thomson for low energy, Klein Nishina for higher
    """
    return 

def compute_hotcross(dist_func: DistFunc,**kwargs):
    """pre-compute the hotcross for a set of parameters given a distribution function"""

    norm = dist_func.norm();
    gammae = lambda beta: np.sqrt(1/(1-beta**2))
    sigma = get_sigma(**kwargs)
    integrand  = lambda beta,mu,phi : beta * np.sqrt(1 - beta**2) * (1 - mu * beta) * sigma
