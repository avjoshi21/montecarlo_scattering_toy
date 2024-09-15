"""Common utility functions/variables"""
from functools import wraps
import numpy as np

# units are in cgs
constants={}
constants['c'] = 29979245800.0
constants['me'] = 9.1093837015e-28
constants['kb'] = 1.380649e-16
constants['G'] = 6.674015e-8
constants['M0'] = 1.99e33
constants['pc'] = 3.0857e18
constants['h'] = 6.62607015e-27
constants['sigma_thomson'] = 0.665245873e-24
constants['eV'] = 1.602176634e-12
constants["Lsun"] = 3.837e33


bounds = {}
bounds['thetae_min'] = 1e-4
bounds['photon_energy_min'] = 1e-12
bounds['gammae_max'] = lambda thetae: 1 + 12*thetae
# bounds['gammae_max'] = 12
bounds['weight_min']=1e8;


def beta(gamma):
    return np.sqrt(1 - 1/gamma**2)

def gamma(beta):
    return np.sqrt(1/(1-beta**2))

class operate_along_axis_member_function:
    """
    Decorator for performing operations along the specified axis of a NumPy array.
    This one is for member functions of classes
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self.func
        return self.func.__get__(instance)

    def wrapper(self, instance):
        @wraps(self.func)
        def wrapped_func(arr, *args, axis=None, **kwargs):
            if axis is None:
                return self.func(instance, arr, *args, **kwargs)
            else:
                return np.apply_along_axis(self.func, axis, arr, *args, **kwargs)
        return wrapped_func

class operate_along_axis_function:
    """
    Decorator for performing operations along the specified axis of a NumPy array.
    This one is for generic functions (not member functions)
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, arr, *args, axis=None, **kwargs):
        if axis is None:
            return self.func(arr, *args, **kwargs)
        else:
            return np.apply_along_axis(self.func, axis, arr, *args, **kwargs)   


def getCDF(dist):
    sortedDist = sorted(dist)
    cdf = np.arange(len(dist))/(len(dist)-1)
    return cdf,sortedDist