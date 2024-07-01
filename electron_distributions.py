import numpy as np 
import scipy.special as special
import scipy.optimize as optimize
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import functools

import sampling
import utils

class DistFunc:
    def __init__(self):
        return

    def norm(self):
        return 1

    def get_frac(self,**kwargs):
        """evaluate distribution function"""
        return

    def sample_distribution(self,**kwargs):
        return

    def check_scattering_limits(self,photon_energy,**kwargs):
        """
        given some photon energy, return the appropriate scattering 
        """
        return -1

class ThermalDistribution(DistFunc):
    def __init__(self):
        DistFunc.__init__(self)
        return

    def norm(self):
        """returns normalization constant for integral of dndgammae function below. 1 for thermal MJ distribution"""
        return 1

    def frac(self,**kwargs):
        """function that takes gamma to get the simplified distribution df/dlog(gamma) for sampling (ignores normalization constants)"""
        gammae = kwargs["gammae"]
        thetae = kwargs["thetae"]
        return gammae**2 * np.sqrt(gammae**2 - 1) * np.exp(-gammae/thetae)

    def frac_normalized(self,**kwargs):
        """relativistic thermal distribution function wrt log(gammae)"""
        thetae = kwargs["thetae"]
        ne = kwargs["ne"]
        gammae = kwargs["gammae"]
        beta = np.sqrt(1 - 1/gammae**2)
        return ne/thetae / (special.kn(2,1/thetae)) * self.frac(**kwargs)

    def dndgammae(self,**kwargs):
        """relativistic thermal distribution function wrt (gammae)"""
        thetae = kwargs["thetae"]
        ne = kwargs["ne"]
        gammae = kwargs["gammae"]
        beta = np.sqrt(1 - 1/gammae**2)
        # multiply K2 by exp(1/thetae) for numerical stability -- copied from igrmonty
        if(thetae > 1e-2):
            k2 = special.kn(2,1/thetae) * np.exp(1/thetae)
        else:
            k2 = np.sqrt(np.pi * thetae/ 2)
        return ne/(thetae*k2) * gammae * np.sqrt(gammae**2 - 1) * np.exp(-(gammae-1)/thetae)

    def frac_derivative(self,**kwargs):
        """gives parts of derivative of frac g(gamma) such that dist. func is maximized when g(gamma)=0"""
        gammae = kwargs["gammae"]
        thetae = kwargs["thetae"]
        return gammae - gammae**3 + thetae*(-2 + 3 * gammae**2)

    def sample_distribution(self,**kwargs) -> float:
        """sample the distribution function, might be generic enough to put in sampling.py
        returns sampled ln(gamma) of electron from distribution
        """
        def deriv_function(ln_gammae):
            return self.frac_derivative(gammae=np.exp(ln_gammae),**kwargs)
        mean_gammae = np.log(3*kwargs["thetae"] + special.kn(1,1/kwargs["thetae"])/special.kn(2,1/kwargs["thetae"]))
        dist_max_log_gammae = optimize.fsolve(deriv_function,mean_gammae)
        # print(dist_max_log_gammae);exit()
        dist_max = self.frac_normalized(gammae=np.exp(dist_max_log_gammae),**kwargs)
        # sampling_function = functools.partial(self.frac_normalized())
        def sampling_func(ln_gammae):
            return self.frac_normalized(gammae=np.exp(ln_gammae),**kwargs)
        log_gammae_samp,niter = sampling.rejection_sample_generic(sampling_func,[np.log(kwargs["gamma_min"]),np.log(kwargs["gamma_max"])],dist_max)
        return log_gammae_samp

    def sample_direction(self) -> list:
        """return a random direction of (theta,phi)"""
        return [sampling.sample_1d(0,np.pi),sampling.sample_1d(0,2*np.pi)]


    def check_scattering_limits(self,photon_energy,**kwargs):
        """
        return sigma_t or sigma_kn cross section based on dist func params
        there's a normal way to do this and a numpy way to do this for speedup
        """

        from hotcross import sigma_kn
        # print(photon_energy,kwargs['thetae'])
        # print(utils.bounds['thetae_min'],utils.bounds['photon_energy_min'])
        if(kwargs['thetae'] < utils.bounds['thetae_min'] and photon_energy < utils.bounds['photon_energy_min']):
            return utils.constants['sigma_thomson']
        elif (kwargs['thetae'] < utils.bounds['thetae_min']):
            return sigma_kn(photon_energy)
        else:
            return -1

        # # numpy way
        # condition_numpy = np.where(kwargs['thetae'] < utils.bounds['thetae_min'] and photon_energy < utils.bounds['photon_energy_min'],utils.constants['sigma_thomson'],\
        #     np.where(kwargs['thetae'] < utils.bounds['thetae_min'] and photon_energy >= utils.bounds['photon_energy_min'],sigma_kn(photon_energy),-1))
        # return condition_numpy

    def test_sampling(self,thetae=10,nsamples=1e3,gamma_min=1,gamma_max=1e3,**kwargs):
        """ tests the sampling procedure by sampling nsamples of gamma from the distribution function and plots out vs. the expected distribution"""
        log_gammas=[]
        param_dict = {**kwargs,"thetae":thetae,"gamma_min":gamma_min,"gamma_max":gamma_max}
        for _ in np.arange(nsamples):
            log_gammas.append(self.sample_distribution(**param_dict)*np.log10(np.e))
        # now plot
        fig,ax = plt.subplots()
        hist_vals,_,_=ax.hist(log_gammas,density=True)
        log_gamma_values = np.linspace(np.log10(gamma_min),np.log10(gamma_max),1000)
        dist_values = self.frac_normalized(gammae=10**(log_gamma_values),**param_dict)
        dist_values *= np.max(hist_vals)/np.max(dist_values)
        ax.plot(log_gamma_values,dist_values)
        plt.savefig("tests/plots/test_thermal_sampling_log_2.png",bbox_inches="tight",dpi=150)
    
    def test_sampling_convergence(self,thetae=10,gamma_min=1,gamma_max=1e3,**kwargs):
        """ tests convergence rate of the sampling function"""
        nsamples_array = np.logspace(1,6,10,base=10)
        mean_ln_gammaes = []
        for nsamples in nsamples_array:
            log_gammas=[]
            param_dict = {**kwargs,"thetae":thetae,"gamma_min":gamma_min,"gamma_max":gamma_max}
            for _ in np.arange(nsamples):
                log_gammas.append(self.sample_distribution(**param_dict)*np.log10(np.e))
            mean_ln_gammaes.append(np.mean(log_gammas))


if __name__ == "__main__":
    thermal_dist = ThermalDistribution()
    # print(issubclass(thermal_dist,DistFunc))
    # print(type(thermal_dist))
    # print(thermal_dist.dndgammae(ne=1,thetae=1e-4,gammae=1.0011775))
    thermal_dist.test_sampling(nsamples=1e4,gamma_min=1,gamma_max=1e3,thetae=100,ne=1)
