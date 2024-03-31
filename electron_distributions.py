import numpy as np 
import scipy.special as special
import scipy.optimize as optimize
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import sampling

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

class ThermalDistribution(DistFunc):
    def __init__(self):
        DistFunc.__init__(self)
        return

    def norm(self):
        """returns normalization constant for integral of dndgamma function below. 1 for thermal MJ distribution"""
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
        beta = 1 - 1/gammae**2
        return ne/thetae / (special.kn(2,1/thetae)) * self.frac(**kwargs)

    def dndgamma(self,**kwargs):
        """relativistic thermal distribution function wrt (gammae)"""
        thetae = kwargs["thetae"]
        ne = kwargs["ne"]
        gammae = kwargs["gammae"]
        beta = 1 - 1/gammae**2
        return ne/thetae / (special.kn(2,1/thetae)) * gammae**2 * beta * np.exp(-gammae/thetae)

    def frac_derivative(self,**kwargs):
        """gives parts of derivative of frac g(gamma) such that dist. func is maximized when g(gamma)=0"""
        gammae = kwargs["gammae"]
        thetae = kwargs["thetae"]
        return gammae - gammae**3 + thetae*(-2 + 3 * gammae**2)

    def sample_distribution(self,**kwargs) -> float:
        """sample the distribution function, might be generic enough to put in sampling.py
        returns sampled ln(gamma) of electron from distribution
        """
        def deriv_function(gammae):
            return self.frac_derivative(gammae=gammae,**kwargs)
        mean_gammae = np.log(3*kwargs["thetae"] + special.kn(1,1/kwargs["thetae"])/special.kn(2,1/kwargs["thetae"]))
        dist_max_log_gammae = np.log(optimize.fsolve(deriv_function,np.exp(mean_gammae)))
        # print(dist_max_log_gammae);exit()
        dist_max = self.frac_normalized(gammae=np.exp(dist_max_log_gammae),**kwargs)
        niter=0
        while (1):
            log_gamma_min = np.log(kwargs["gamma_min"])
            log_gamma_max=np.log(kwargs["gamma_max"])
            log_gammae_samp = sampling.sample_1d(log_gamma_min,log_gamma_max)
            dist_sample = self.frac_normalized(gammae=np.exp(log_gammae_samp),**kwargs)
            rand_num = random.random()
            niter+=1
            if(dist_sample>dist_max):
                print(f"error! f(sample) > f(maximum). sampled gamma = {np.exp(log_gammae_samp)} maximum gamma = {np.exp(dist_max_log_gammae)}");
            if (dist_sample/dist_max > rand_num):
                # print(niter)
                # print(dist_sample/dist_max,rand_num,niter)
                break
        return log_gammae_samp
        # dist_max_gammae = np.exp(optimize.fsolve(deriv_function,1.0))
        # # print(np.log(dist_max_gammae),kwargs["thetae"]);exit()
        # dist_max = self.dndgamma(gammae=dist_max_gammae,**kwargs)
        # niter=0
        # while (1):
        #     gamma_min = kwargs["gamma_min"]
        #     gamma_max=kwargs["gamma_max"]
        #     gammae_samp = sampling.sample_1d_log(np.log(gamma_min),np.log(gamma_max))
        #     dist_sample = self.dndgamma(gammae=gammae_samp,**kwargs)
        #     rand_num = random.random()
        #     niter+=1
        #     if (dist_sample/dist_max > rand_num):
        #         # print(dist_sample/dist_max,rand_num,niter)
        #         break
        # return gammae_samp


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
        plt.savefig("plots/testing/test_thermal_sampling_log.png",bbox_inches="tight",dpi=150)


if __name__ == "__main__":
    thermal_dist = ThermalDistribution()
    thermal_dist.test_sampling(nsamples=1e4,gamma_min=1,gamma_max=1e3,thetae=100,ne=1)
