import numpy as np 
import scipy.special as special
import scipy.optimize as optimize
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import functools
import multiprocessing as mp
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
        return [np.arccos(sampling.sample_1d(-1,1)),sampling.sample_1d(0,2*np.pi)]


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


class AnisotropicThermalDistribution(ThermalDistribution):
    """
    Anisotropic distribution following Treumann and Baumjohann 2016
    """
    def __init__(self):
        ThermalDistribution.__init__(self)
        return   

    def sample_aniso_momentum(self,A,**kwargs):
        """ 
        Sample the momentum of an electron. Since it's anisotropic, need to give both energy and direction.
        Done by sampling an isotropic distribution and suitably transforming p_parallel to sample the Treumann anisotropic distribution.
        The temperature provided, thetae, is the temperature of the perpendicular Maxwellian distribution.
        
        Parameters:
            A - anisotropy parameter, defined as T_perp/T_parallel
            kwargs - parameters necessary to sample isotropic distribution
        Returns:
            ln_gammae - natural log of lorentz factor of sample electron
            theta - angle of electron wrt T_parallel (or magnetic field)
            phi - azimuthal angle (sampled randomly)
        """
        ln_gamma=ThermalDistribution.sample_distribution(self,**kwargs)
        theta,phi=ThermalDistribution.sample_direction(self)

        if(A==1):
            return [ln_gamma,theta,phi]
        
        sinth,costh=np.sin(theta),np.cos(theta)
        # transform p_parallel' = p_parallel/sqrt(A).  
        gamma = np.exp(ln_gamma)
        # note here, psq = (p/(me*c)**2)
        psq = gamma**2 - 1
        psq_shifted = psq*costh**2/A + psq*sinth**2
        ln_gamma_shifted = np.log(np.sqrt(psq_shifted+1))
        theta_shifted=np.arctan2(abs(sinth),costh/np.sqrt(A))
        return [ln_gamma_shifted,theta_shifted,phi]

    def dnd3p(self,A,p,theta,ne,thetae_perp,**kwargs):
        """
        Compute normalized anisotropic distribution function value for a given electron momentum, anisotropy value and perpendicular temperature.

        Parameters:
            A - anisotropy parameter, defined as T_perp/T_parallel
            p - momentum of electron (units of momentum/mc)
            theta - angle the momentum makes wrt T_parallel (or B field)
            ne - number density, cgs
            thetae_perp - temperature of perpendicular component, dimensionless units
        Returns:
            dne_d3p - the normalized distribution function (dne/d^3p), see Treumann et al. 2016
        """
        # kwargs['thetae'] = thetae_perp
        psq=p**2
        psq_perp = psq * np.sin(theta)**2
        psq_par  = psq * np.cos(theta)**2

        prefactor = ne * np.sqrt(A) / thetae_perp / (4 * np.pi * special.kn(2,1/thetae_perp))
        # prefactor = ne * np.sqrt(A) / thetae_perp / (4 * np.pi * (utils.constants['me'] * utils.constants['c'])**3 * special.kn(2,1/thetae_perp))
        # return prefactor
        #  return np.exp(-np.sqrt(1 + psq_perp + A*psq_par)/thetae_perp)
        # print(-np.sqrt(1 + psq_perp + A*psq_par)/(thetae_perp * utils.constants['me'] * utils.constants['c']**2))
        # print(prefactor)
        return prefactor * np.exp(-np.sqrt(1 + psq_perp + A*psq_par)/(thetae_perp))

    def dnd2p(self,A,p,theta,ne,thetae_perp,**kwargs):
        """
        dist function in cylindrical coordinates dne/dp_perp/dp_par

        Parameters:
            A - anisotropy parameter, defined as T_perp/T_parallel
            p - momentum of electron (units of momentum/mc)
            theta - angle the momentum makes wrt T_parallel (or B field)
            ne - number density, cgs
            thetae_perp - temperature of perpendicular component, dimensionless units
        Returns:
            dne_d2pdphi - the normalized distribution function integrating out the azimuth, see Treumann et al. 2016
        """
        psq=p**2
        psq_perp = psq * np.sin(theta)**2
        dne_d3p = self.dnd3p(A,p,theta,ne,thetae_perp,**kwargs)

        return dne_d3p * 2 * np.pi * np.sqrt(psq_perp)
        # return dne_d3p * 2 * np.pi * p

    def get_momenta_sampling(self,A,**param_dict):
        """
        helper function to parallelize sampling
        """
        momentum=self.sample_aniso_momentum(A,**param_dict)
        # work in momentum (p/mc) space instead of lorentz factor space
        p=np.sqrt(np.exp(momentum[0])**2-1)
        return [p,momentum[1]],[p*np.sin(momentum[1]),p*np.cos(momentum[1])]

    
    def test_sampling(self,A=1,thetae=10,nsamples=1e3,gamma_min=1,gamma_max=1e3,parallel=False,convergence=False,**kwargs):
        """ tests the sampling procedure by sampling nsamples of gamma from the distribution function and plots out vs. the expected distribution
        NOTE: The parallelization of the function (due to issues with multiprocessing and member functions) means that it will not work if 
        there are member variables specific to the object calling this function.
        """
        nsamples = int(nsamples)
        momenta_xy=np.zeros((nsamples,2))
        momenta=np.zeros((nsamples,2))
        param_dict = {**kwargs,"thetae_perp":thetae,"thetae":thetae,"gamma_min":gamma_min,"gamma_max":gamma_max}
        

        if(parallel):
            nprocs = mp.cpu_count()
            with mp.Pool(processes=int(nprocs*0.9)) as pool:
                partialfunc = functools.partial(self._get_momenta_sampling,A=A,**param_dict)
                momenta_array = np.array(pool.map(partialfunc,range(nsamples)))
                momenta = momenta_array[:,0,:]
                momenta_xy = momenta_array[:,1,:]
            
        else:
            for i in np.arange(nsamples):
                # sine and cos flipped because theta is wrt B which is taken as along +y
                momenta[i],momenta_xy[i]= self.get_momenta_sampling(A,**param_dict)
        
        # print(momenta);exit()
        # now plot
        fig,axs = plt.subplots(2,1,figsize=(7,9),sharex=True)

        # hist_vals,x_bins,y_bins,im=ax.hist2d(momenta_xy[:,0]*np.log10(np.e),momenta_xy[:,1]*np.log10(np.e),bins=20,density=True)
        hist_vals,x_edges,y_edges = np.histogram2d(momenta_xy[:,0],momenta_xy[:,1],bins=50,density=False)
        assert(np.sum(hist_vals)==nsamples)
        # divide by n for probability of bin
        hist_vals/=nsamples

        # pdf = lambda x,y: self.dnd2p(A,np.sqrt(x**2+y**2),np.arctan2(x,y),thetae_perp=thetae,**kwargs)
        pdf = lambda x,y: 2*np.pi*x*self.dnd3p(A,np.sqrt(x**2+y**2),np.arctan2(abs(x),y),thetae_perp=thetae,**kwargs)
        analytic_prob_mass = np.zeros(hist_vals.shape)
        x_mesh = np.zeros(hist_vals.shape)
        y_mesh = np.zeros(hist_vals.shape)
        # Loop over each bin to compute the analytic probability mass in that bin
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                # Define the bin boundaries
                x_min, x_max = x_edges[i], x_edges[i+1]
                y_min, y_max = y_edges[j], y_edges[j+1]
                
                # Compute the analytic probability mass for the bin (integral of the PDF over the bin)
                # We approximate the integral by evaluating the PDF at the bin center and multiplying by the bin area
                bin_center_x = (x_min + x_max) / 2
                bin_center_y = (y_min + y_max) / 2

                x_mesh[i,j] = bin_center_x
                y_mesh[i,j] = bin_center_y
                # this is actually a 3D bin so account for azimuth
                bin_area = (x_max - x_min) * (y_max - y_min)
                analytic_prob_mass[i,j] = pdf(bin_center_x, bin_center_y) * bin_area
            
                # hist_vals[i,j]*= 2*np.pi*bin_center_x
            
                # counts = (momenta_xy[(momenta_xy[:,0]< x_max) & (momenta_xy[:,0]> x_min) & (momenta_xy[:,1]< y_max) & (momenta_xy[:,1]> y_min)].shape)
                # if(counts[0]!=0):
                #     print(hist_vals[i,j])
                # # Get the sample count in this bin
                # sample_count = hist_vals[i, j]
        # exit();
        # hist_vals*=np.max(analytic_prob_mass)/np.max(hist_vals)
        # print(np.max(np.log10(hist_vals)));exit()    
        # im = ax.contour(x_mesh,y_mesh,analytic_prob_mass,levels=20)
        # analytic_prob_mass = np.log10(analytic_prob_mass)
        # hist_vals = np.log10(hist_vals)
        if(not convergence):
            vmin=min(np.min(analytic_prob_mass),np.min(hist_vals));vmax=max(np.max(analytic_prob_mass),np.max(hist_vals));
            im0 = axs[0].pcolormesh(x_edges,y_edges,analytic_prob_mass.T,vmin=vmin,vmax=vmax)
            im1 = axs[1].pcolormesh(x_edges,y_edges,(hist_vals.T),vmin=vmin,vmax=vmax)
            axs[0].set_title(fr'$T_{{\perp}}/T_{{\parallel}} = {A}$')
            axs[1].set_title(fr'Sampled values')
            axs[1].set_xlabel(r'$ p \sin(\theta)$')
            plt.colorbar(im0)
            plt.colorbar(im1)
            for ax in axs:
                ax.set_ylabel(r'$ p \cos(\theta)$')
            
            plt.tight_layout()
            plt.savefig("tests/plots/test_aniso_sampling_log.png",bbox_inches="tight",dpi=150)
        else:
            l1_error = np.sum(np.abs(hist_vals - analytic_prob_mass))
            return l1_error
        
    def test_sampling_convergence(self, A=1,thetae=10,gamma_min=1,gamma_max=1e3,datafile=None,**kwargs):
        if(not datafile):
            nsamples_array = np.logspace(3,7,10)
            errors=[]
            for i,nsamples in enumerate(nsamples_array):
                if(i<len(nsamples_array)-4):
                    parallel=False
                else:
                    parallel=True
                errors.append(self.test_sampling(A,thetae,nsamples,gamma_min,gamma_max,parallel,True,**kwargs))
                print(nsamples,errors[-1])
        else:
            convergence_data = np.loadtxt(datafile)
            nsamples_array=convergence_data[:,0]
            errors = convergence_data[:,1]
            fig,ax=plt.subplots()
            ax.plot(np.log10(nsamples_array),np.log10(errors),label='L1 error')
            ax.plot(np.log10(nsamples_array),np.log10((nsamples_array * errors[0]/nsamples_array[0])**-0.5),label=r'$N^{-1/2}$')
            ax.set_title(fr'$T_{{\perp}}/T_{{\parallel}} = {A}$')
            ax.set_xlabel(r'$\log N$')
            ax.set_ylabel(r'$\log$ L1 norm')
            plt.legend()
            plt.tight_layout()
            plt.savefig("tests/plots/test_aniso_sampling_log_convergence.png",bbox_inches="tight",dpi=150)

    # static method necessary to parallelize with multiprocessing (issues with pickling)
    @staticmethod
    def _get_momenta_sampling(_,A,**param_dict):
        return AnisotropicThermalDistribution().get_momenta_sampling(A,**param_dict)

def samplefunc(x):
    return x

def test_multiprocessing():

    nprocs = mp.cpu_count()
    rey = np.array(range(nprocs))
    for i in range(10):
        with mp.Pool(processes=int(0.9*nprocs)) as pool:
            output = pool.map(samplefunc,rey)
        print(output)


if __name__ == "__main__":
    # thermal_dist = ThermalDistribution()
    # print(issubclass(thermal_dist,DistFunc))
    # print(type(thermal_dist))
    # print(thermal_dist.dndgammae(ne=1,thetae=1e-4,gammae=1.0011775))
    # thermal_dist.test_sampling(nsamples=1e4,gamma_min=1,gamma_max=1e3,thetae=100,ne=1)
    # test_multiprocessing();exit()
    aniso_dist = AnisotropicThermalDistribution()
    aniso_dist.test_sampling_convergence(A=10,thetae=5,gamma_min=1,gamma_max=1e3,ne=1,datafile="tests/aniso_sampling_convergence.txt")
