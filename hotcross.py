""" module to load and precompute the 'hot cross section' for a given distribution function"""
from scipy import integrate
import numpy as np
import utils
import electron_distributions as edf

def sigma_thomson():
    """return thomson cross section"""
    return utils.constants['sigma_thomson']

def sigma_kn(eps_e):
    """return the klein nishina cross section
    parameters:
    eps_e -- photon energy in electron rest frame
    """
    if(eps_e<1e-3):
        return utils.constants['sigma_thomson'] * (1 - 2*eps_e)
    else:
        return utils.constants['sigma_thomson'] * 3/(4*eps_e**2) * \
            (2 + eps_e**2*(1+eps_e)/(1+2*eps_e)**2 + (eps_e**2 - 2*eps_e -2)/(2*eps_e)\
                 * np.log(1 + 2*eps_e))


def compute_hotcross(dist_func,photon_energy,**kwargs):
    """
    compute the hot cross section by numerical integration over the momentum space of the distribution function for a set of parameters given a distribution function
    parameters:
    dist_func -- distribution function of the electrons
    photon_energy -- k^0 in plasma rest frame (in units of electron rest mass energy)
    returns:
    sigma_hot -- hot cross section
    """
    # factor multiplied to integral to get hotcross, usually 1 / integrate(dndgammae, d3p)
    norm = dist_func.norm();
    beta = lambda gammae: np.sqrt(1 - 1/gammae**2)
    # call function to return asymptotic values of cross section based on photon, electron energies
    sigma = dist_func.check_scattering_limits(photon_energy,**kwargs)
    if (sigma != -1):
        print("low limits")
        return sigma
    # get klein nishina cross section based on energy of photon in electron rest frame (function of mu, gammae)
    sigma_invariant = lambda photon_energy,gammae,mu: sigma_kn(photon_energy * gammae * (1 - mu * beta(gammae)))
  
    def hotcross_integrand(gammae,mu,phi=0):
        """collect all terms of integrand for hotcross integration
        parameters:
        mu -- cosine(theta) where theta is the angle between photon and electron
        phi -- azimuthal angle about the spatial part of the photon k^mu

        Note that for anisotropic distributions, there will be another angle of relevance that is the angle (xi) between the photon and B in the plasma frame.
        When evaluating the distribution function, the angles mu and phi need to be changed into xi and varphi (azimuthal angle about B).
        """
        dist_func_args = kwargs
        dist_func_args["mu"] = mu
        dist_func_args["phi"] = phi
        return gammae**2 * (1 - mu * beta(gammae)) * sigma_invariant(photon_energy,gammae,mu) \
        * dist_func.dndgammae(gammae=gammae,**dist_func_args)

    # attempt to integrate wrt beta instead since it's bounded between 0 and 1 doesn't produce better results than integrating gammae from 1 to inf   
    # result_betaint = norm*np.array(integrate.tplquad(lambda beta,mu,phi: beta * (1-beta**2)**(-3/2)*hotcross_integrand(1/np.sqrt(1-beta**2),mu,phi),0,2*np.pi,-1,1,0,1))
    result_tplint = norm*np.array(integrate.tplquad(hotcross_integrand,0,2*np.pi,-1,1,1,utils.bounds['gammae_max'](kwargs['thetae'])))
    return result_tplint


def generate_sigma_hot_fit(dist_func: edf.DistFunc,table_params,**kwargs):
    """
    Compute values of the hot cross section based on a table of parameters ((photon energy, temperature) for thermal dist, for example)
    parameters:
    dist_func -- distribution function of electrons
    table_params -- np.meshgrid object file spanning n dimensions consisting of parameters of the distribution function and incident photon energy
    Returns:
    sigma_hot_fit_func -- An n-dimensional linear interpolation function over the table of parameters
    """

    # data = np.zeros_like(table_params[0])
    # table_params[]
    return

if __name__ == "__main__":
    thermal_dist = edf.ThermalDistribution()


    print(thermal_dist.dndgammae(ne=1,gammae=1.5,thetae=1e-5))
    kwargs = {'ne':1,'thetae':1}
    print(compute_hotcross(thermal_dist,photon_energy=2.82*kwargs['thetae'],**kwargs))