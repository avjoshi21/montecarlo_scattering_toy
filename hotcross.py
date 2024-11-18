""" module to load and precompute the 'hot cross section' for a given distribution function"""
from scipy import integrate
from scipy import interpolate
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import sys,os,glob

import utils
import electron_distributions as edf

def sigma_thomson():
    """return thomson cross section"""
    return utils.constants['sigma_thomson']

def sigma_kn(eps_e):
    """return the klein nishina cross section
    
    Parameters:
    eps_e -- photon energy in electron rest frame
    
    Returns:
    Klein-Nishina cross section
    """
    eps_e = np.asarray(eps_e)
    kn_loweps =  (1 - 2*eps_e)
    kn_cross = 3/(4*eps_e**2) * \
            (2 + eps_e**2*(1+eps_e)/(1+2*eps_e)**2 + (eps_e**2 - 2*eps_e -2)/(2*eps_e)\
                 * np.log(1 + 2*eps_e))

    return np.where(eps_e < 1e-3, kn_loweps,kn_cross)
   
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
        # print("low limits")
        return sigma
    # get klein nishina cross section based on energy of photon in electron rest frame (function of mu, gammae)
    sigma_invariant = lambda photon_energy,gammae,mu: sigma_kn(photon_energy * gammae * (1 - mu * beta(gammae)))
    
    def hotcross_integrand(gammae,mu,phi=0):
        """
        Collect all terms of integrand for hotcross integration
        
        Parameters:
        mu -- cosine(theta) where theta is the angle between photon and electron
        phi -- azimuthal angle about the spatial part of the photon k^mu

        Returns:
        integrand for hotcross integration

        Note that for anisotropic distributions, there will be another angle of relevance that is the angle (xi) between the photon and B in the plasma frame.
        When evaluating the distribution function, the angles mu and phi need to be changed into xi and varphi (azimuthal angle about B).
        """
        dist_func_args = kwargs
        dist_func_args["mu"] = mu
        dist_func_args["phi"] = phi
        return  (1 - mu * beta(gammae)) * sigma_invariant(photon_energy,gammae,mu) \
        * dist_func.dndgammae(gammae=gammae,**dist_func_args)

    # # attempt to integrate wrt beta instead since it's bounded between 0 and 1 doesn't produce better results than integrating gammae from 1 to inf   
    # # result_betaint = norm*np.array(integrate.tplquad(lambda beta,mu,phi: beta * (1-beta**2)**(-3/2)*hotcross_integrand(1/np.sqrt(1-beta**2),mu,phi),0,2*np.pi,-1,1,0,1))
    # dmu = 0.005
    # dgammae = dmu*kwargs['thetae']
    # mu_vals = np.arange(-1,1,dmu)
    # gammae_vals = np.arange(1,utils.bounds['gammae_max'](kwargs['thetae']),dgammae)
    # mesh_mu_vals,mesh_gammae_vals = np.meshgrid(mu_vals+dmu/2,gammae_vals+dgammae/2)
    # print(mu_vals.shape,gammae_vals.shape);
    # print(np.max(gammae_vals))
    # integrand_vals = hotcross_integrand(mesh_gammae_vals,mesh_mu_vals)
    # # 0.5 is from dn/dgammae/dmu = 0.5 * dn/dgammae
    # result_trapint = 0.5*norm*np.sum(integrand_vals*dmu*dgammae) * utils.constants['sigma_thomson']
    # print(result_trapint)
    gammae_max = np.inf
    # gammae_max = utils.bounds['gammae_max'](kwargs['thetae'])
    result_dblint = 0.5* utils.constants['sigma_thomson']*norm*np.array(integrate.dblquad(lambda gammae,mu: hotcross_integrand(gammae=gammae,mu=mu),-1,1,1,gammae_max))
    # print(result_dblint)
    # result = result_tplint
    # result = result_trapint
    result  = result_dblint
    return result

def compute_hotcross_anisotropic(dist_func,photon_energy,A,xi,**kwargs):
    """
    compute the hot cross section by numerical integration over the momentum space of the distribution function for a set of parameters given a distribution function
    for the anisotropic case, the integral is done in cylindrical coordinates, p_perp p_par and phi (3D integral being necessary as there are no exploitable symmetries of the cosine mu_photon)
    
    parameters:
    dist_func -- distribution function of the electrons
    photon_energy -- k^0 in plasma rest frame (in units of electron rest mass energy)
    A -- anisotropy factor
    xi -- pitch angle of the photon
    returns:
    sigma_hot -- hot cross section
    """
    # factor multiplied to integral to get hotcross, usually 1 / integrate(dndgammae, d3p)
    norm = dist_func.norm();
    beta = lambda gammae: np.sqrt(1 - 1/gammae**2)
    # # call function to return asymptotic values of cross section based on photon, electron energies
    sigma = dist_func.check_scattering_limits(photon_energy,thetae=kwargs['thetae_perp'],**kwargs)
    if (sigma != -1):
        # print("low limits")
        return sigma
    # get klein nishina cross section based on energy of photon in electron rest frame (function of mu, gammae)
    sigma_invariant = lambda photon_energy,gammae,mu_photon: sigma_kn(photon_energy * gammae * (1 - mu_photon * beta(gammae)))
    
    def hotcross_integrand(p_par,p_perp,phi=0):
        """
        Collect all terms of integrand for hotcross integration
        
        Parameters:
        p_par, p_perp -- parallel and perpendicular components of the electron momentum
        phi -- azimuth of the electron momentum wrt photon k

        Returns:
        integrand for hotcross integration

        NOTE: for anisotropic distributions, there will be another angle of relevance that is the angle (xi) between the photon and B in the plasma frame.
        """
        dist_func_args = kwargs
        psq = p_perp**2 + p_par**2
        gammae = np.sqrt(psq+1)
        pvec = [p_par,p_perp,phi]
        p = np.sqrt(psq)
        k = [np.cos(xi),np.sin(xi),0]
        # choice of coordinates [z,r,phi]: the plane spanning k and B is phi=0, T_par and T_perp form remaining cylindrical axes. Assumed k is normalized to 1 here.
        # return dist_func.dnd3p(A,p,None,kwargs['ne'],kwargs['thetae_perp'],p_par,p_perp) * p_perp
        mu_photon = (p_perp*k[1]*np.cos(phi) + p_par*k[0])/p
        return  (1 - mu_photon * beta(gammae)) * sigma_invariant(photon_energy,gammae,mu_photon) * dist_func.dnd3p(A,p,None,kwargs['ne'],kwargs['thetae_perp'],p_par,p_perp) * p_perp

    # p_perp_max = p_par_max = np.inf
    p_perp_max = utils.bounds['p_max'](kwargs['thetae_perp'])
    p_par_max = utils.bounds['p_max'](kwargs['thetae_perp']/A)
    t1 = time.time()
    result_tplquad  = norm*np.array(integrate.tplquad(lambda p_par,p_perp,phi: hotcross_integrand(p_par,p_perp,phi),0,2*np.pi,0,p_perp_max,-p_par_max,p_par_max)) * utils.constants['sigma_thomson']
    t2 = time.time()
    print(f"time taken {t2-t1} seconds")
    result = result_tplquad
    return result


def generate_sigma_hot_fit(dist_func: edf.DistFunc,table_params,**kwargs):
    """
    Compute values of the hot cross section based on a table of parameters ((photon energy, temperature) for thermal dist, for example)
    
    Parameters:
    dist_func -- distribution function of electrons
    table_params -- dictionary consisting of parameters of the distribution function and incident photon energy
    
    Returns:
    sigma_hot_fit_func -- An n-dimensional linear interpolation function over the table of parameters
    """

    # data = np.zeros_like(table_params[0])
    param_keys=[]
    param_vals=[]
    for key,val in table_params.items():
        param_keys.append(key)
        param_vals.append(val)
    if("photon_energy" not in param_keys):
        print("'photon energy' necessary parameter for computing cross sections!");exit();
    table_params_meshgrid = np.array(np.meshgrid(*param_vals,indexing='ij'))
    sigma_hot_vals = np.zeros(shape=table_params_meshgrid[0].shape)
    for i,_ in enumerate(sigma_hot_vals.flat):
        for j,param in enumerate(param_keys):
            kwargs[param]  = table_params_meshgrid[j].flat[i]
        
        sigma_hot_vals[np.unravel_index(i,sigma_hot_vals.shape)] = compute_hotcross(dist_func=dist_func,**kwargs)
        # sigma_hot_vals[np.unravel_index(i,sigma_hot_vals.shape)] = 1.0
    
    # print([np.array(i).shape for i in param_vals],sigma_hot_vals.shape)
    sigma_hot_interp = interpolate.RegularGridInterpolator(tuple(param_vals),sigma_hot_vals,method='linear')
    return sigma_hot_interp

def save_sigma_hotcross_interp(dist_func: edf.DistFunc, picklefile="sigma_hotcross_interp.p",**kwargs):
    """
    Saves sigma_hotcross as a pickle file for values similar to that given in grmonty
    Since the parameters are dependent on the distribution function, perhaps this should be located there
    """
    photon_energy_array = np.logspace(-12,15,220,base=10)
    temp_array = np.logspace(-4,4,80)
    table_params={"photon_energy":photon_energy_array,"thetae":temp_array}
    sigma_hot_interp = generate_sigma_hot_fit(dist_func=dist_func,table_params=table_params,**kwargs)
    with open(picklefile,"wb") as fp:
        pickle.dump(sigma_hot_interp,fp)
    return

def hotcross_lookup(dist_func: edf.DistFunc,sigma_hot_fit,params,**kwargs):
    """
    Given a fitting function has been generated, provide a lookup by first checking if 
    the parameters for the hotcross are within interpolation bounds
    """
    sigma = dist_func.check_scattering_limits(params[0],**kwargs)
    if (sigma != -1):
        return sigma
    else:
        return sigma_hot_fit(params)

def test_sigma_hotcross():
    """test sigma hotcross computation comparing values to Wienke 1985
    """
    thermal_dist = edf.ThermalDistribution()
    # temperature in keV from Wienke
    tempVals = np.array([1,10,100,500,1000,5000])
    # photon frequency values in keV
    nuVals = np.array([1,10,100,500,1000,5000])

    tempVals = tempVals * 1e3 * utils.constants['eV'] / (utils.constants['me'] * utils.constants['c']**2)
    nuVals = nuVals * 1e3 * utils.constants['eV'] / (utils.constants['me'] * utils.constants['c']**2)

    kwargs = {'ne':1,'thetae':1e-4}
    # print(thermal_dist.dndgammae(ne=1,gammae=1.0011225,thetae=0.0001));exit()
    # print(compute_hotcross(thermal_dist,photon_energy=1e-12,**kwargs))
    sigma_hotcross_array = np.zeros((nuVals.shape[0],tempVals.shape[0]))
    for i,eps in enumerate(nuVals):
        for j,thetae in enumerate(tempVals):
            sigma_hotcross_array[i,j] = compute_hotcross(thermal_dist,photon_energy=eps,thetae=thetae,ne=1)
            print(f"{sigma_hotcross_array[i,j]:.4e}")

def test_sigma_hotcross_interp():
    thermal_dist = edf.ThermalDistribution()
    # temperature in keV from Wienke
    tempVals = np.array([1,10,100,500,1000,5000])
    # photon frequency values in keV
    nuVals = np.array([1,10,100,500,1000,5000])

    tempVals = tempVals * 1e3 * utils.constants['eV'] / (utils.constants['me'] * utils.constants['c']**2)
    nuVals = nuVals * 1e3 * utils.constants['eV'] / (utils.constants['me'] * utils.constants['c']**2)
    params={}
    params["photon_energy"] = nuVals
    params["thetae"] = tempVals
    hotcross_fit = generate_sigma_hot_fit(thermal_dist,params,ne=1)
    print(hotcross_fit((nuVals[0]+np.diff(nuVals)[0]/2,tempVals[0])))

def test_compare_hotcross_isotropic_anisotropic():
    """test sigma hotcross computation for the anisotropic case by comparing it to the isotropic integral
    """
    aniso_dist = edf.AnisotropicThermalDistribution()
    A=1
    xi=0
    thermal_dist = edf.ThermalDistribution()
    eps=1e-2
    thetae=10
    ne=1
    # hotcross_iso = (compute_hotcross(thermal_dist,photon_energy=eps,thetae=thetae,ne=ne))
    hotcross_ani = (compute_hotcross_anisotropic(aniso_dist,photon_energy=eps,thetae_perp=thetae,A=A,xi=xi,ne=ne))
    # print(f"anisotropic: {hotcross_ani} isotropic{hotcross_iso} relative error{abs(hotcross_iso-hotcross_ani)/hotcross_iso}")

def test_hotcross_anisotropic_vs_A(check_file_first=True,toplot=False):
    filename = "tests/test_hotcross_anisotropic_vs_A.txt"
    if not(check_file_first and os.path.isfile(filename)):
        aniso_dist = edf.AnisotropicThermalDistribution()
        xi=np.pi/4
        eps=1e-2
        thetae=1
        ne=1
        A_vals = np.logspace(-3,3,7)
        hotcross_vals = []
        if os.path.isfile(filename):
            old_data = np.loadtxt(filename)
        with open(filename,'a') as fp:
            for A in A_vals:
                row_exists=False
                for i in old_data[:]:
                    row_exists= row_exists or np.isclose([A,xi,eps,thetae],i[:-1],rtol=1e-4).all()
                if(not row_exists):
                    val = (compute_hotcross_anisotropic(aniso_dist,eps,A,xi,ne=ne,thetae_perp=thetae)[0])
                    print(f"{A} {xi} {eps} {thetae} {val}")
                    hotcross_vals.append(val)
                    fp.write(f"{A} {xi} {eps:.4e} {thetae} {val}\n")
        
    if toplot:
        fig,ax = plt.subplots()
        hotcross_data = np.loadtxt(filename)
        # make line plot vs A
        for xi in [0,np.pi/4,np.pi/2]:
            A_vals = hotcross_data[hotcross_data[:,1]==xi,0]
            hotcross_vals = hotcross_data[hotcross_data[:,1]==xi,-1]
            plt.plot(np.log10(A_vals),hotcross_vals,label=fr"$\xi={xi:.2f}$")
            ax.set_xlabel(r'$\log \left(A \equiv T_{\perp}/T_{\parallel}\right)$')
            ax.set_ylabel(r'$\sigma_\mathrm{hot}$')
            plt.legend()
            plt.tight_layout()
            plt.savefig("tests/plots/test_hotcross_anisotropic_A.png",dpi=200)


if __name__ == "__main__":
    test_hotcross_anisotropic_vs_A(check_file_first=False,toplot=True)
    # test_compare_hotcross_isotropic_anisotropic()
    # test_sigma_hotcross()