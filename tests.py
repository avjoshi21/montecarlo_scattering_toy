# script for running test problems
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stats

import domain
import hotcross
import electron_distributions as edf
import geodesics
import io_utils
import utils
import scattering
from utils import constants

def test_superphoton_evolution(**domain_kwargs):
    one_zone_domain = domain.OnezoneScattering(**domain_kwargs)
    one_zone_domain.init_superphotons()
    flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()
    dl = domain_kwargs['radius']*2
    io_utils.write_superphoton_data("tests/superphoton_data.txt",prob_domain=one_zone_domain,fileflag="w")
    for ii,superphoton in enumerate(one_zone_domain.superphotons):
        one_zone_domain.superphotons[ii].x,one_zone_domain.superphotons[ii].k = flatspace_geodesic.evolve_geodesic(superphoton.x_init,superphoton.k_init,dl)
    io_utils.write_superphoton_data("tests/superphoton_data.txt",prob_domain=one_zone_domain,fileflag="a")

def test_spectra_blackbody(**domain_kwargs):

    one_zone_domain = domain.OnezoneScattering(**domain_kwargs)
    one_zone_domain.init_superphotons()
    flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()
    dl = domain_kwargs['radius']*2
    for ii,superphoton in enumerate(one_zone_domain.superphotons):
        one_zone_domain.superphotons[ii].evolve_photon_noscattering(flatspace_geodesic,dl=dl)
        # one_zone_domain.superphotons[ii].x,one_zone_domain.superphotons[ii].k = flatspace_geodesic.evolve_geodesic(superphoton.x_init,superphoton.k_init,dl)
        # if one_zone_domain.record_criterion(one_zone_domain.superphotons[ii]):
        #     one_zone_domain.record_superphoton(one_zone_domain.superphotons[ii])
    nuLnu = one_zone_domain.get_nuLnu()
    plt.plot(np.log10(np.exp(one_zone_domain.freq_bins)),np.log10(nuLnu))
    # print(one_zone_domain.freq_bins)
    plt.xlabel(r"$\log \nu$ (Hz)")
    plt.ylabel(r"$\log \nu L_\nu$")
    plt.savefig("tests/plots/blackbody_test.png")

def test_initial_frequencies(**domain_kwargs):
    one_zone_domain = domain.OnezoneScattering(**domain_kwargs)
    one_zone_domain.init_superphotons()
    # freqs = []
    # for ii,superphoton in enumerate(one_zone_domain.superphotons):
    #     freqs.append(superphoton.nu)
    # # plt.hist(np.log10(freqs))
    bnu_vals = one_zone_domain.bnu(np.exp(one_zone_domain.freq_bins))
    plt.plot(np.exp(one_zone_domain.freq_bins),bnu_vals)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("tests/plots/blackbody_frequencies.png")

def test_geodesic_paths(**domain_kwargs):
    one_zone_domain = domain.OnezoneScattering(**domain_kwargs)
    one_zone_domain.init_superphotons()
    flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()
    # dl = domain_kwargs['radius']*2
    alpha_scattering = domain_kwargs["zone_tau"]/one_zone_domain.radius
    dl = scattering.compute_scattering_distance(alpha_scattering=alpha_scattering,bias=1)
    for ii,_ in enumerate(one_zone_domain.superphotons):
        one_zone_domain.superphotons[ii].evolve_photon_noscattering(flatspace_geodesic,dl=dl)
        # one_zone_domain.superphotons[ii].x,one_zone_domain.superphotons[ii].k = flatspace_geodesic.evolve_geodesic(superphoton.x_init,superphoton.k_init,dl)
        # if one_zone_domain.record_criterion(one_zone_domain.superphotons[ii]):
        #     one_zone_domain.record_superphoton(one_zone_domain.superphotons[ii])
    nuLnu = one_zone_domain.get_nuLnu()
    plt.plot(np.log10(np.exp(one_zone_domain.freq_bins)),np.log10(nuLnu))
    # print(one_zone_domain.freq_bins)
    plt.xlabel(r"$\log \nu$ (Hz)")
    plt.ylabel(r"$\log \nu L_\nu$")
    plt.savefig("tests/plots/blackbody_test.png")

def test_alpha_scattering(**domain_kwargs):
    one_zone_domain = domain.OnezoneScattering(**domain_kwargs)
    thermal_dist = edf.ThermalDistribution()
    domain_kwargs["dist_func"] = thermal_dist
    with open("sigma_hotcross_interp_thermal.p","rb") as fp:
        sigma_hotcross_func = pickle.load(fp)
    domain_kwargs["params"]=[1e15,1e-4]
    print(scattering.compute_alpha_scattering(sigma_hotcross_func=sigma_hotcross_func,**domain_kwargs))

def test_photon_beam_scattering(**domain_kwargs):
    """
    scatter superphotons moving in the +x direction at an arbitrary point for electrons at rest, test thomson scattering
    """
    stream_domain = domain.particleStream(**domain_kwargs)
    stream_domain.init_superphotons()
    flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()
    io_utils.write_superphoton_data("tests/stream_superphoton_data.txt",prob_domain=stream_domain,fileflag="w")


    thermal_dist = edf.ThermalDistribution()
    domain_kwargs["dist_func"] = thermal_dist
    kwargs = domain_kwargs
    kwargs["gamma_min"] = 1
    kwargs["gamma_max"] = utils.bounds["gammae_max"](kwargs["thetae"])

    dl=10
    for ii,_ in enumerate(stream_domain.superphotons):
        stream_domain.superphotons[ii].evolve_photon_noscattering(flatspace_geodesic,dl=2*dl)
    
    io_utils.write_superphoton_data("tests/stream_superphoton_data.txt",prob_domain=stream_domain,fileflag="a")

    scattering_loc = flatspace_geodesic.geom.spher_to_cart(stream_domain.superphotons[0].x)
    for ii,_ in enumerate(stream_domain.superphotons):
        stream_domain.superphotons[ii].scatter_superphoton(flatspace_geodesic,**kwargs)
        stream_domain.superphotons[ii].evolve_photon_noscattering(flatspace_geodesic,dl=dl)

    io_utils.write_superphoton_data("tests/stream_superphoton_data.txt",prob_domain=stream_domain,fileflag="a")
    x_array_spherical = np.array([superphoton.x for superphoton in stream_domain.superphotons])
    x_array=np.apply_along_axis(flatspace_geodesic.geom.spher_to_cart,-1,x_array_spherical)
    # set scattering location to origin to sample angles properly
    x_array_shifted = x_array - scattering_loc
    # avoid singularity at origin
    x_array_shifted[:,1]+=1e-10

    # scattering angle is wrt k vector, in our case along x-axis
    # scattering_angles = np.arctan2(np.sqrt(x_array_shifted[:,2]**2 + x_array_shifted[:,3]**2),x_array_shifted[:,1])
    # plt.hist(np.cos(scattering_angles),bins=100,density=True)
    # xvals = np.linspace(-1,1,1000)
    # plt.plot(xvals,scattering.thomson_differential(xvals),label=r"$\frac{2\pi}{\sigma}\frac{d\sigma}{d\Omega}$")
    # plt.xlabel(r"$\cos \theta$")
    # plt.legend()
    # plt.tight_layout()
    # if(stream_domain.superphotons[0].k_init[0]<1e-4):
    #     plt.savefig("tests/plots/scattering_thomson.png")
    # else:
    #     plt.savefig("tests/plots/scattering_kleinnishina.png")


def test_photon_beam_scattering_convergence(**domain_kwargs):
    """
    convergence test for scatter superphotons moving in the +x direction at an arbitrary point
    """
    num_superphotons_array = np.logspace(2,5,20,base=10)
    # num_superphotons_array = [1e2]
    errors = []
    solution_sum = 1
    for num_superphotons in map(int,num_superphotons_array):
        domain_kwargs["num_superphotons"] = num_superphotons
        stream_domain = domain.particleStream(**domain_kwargs)
        stream_domain.init_superphotons()
        flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()

        thermal_dist = edf.ThermalDistribution()
        domain_kwargs["dist_func"] = thermal_dist
        kwargs = domain_kwargs
        kwargs["gamma_min"] = 1
        kwargs["gamma_max"] = utils.bounds["gammae_max"](kwargs["thetae"])

        dl=10
        for ii,_ in enumerate(stream_domain.superphotons):
            stream_domain.superphotons[ii].evolve_photon_noscattering(flatspace_geodesic,dl=2*dl)
        scattering_loc = flatspace_geodesic.geom.spher_to_cart(stream_domain.superphotons[0].x)

        for ii,_ in enumerate(stream_domain.superphotons):
            stream_domain.superphotons[ii].scatter_superphoton(flatspace_geodesic,**kwargs)
            stream_domain.superphotons[ii].evolve_photon_noscattering(flatspace_geodesic,dl=dl)
        
        x_array_spherical = np.array([superphoton.x for superphoton in stream_domain.superphotons])
        x_array=np.apply_along_axis(flatspace_geodesic.geom.spher_to_cart,-1,x_array_spherical)
        # set scattering location to origin to sample angles properly
        x_array_shifted = x_array - scattering_loc
        # avoid singularity at origin
        # x_array_shifted[:,1]+=1e-10

        # scattering angle is wrt k vector, in our case along x-axis
        scattering_angles = np.arctan2(np.sqrt(x_array_shifted[:,2]**2 + x_array_shifted[:,3]**2),(x_array_shifted[:,1]))
        # plt.hist(np.cos(scattering_angles),bins=100,density=True)
        # xvals = np.linspace(-1,1,1000)
        # plt.plot(xvals,list(map(lambda x:3/8 * (1+x**2),xvals)))
        # error=abs(1 - 2/num_superphotons * np.sum(scattering.thomson_differential(np.cos(scattering_angles))))
        # scattering_angles_cdf,sorted_angles = utils.getCDF(np.cos(scattering_angles))
        cdf_func = lambda x: (x+x**3/3+4/3)*3/8
        # plt.step(sorted_angles,scattering_angles_cdf,label='cdf')
        # plt.plot(sorted_angles,list(map(cdf_func,sorted_angles)))
        # plt.savefig("tests/plots/scattering_thomson_cdf.png")
        # error = np.max(np.abs(np.array(list(map(cdf_func,sorted_angles))) - scattering_angles_cdf))
        error,pval= stats.ks_1samp(np.cos(scattering_angles),cdf_func)
        print(num_superphotons,error,pval)
        errors.append(error)    
    plt.loglog(num_superphotons_array,errors,label="KS Statistic")
    plt.loglog(num_superphotons_array,0.5*np.max(errors)*num_superphotons_array[0]**0.5*num_superphotons_array**-0.5,label=r"$1/\sqrt{N}$")
    plt.xlabel(r"$N_\mathrm{superphotons}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tests/plots/scattering_thomson_convergence.png")
    

