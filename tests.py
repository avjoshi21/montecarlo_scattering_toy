# script for running test problems
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stats
import copy
import h5py
import sys,os,glob
from utils import time_function

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
    alpha_scattering = one_zone_domain.tau/one_zone_domain.radius
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
    
@time_function
def test_spectra_pozdnyakov(file_check=None,grmonty_file="/home/avjoshi2/igrmonty/spectrum.h5",**domain_kwargs):
    if(file_check is None):
        file_check=""
    domain_kwargs["num_superphotons"] = int(1e5)
    domain_kwargs["zone_tau"]=1e-4
    domain_kwargs["bias_init"] = 1

    thermal_dist = edf.ThermalDistribution()
    domain_kwargs["dist_func"] = thermal_dist
    kwargs = domain_kwargs
    kwargs["gamma_min"] = 1
    kwargs["gamma_max"] = utils.bounds["gammae_max"](kwargs["thetae"])

    one_zone_domain = domain.OnezoneScattering(**kwargs)
    kwargs["ne"]=one_zone_domain.ne
    one_zone_domain.init_superphotons()
    flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()
    if(not os.path.isfile(file_check)):
        # dl = domain_kwargs['radius']*0.7
        kwargs["alpha_scattering"] = one_zone_domain.tau/one_zone_domain.radius
        print("num_superphotons_init, num_superphotons",(one_zone_domain.num_superphotons_init,one_zone_domain.num_superphotons))
        print(len(one_zone_domain.superphotons))
        for ii in range(len(one_zone_domain.superphotons)):
            state=one_zone_domain.superphotons[ii].evolve_photon(flatspace_geodesic,**kwargs)
            # one_zone_domain.superphotons[ii].x,one_zone_domain.superphotons[ii].k = flatspace_geodesic.evolve_geodesic(superphoton.x_init,superphoton.k_init,dl)
            # if one_zone_domain.record_criterion(one_zone_domain.superphotons[ii]):
            #     one_zone_domain.record_superphoton(one_zone_domain.superphotons[ii])
        
        # iterate through and remove all the absorbed superphotons
        # for ii,superphoton in enumerate(copy.copy(one_zone_domain.superphotons)):
        #     if superphoton.state_flag == 1:
        #         one_zone_domain.superphotons.remove(superphoton)

        # with open("tests/test_spectra_pozdnyakov_superphoton.pkl","wb") as fp:
        #     pickle.dump(one_zone_domain.superphotons,fp)
        # io_utils.write_superphoton_data("tests/test_spectra_pozdnyakov.txt",prob_domain=one_zone_domain,fileflag="w")
        print("num_scattered",one_zone_domain.num_scattered)
        print("num_superphotons_init, num_superphotons",(one_zone_domain.num_superphotons_init,one_zone_domain.num_superphotons))
        nuLnu = one_zone_domain.get_nuLnu()[:,0,:]
        ln_nu = one_zone_domain.freq_bins
        nu = np.exp(ln_nu)
        np.savetxt("tests/test_spectra_pozdnyakov_nuLnu.txt",np.vstack((ln_nu,nuLnu)))

    else:
        spectra_data = np.loadtxt(file_check)
        ln_nu = spectra_data[0]
        nuLnu = spectra_data[1:]
        nu = np.exp(ln_nu)

    for scattering_index in range(1):
        # print(([i.weight for i in one_zone_domain.superphotons if i.times_scattered==scattering_index]))
        # plt.hist([np.log10(i.weight* i.energy * utils.constants["me"] * utils.constants["c"]**2 ) for i in one_zone_domain.superphotons if i.times_scattered==scattering_index],alpha=0.6,bins=6,label=f"{scattering_index}",density=True)
        output=nuLnu[scattering_index]/nu
        plt.step(np.log10(nu),np.log10(output),label=f"{scattering_index}")
    output=np.sum(nuLnu,axis=0)/nu
    plt.step(np.log10(nu),np.log10(output),label=f"Test code")

    hfp = h5py.File(grmonty_file,'r')
    nu = np.power(10.,hfp["output"]["lnu"]) * constants["me"] * constants["c"]**2 / constants["h"]
    nuLnu_grmonty = np.array(hfp["output"]["nuLnu"]) * constants["Lsun"]
    nuLnu_grmonty = np.sum(nuLnu_grmonty[:,:,-1],axis=0)
    nuLnu_grmonty *= np.max(nuLnu)/np.max(nuLnu_grmonty)
    hfp.close()
    plt.step(np.log10(nu), np.log10(nuLnu_grmonty/nu), label="grmonty")
    plt.plot()
    plt.legend()
    # # print(one_zone_domain.freq_bins)
    # plt.ylim([np.max(np.log10(nuLnu[0]))-10,np.max(np.log10(nuLnu[0]))])
    # plt.ylim([np.log10(np.max(output))-5,np.log10(np.max(output)*1.1)])
    plt.xlabel(r"$\log \nu$ (Hz)")
    # plt.ylabel(r"$\log \nu L_\nu$")
    plt.ylabel(r"$\log F_\nu$")

    plt.tight_layout()
    plt.savefig("tests/plots/pozdnyakov_spectra_grmonty_comparison.png")
    
def test_spectra_pozdnyakov_convergence(file_check=None,**domain_kwargs):
    domain_kwargs["zone_tau"]=1e-8
    domain_kwargs["bias_init"] = 1

    thermal_dist = edf.ThermalDistribution()
    domain_kwargs["dist_func"] = thermal_dist
    kwargs = domain_kwargs
    kwargs["gamma_min"] = 1
    kwargs["gamma_max"] = utils.bounds["gammae_max"](kwargs["thetae"])

    one_zone_domain = domain.OnezoneScattering(**kwargs)
    kwargs["ne"]=one_zone_domain.ne
    one_zone_domain.init_superphotons()
    flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()

    N_values = np.logspace(3, 6, num=10, base=10, dtype=int)
    L1_norms = []

    for N in N_values:
        print(N)
        if(file_check is None):
            file_check=""
        else:
            file_check = f"tests/test_pozdnyakov_spectra_nuLnu_convergence_N{int(N):08d}.txt"
        if(not os.path.isfile(file_check)):
            domain_kwargs["num_superphotons"] = N
            one_zone_domain = domain.OnezoneScattering(**domain_kwargs)
            one_zone_domain.init_superphotons()
            flatspace_geodesic = geodesics.FlatspaceSphericalGeodesics()
            kwargs = domain_kwargs
            kwargs["gamma_min"] = 1
            kwargs["gamma_max"] = utils.bounds["gammae_max"](kwargs["thetae"])
            kwargs["alpha_scattering"] = one_zone_domain.tau / one_zone_domain.radius
            for ii in range(len(one_zone_domain.superphotons)):
                one_zone_domain.superphotons[ii].evolve_photon(flatspace_geodesic, **kwargs)

            nuLnu = one_zone_domain.get_nuLnu()[:, 0, :]
            nu = one_zone_domain.freq_bins
            np.savetxt(f"tests/test_spectra_pozdnyakov_nuLnu_convergence_N{int(N):08d}.txt", np.vstack((nu, nuLnu)))
        else:
            print(N)
            spectra_data = np.loadtxt(f"tests/test_spectra_pozdnyakov_nuLnu_convergence_N{int(N):08d}.txt")
            ln_nu = spectra_data[0]
            nuLnu = spectra_data[1:]
            nu = np.exp(ln_nu)
        
        output = np.sum(nuLnu, axis=0) / np.exp(nu)
        hfp = h5py.File(f"/home/avjoshi2/igrmonty/sphere_convergence/python_comp/spectrum_{N:08d}.h5", 'r')
        nu_grmonty = np.power(10., hfp["output"]["lnu"]) * constants["me"] * constants["c"]**2 / constants["h"]
        nuLnu_grmonty = np.array(hfp["output"]["nuLnu"]) * constants["Lsun"]
        nuLnu_grmonty = np.sum(nuLnu_grmonty[:, :, -1], axis=0)/nu_grmonty
        nuLnu_grmonty *= np.max(output) / np.max(nuLnu_grmonty)
        hfp.close()

        L1_norm = np.sum(np.abs(output - nuLnu_grmonty))
        L1_norms.append(L1_norm)

    plt.figure()
    plt.loglog(N_values, L1_norms, label="L1 Norm")
    plt.xlabel(r"$N_\mathrm{superphotons}$")
    plt.ylabel(r"L1 Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tests/plots/L1_norm_convergence.png")
