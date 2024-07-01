# main script for running test scattering problem
import numpy as np
import matplotlib.pyplot as plt

import domain
import hotcross
import electron_distributions as edf
import geodesics
import io_utils
from utils import constants

from tests import *

def save_sigma_hotcross(**domain_kwargs):
    one_zone_domain = domain.OnezoneScattering(**domain_kwargs)
    thermal_dist = edf.ThermalDistribution()
    hotcross.save_sigma_hotcross_interp(dist_func=thermal_dist,**domain_kwargs)


def main():
    """
    pseudo code for now
    # define scattering domain
    # initialize Ninit superphotons from origin as blackbody, each with same weight and bias values
    # parallelize across available cores using jax. Each core works on a superphoton and the recursively scattered ones until completion
        # for each superphoton, randomly draw a number between 0,1 which determines the distance traveled at which the scattering occurs
        # since the absorption coefficient is constant, computing the distance where the next scattering occurs is not difficult
        # evolve the superphoton by that distance. if the final position is beyond the radius of the sphere, record it at a detector
        # if not, create a scattered superphoton at the scattering position by sampling a scattering kernel for energy and direction.
    """
    domain_kwargs = {
        "num_superphotons":int(1e5),
        "thetae":1,
        "thetabb":1e-8,
        'ne':10**6,
        # 'radius': 1,
        'radius': 1 * constants['G'] * constants["M0"] / constants['c']**2,
        'ln_nu0':np.log(1e8),
        'ln_nu1':np.log(1e14),
        'zone_tau':1
    }
    # test_superphoton_evolution()
    # test_spectra_blackbody(**domain_kwargs)
    # test_initial_frequencies(**domain_kwargs)
    # save_sigma_hotcross(**domain_kwargs)
    # test_alpha_scattering(**domain_kwargs)
    # test_photon_beam_scattering(**domain_kwargs)
    test_photon_beam_scattering_convergence(**domain_kwargs)


if __name__ == "__main__":
    main()