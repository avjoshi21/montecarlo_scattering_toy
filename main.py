# main script for running test scattering problem
import numpy as np
import matplotlib.pyplot as plt
import time

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
    Define domain keyword args and call specific test functions given in tests.py
    """

    domain_kwargs = {
        "num_superphotons":int(1e5),
        "thetae":4,
        "thetabb":1e-8,
        # 'ne':10**6,
        # 'radius': 100,
        'radius': 100 * constants['G'] * constants["M0"] / constants['c']**2,
        'ln_nu0':np.log(1e-12*constants['me']*constants['c']**2/constants['h']),
        # 'ln_nu0':np.log(1e8),
        'ln_nu1':np.log(1e24),
        'zone_tau':0.1
    }
    # test_superphoton_evolution()
    # test_spectra_blackbody(**domain_kwargs)
    # test_initial_frequencies(**domain_kwargs)
    # save_sigma_hotcross(**domain_kwargs)
    # test_alpha_scattering(**domain_kwargs)
    # test_photon_beam_scattering(**domain_kwargs)
    # test_photon_beam_scattering_convergence(**domain_kwargs)
    test_spectra_pozdnyakov("tests/test_pozdnyakov_spectra_nuLnu.txt",**domain_kwargs)
    # test_spectra_pozdynakov(**domain_kwargs)


if __name__ == "__main__":
    main()