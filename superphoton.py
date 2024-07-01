from __future__ import annotations
import numpy as np

import sampling
import electron_distributions
import scattering
import geodesics
import domain
import tetrads
import hotcross
import utils

class Superphoton:
    def __init__(self,dom: domain.Domain):
        # member variables, set all to zero for first init
        self.x_init = np.zeros(4)
        self.x = np.zeros(4)
        self.k_init = np.zeros(4)
        self.k = np.zeros(4)
        self.weight = 0.0
        self.bias=0.0
        self.energy=0.0
        self.nu=0.0
        self.Inu = 0.0
        self.tau_scatter = 0.0
        self.tau_absorp = 0.0

        # this is circular design, perhaps there's a better way
        self.dom = dom

    # member functions
    def evolve_photon(self,geodesic: geodesics.FlatspaceSphericalGeodesics, **kwargs):
        """
        higher level function for pushing out the photon.
        1. compute scattering location
        2. evolve superphoton to that location
        3. if superphoton is within domain, scatter it
        4. call evolve_photon for the scattered superphoton
        
        Parameters:
        geodesic -- Geodesic object necessary to evolve superphoton

        Returns:
        error code for end state of superphoton
        0 -- successfully exited system and was recorded
        1 -- not enough weight, superphoton killed
        """
        if(self.weight)<1:
            return 1

        dl = scattering.compute_scattering_distance(**kwargs)
        dlambda = dl/self.energy

        self.x, self.k = geodesic.evolve_geodesic(self.x,self.k,dlambda)

        if self.dom.record_criterion(self):
            self.dom.record_superphoton(self)
            return 0
        else:
            self.scatter_superphoton(geodesic=geodesic,**kwargs)
    
    def scatter_superphoton(self,geodesic: geodesics.FlatspaceSphericalGeodesics, **kwargs):
        """
        scattering kernel of superphoton.
        1. sample energy and direction of scattering electron, in plasma frame, tetrad basis, done by rejection sampling integrand in hot cross section formula
        2. boost into electron frame
        3. compute direction and energy of scattered superphoton
        4. adjust weights of incident and scattered superphoton
        5. return if any photon weights <1
        6. evolve_photon for scattered superphoton
        """

        # get tetrad basis of plasma frame, for now e^mu_a = delta^mu_a
        tetrad = tetrads.Tetrad(geodesic.geom)
        econ,ecov = tetrad.make_tetrad(self.x)
        # compute k^a from e^a_mu k^mu
        kcona = ecov @ self.k

        # electron_pcona = scattering.sample_e_momentum(kcona=kcona,**kwargs)
        
        # simpler case of using stationary electrons
        electron_p = np.array([1,0,0,0])
        electron_pcona = ecov @ electron_p

        scatter_ph_kcona = scattering.sample_scattered_superphoton(kcona,electron_pcona,**kwargs)
        # print(econ,ecov,econ.T @ ecov);exit()
        self.k = econ @ scatter_ph_kcona
        self.energy = self.k[0]


    def evolve_photon_noscattering(self,geodesic: geodesics.FlatspaceSphericalGeodesics, **kwargs):
        """
        Simple version of pushing superphoton. No scattering.
        """
        dl = kwargs["dl"]
        dlambda = dl/self.energy
        self.x, self.k = geodesic.evolve_geodesic(self.x,self.k,dlambda)

        if self.dom.record_criterion(self):
            self.dom.record_superphoton(self)
