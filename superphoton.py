from __future__ import annotations
import numpy as np
import copy

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
        self.bias=1.0
        self.energy=0.0
        self.nu=0.0
        self.Inu = 0.0
        self.tau_scatter = 0.0
        self.tau_absorp = 0.0
        # see return values for evolve_photon. -1 means the superphoton should still be evolved
        self.state_flag = -1
        

        self.times_scattered = 0

        # this is circular design, perhaps there's a better way
        self.dom = dom

    def shallow_copy(self):
        # return a shallow copy of the superphoton. self.dom remains a reference to the original domain variable
        return copy.copy(self)

    # member functions
    def evolve_photon(self,geodesic: geodesics.FlatspaceSphericalGeodesics, **kwargs):
        """
        higher level function for pushing out the photon.
        1. compute scattering location (or use kwargs value)
        2. evolve superphoton to that location
        3. apply scattering absorption till that location
        4. if superphoton is within domain, scatter it
        5. call evolve_photon for the scattered superphoton
        6. else, repeat until superphoton terminates
        
        Parameters:
        geodesic -- Geodesic object necessary to evolve superphoton

        Returns:
        error code for end state of superphoton (superphoton.state_flag)
        -1 -- superphoton still evolving, should not be returned
        0  -- successfully exited system and was recorded
        1  -- not enough weight, superphoton killed
        """
        # checking if the photon should even be evolved. Some redundancy below but the first one is broad level while the ones below check specifics such as sufficient weight
        if(self.state_flag!=-1):
            return self.state_flag

        while (self.state_flag == -1):

            if(self.weight)<utils.bounds["weight_min"]:
                self.state_flag=1
                self.dom.num_superphotons-=1
                # self.delete_superphoton()
                return self.state_flag
            # maybe there's a better way to choose whether to compute dl or just use fixed value
            # dl here refers to the distance where the scattering should occur
            self.bias = self.get_bias()
            # try:
            #     dl = kwargs["dl"]
            # except KeyError:
            dl = scattering.compute_scattering_distance(bias=self.bias,**kwargs)
            dlambda = dl/self.energy
            
            self.x, self.k = geodesic.evolve_geodesic(self.x,self.k,dlambda)

            # apply scattering absorption
            dtau = kwargs["alpha_scattering"] * dl
            if(dtau < 1e-3):
                self.weight *= (1. - dtau/24.*(24. - dtau*(12. - dtau*(4. - dtau))));
            else:
                self.weight *= np.exp(-dtau);
            # print(self.x_init,self.k_init)
            # print("k norm",np.einsum("i,ij,j->",self.k_init,geodesic.geom.gcov(self.x_init),self.k_init))
            # print(dl,dlambda)
            # print(self.dom.radius)
            # print(self.x,self.k)
            if self.dom.record_criterion(self):
                self.dom.record_superphoton(self)
                self.state_flag = 0
            else:
                # create a new superphoton, adjust incoming superphoton weight and scatter new superphoton
                scattered_superphoton = self.create_scattered_superphoton()
                if(scattered_superphoton.state_flag == -1):
                    scattered_superphoton.scatter_superphoton(geodesic=geodesic,**kwargs)
                    # additional bookkeeping, technically k_init for scattered_superphoton is k after scattering
                    scattered_superphoton.k_init = scattered_superphoton.k
                    scattered_superphoton.evolve_photon(geodesic,**kwargs)
                self.weight *= (1 - 1/self.bias)
        
        return self.state_flag

    def create_scattered_superphoton(self):
        """
        get new superphoton + bookkeeping domain variables
        """
        scattered_superphoton = self.shallow_copy()
        scattered_superphoton.times_scattered+=1
        # if(scattered_superphoton.times_scattered>5):
        #     print("something wrong")
        scattered_superphoton.x_init = scattered_superphoton.x
        scattered_superphoton.k_init = scattered_superphoton.k
        scattered_superphoton.weight = self.weight/self.bias
        if(scattered_superphoton.weight<utils.bounds["weight_min"]):
            self.state_flag=1
        # domain bookkeeping
        else:
            self.state_flag=-1
            # self.dom.superphotons.append(scattered_superphoton)
            self.dom.num_superphotons+=1
            self.dom.num_scattered+=1
        # print(scattered_superphoton.weight)
        return scattered_superphoton

    def delete_superphoton(self):
        self.dom.superphotons.remove(self)
        self.dom.num_superphotons-=1
        print(f"deleted superphoton, len(superphotons) {len(self.dom.superphotons)}  num_superphotons={self.dom.num_superphotons}")            

    def get_bias(self):
        # compute the bias for scattering. written here because the limiter is based on the photon's weight
        # return 1;
        max_val = 0.5 * self.weight/utils.bounds["weight_min"]

        bias = max(1/self.dom.tau , 1)
        return max_val if bias > max_val else bias

    def scatter_superphoton(self,geodesic: geodesics.FlatspaceSphericalGeodesics, **kwargs):
        """
        Scatter superphoton kernel. Determine scattering k^mu of superphoton.
        Since scattered superphotons are typically newly created superphotons that are evolved separately,
        handling the bias and weight adjustments should occur in the function that calls this
        1. sample energy and direction of scattering electron, in plasma frame, tetrad basis, done by rejection sampling integrand in hot cross section formula
        2. boost into electron frame
        3. compute direction and energy of scattered superphoton
        TODO: maybe needs just geometry object instead of geodesic?
        """
        # get tetrad basis of plasma frame, for now e^mu_a = delta^mu_a
        tetrad = tetrads.Tetrad(geodesic.geom)
        econ,ecov = tetrad.make_tetrad(self.x)
        # compute k^a from e^a_mu k^mu
        kcona = ecov @ self.k

        electron_pcona = scattering.sample_e_momentum(kcona=kcona,**kwargs)
        
        # # simpler case of using stationary electrons
        # electron_p = np.array([1,0,0,0])
        # electron_pcona = ecov @ electron_p

        scatter_ph_kcona = scattering.sample_scattered_superphoton(kcona,electron_pcona,**kwargs)
        # print(econ,ecov,econ.T @ ecov);exit()
        self.k = econ @ scatter_ph_kcona
        self.energy = self.k[0]
        self.nu = self.energy * (utils.constants['me'] * utils.constants['c']**2) / utils.constants["h"]


    def evolve_photon_noscattering(self,geodesic: geodesics.FlatspaceSphericalGeodesics, **kwargs):
        """
        Simple version of pushing superphoton. No scattering.
        """
        dl = kwargs["dl"]
        dlambda = dl/self.energy
        self.x, self.k = geodesic.evolve_geodesic(self.x,self.k,dlambda)

        if self.dom.record_criterion(self):
            self.dom.record_superphoton(self)
