import numpy as np
import superphoton as sp
import sampling
from utils import constants
import geometry


class Domain:
    """
    Generic domain class
    """
    def __init__(self,num_superphotons):
        self.num_superphotons=num_superphotons


class OnezoneScattering(Domain):
    """
    create a one-zone sphere of radius r
    """
    def __init__(self,**kwargs):
        Domain.__init__(self,kwargs["num_superphotons"])
        # temperature of scattering medium
        self.thetae = kwargs["thetae"]
        # temperature of blackbody source at center
        self.thetabb = kwargs["thetabb"]
        # number density of scattering medium
        self.ne = kwargs["ne"]
        # radius of scattering medium
        self.radius = kwargs["radius"]
        # superphotons array object. number will vastly increase as scatterings occur, keep as array?
        self.superphotons = np.array([sp.Superphoton() for _ in range(self.num_superphotons)])
        # geometry of problem (currently not used as geodesics handles flatspace)
        self.geom = geometry.MinkowskiSpherical()
        
    def init_superphotons(self,nuMin,nuMax):
        for superphoton in self.superphotons:
            superphoton.x = [0,1e-5,np.pi/2,0]
            superphoton.nu = self.sample_frequency(nuMin,nuMax)
            # igrmonty sets units of k^0 in electron rest mass units
            superphoton.energy = constants["h"] * superphoton.nu / (constants['me'] * constants['c']**2)
            # technically should generate in plasma rest frame and convert back to coordinate units
            # but in this problem plasma is at rest and flat space throughout
            # hand checking the code in tetrads.c seems to have Econ[k][l] = delta_k^l
            # assign a random direction
            costh = sampling.sample_1d(-1,1)
            sinth = np.sqrt(1-costh**2)
            cosphi = sampling.sample_1d(-1,1)
            sinphi = np.sqrt(1-sinphi**2)
            superphoton.k = [superphoton.energy,costh,sinth*cosphi,sinth*sinphi]
            # stefanBoltzmann = 5.67e-5 / np.pi * (self.thetae * constants['me'] * constants['c']**2) / constants['kb']
            # superphoton.weight = self.bnu(superphoton.nu)/self.bnumax() * self.num_superphotons / stefanBoltzmann
            superphoton.weight = self.bnu(superphoton.nu)/self.bnumax() * 1e10

    def bnu(self,nu):
        """Planck function"""
        return 2 * constants["h"] * nu**3 /(constants['c']**3) * 1/(np.exp(constants['h']*nu / (self.thetabb * constants['me'] * constants['c']**2))-1)

    def bnumax(self):
        """Planck function peak for weights"""
        numax = 2.8214391 / constants['h'] * self.thetabb * constants['me']* constants['c']**2
        return self.bnu(numax)

    def sample_frequency(self,nuMin,nuMax):
        return np.exp(sampling.sample_1d()*np.log(nuMax/nuMin) + np.log(nuMin))