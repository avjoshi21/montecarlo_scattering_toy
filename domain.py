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

    def record_criterion(self):
        pass



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
        # frequency bins for scattering, log is base 10
        self.dlE = 0.25
        self.lE0 = 6
        self.lE1 = 20
        self.freq_bins = np.arange(self.lE0,self.lE1+self.dlE,self.dlE)
        self.n_th_bins = 1
        self.n_scattering_bins = 4
        self.spectra = np.zeros(shape=(4,self.n_th_bins,len(self.freq_bins)))
        
    def init_superphotons(self,nuMin,nuMax):
        for superphoton in self.superphotons:
            superphoton.x = [0,1e-5,np.pi/2,0]
            superphoton.x_init = [0,1e-5,np.pi/2,0]
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
            sinphi = np.sqrt(1-cosphi**2)
            superphoton.k = [superphoton.energy,costh,sinth*cosphi,sinth*sinphi]
            superphoton.k_init = [superphoton.energy,costh,sinth*cosphi,sinth*sinphi]
            # stefanBoltzmann = 5.67e-5 / np.pi * (self.thetae * constants['me'] * constants['c']**2) / constants['kb']
            # superphoton.weight = self.bnu(superphoton.nu)/self.bnumax() * self.num_superphotons / stefanBoltzmann
            superphoton.weight = self.bnu(superphoton.nu)/self.bnumax() * 1e6

    def bnu(self,nu):
        """Planck function"""
        return 2 * constants["h"] * nu**3 /(constants['c']**3) * 1/(np.exp(constants['h']*nu / (self.thetabb * constants['me'] * constants['c']**2))-1)

    def bnumax(self):
        """Planck function peak for weights"""
        numax = 2.8214391 / constants['h'] * self.thetabb * constants['me']* constants['c']**2
        return self.bnu(numax)

    def sample_frequency(self,nuMin,nuMax):
        return np.exp(sampling.sample_1d()*np.log(nuMax/nuMin) + np.log(nuMin))

    def record_criterion(self,superphoton : sp.Superphoton):
        if(superphoton.x[1] > self.radius):
            return True
        else:
            return False

    def record_superphoton(self,superphoton: sp.Superphoton):
        """
        record superphoton in the detector
        """
        lE = np.log10(superphoton.energy)
        freq_bin_ind = np.argmin(np.abs(self.freq_bins - lE))
        self.spectra[0,0,freq_bin_ind] += superphoton.weight * superphoton.energy

    def get_nuLnu(self):
        """
        dimensionalize spectra to get nuLnu
        """
        if(self.spectra.shape[1] == 1):
            dOmega = 4 * np.pi
        else:
            dOmega = "set dOmega"
        nuLnu = constants['me']*constants['c']**2 * (4 * np.pi / dOmega) * 1/self.dlE
        nuLnu *= self.spectra[0,0]
        return nuLnu