import numpy as np
import superphoton as sp
import sampling
from utils import constants
import geometry


class Domain:
    """
    Generic domain class
    """
    def __init__(self, num_superphotons):
        self.num_superphotons=num_superphotons

    def record_criterion(self, superphoton):
        pass

    def record_superphoton(self,superphoton):
        pass



class OnezoneScattering(Domain):
    """
    create a one-zone sphere of radius r
    """
    def __init__(self, **kwargs):
        Domain.__init__(self,kwargs["num_superphotons"])
        # temperature of scattering medium
        self.thetae = kwargs["thetae"]
        # temperature of blackbody source at center
        self.thetabb = kwargs["thetabb"]
        # number density of scattering medium
        self.ne = kwargs["ne"]
        # radius of scattering medium
        self.radius = kwargs["radius"]
        # superphotons list object. number will vastly increase as scatterings occur, keep as array?
        self.superphotons = ([sp.Superphoton(self) for _ in range(self.num_superphotons)])
        # geometry of problem
        self.geom = geometry.MinkowskiSpherical()
        # frequency bins for scattering, in natural logspace
        self.dlE = 0.25
        self.dlnu = self.dlE
        self.ln_nu0 = kwargs["ln_nu0"]
        self.ln_nu1 = kwargs["ln_nu1"]
        self.freq_bins = np.arange(self.ln_nu0,self.ln_nu1+self.dlnu,self.dlnu)
        self.n_th_bins = 1
        self.n_scattering_bins = 4
        self.spectra = np.zeros(shape=(4,self.n_th_bins,len(self.freq_bins)))
        
    def init_superphotons(self):
        for i in range(self.num_superphotons):
            superphoton = self.superphotons[i]
            superphoton.x = np.array([0,1e-5,np.pi/2,0])
            superphoton.x_init = superphoton.x.copy()
            superphoton.nu = self.sample_frequency()
            # igrmonty sets units of k^0 in electron rest mass units
            superphoton.energy = constants["h"] * superphoton.nu / (constants['me'] * constants['c']**2)
            # technically should generate in plasma rest frame and convert back to coordinate units
            # but in this problem plasma is at rest and flat space throughout
            # hand checking the code in tetrads.c seems to have Econ[k][l] = delta_k^l, which is cartesian, no?
            # assign a random direction
            th = sampling.sample_1d(0,np.pi)
            phi = sampling.sample_1d(0,2*np.pi)
            sinth = np.sin(th)
            costh = np.cos(th)
            sinphi = np.sin(phi)
            cosphi = np.cos(phi)   
            k_cart = np.array([1,sinth*cosphi,sinth*sinphi,costh])
            k_spher = np.einsum("ij,j->i",self.geom.dx_spherical_dx_cartesian(self.geom.spher_to_cart(superphoton.x_init)),k_cart)
            superphoton.k = k_spher
            superphoton.k_init = np.copy(k_spher)
            # stefanBoltzmann = 5.67e-5 / np.pi * (self.thetae * constants['me'] * constants['c']**2) / constants['kb']
            # superphoton.weight = self.bnu(superphoton.nu)/self.bnumax() * self.num_superphotons / stefanBoltzmann
            # superphoton.weight = self.bnu(superphoton.nu)/self.bnumax() * 1e6
            superphoton.weight=1e6

    def bnu(self,nu):
        """Planck function"""
        exp_arg = constants["h"] * nu / (self.thetabb * constants["me"] * constants["c"]**2)
        if(exp_arg < 1e-6):
            return 2 * nu**2 * self.thetabb * constants["me"]
        elif(exp_arg > 1e3):
            return 2 * constants["h"] * nu**3 / constants['c']**2 * np.exp(-exp_arg)
        else:
            return 2 * constants["h"] * nu**3 /(constants['c']**2) * 1/(np.exp(constants['h']*nu / (self.thetabb * constants['me'] * constants['c']**2))-1)

    def bnumax(self):
        """Planck function peak for weights"""
        numax = 2.8214391 / constants['h'] * self.thetabb * constants['me']* constants['c']**2
        return self.bnu(numax)

    def sample_frequency(self):
        bnu_maxval = self.bnumax()
        sample_func = lambda ln_nu: self.bnu(np.exp(ln_nu))
        ln_nu,_ = sampling.rejection_sample_generic(sample_func,[self.ln_nu0,self.ln_nu1],bnu_maxval)
        return np.exp(ln_nu)

    def record_criterion(self, superphoton : sp.Superphoton):
        if(superphoton.x[1] > self.radius):
            return True
        else:
            return False

    def record_superphoton(self, superphoton: sp.Superphoton):
        """
        record superphoton in the detector
        """
        lognu = np.log(superphoton.nu)
        freq_bin_ind = np.argmin(np.abs(self.freq_bins - lognu))
        self.spectra[0,0,freq_bin_ind] += superphoton.weight * superphoton.energy

    def get_nuLnu(self):
        """
        dimensionalize spectra to get nuLnu
        """
        print(self.spectra.shape)
        if(self.spectra.shape[1] == 1):
            dOmega = 4 * np.pi
        else:
            dOmega = "set dOmega"
        # need to multiply by me c^2 as photon energy is given in electron rest mass units.
        nuLnu = constants['me']*constants['c']**2 * (4 * np.pi / dOmega) * 1/self.dlnu
        nuLnu *= self.spectra[0,0]
        return nuLnu

class particleStream(Domain):
    """
    create a stream of particles to test the scattering code.
    """
    def __init__(self, **kwargs):
        Domain.__init__(self,kwargs["num_superphotons"])
        # temperature of scattering medium
        self.thetae = kwargs["thetae"]
        # number density of scattering medium
        self.ne = kwargs["ne"]
        # radius of scattering medium
        self.radius = kwargs["radius"]
        # superphotons list object. number will vastly increase as scatterings occur, keep as array?
        self.superphotons = ([sp.Superphoton(self) for _ in range(self.num_superphotons)])
        # geometry of problem
        self.geom = geometry.MinkowskiSpherical()
        # frequency bins for scattering, in natural logspace
        self.dlE = 0.25
        self.dlnu = self.dlE
        self.ln_nu0 = kwargs["ln_nu0"]
        self.ln_nu1 = kwargs["ln_nu1"]
        self.freq_bins = np.arange(self.ln_nu0,self.ln_nu1+self.dlnu,self.dlnu)
        self.n_th_bins = 1
        self.n_scattering_bins = 4
        self.spectra = np.zeros(shape=(4,self.n_th_bins,len(self.freq_bins)))

    def init_superphotons(self):
        for i in range(self.num_superphotons):
            superphoton = self.superphotons[i]
            superphoton.x = np.array([0,-20-1e-5,np.pi/2,0])
            superphoton.x_init = superphoton.x.copy()
            superphoton.nu = 230e9
            # igrmonty sets units of k^0 in electron rest mass units
            superphoton.energy = constants["h"] * superphoton.nu / (constants['me'] * constants['c']**2)
            # print(superphoton.energy);exit()
            # technically should generate in plasma rest frame and convert back to coordinate units
            # but in this problem plasma is at rest and flat space throughout
            # hand checking the code in tetrads.c seems to have Econ[k][l] = delta_k^l, which is cartesian, no?
            # assign direction of superphoton in +x direction
            k_spher = np.array([1,1,0,0])
            # k_spher = np.einsum("ij,j->i",self.geom.dx_spherical_dx_cartesian(self.geom.spher_to_cart(superphoton.x_init)),k_cart)
            superphoton.k = superphoton.energy * k_spher
            superphoton.k_init = np.copy(k_spher)
            superphoton.weight=1e6
