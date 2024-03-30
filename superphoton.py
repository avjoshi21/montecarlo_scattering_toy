import numpy as np
import sampling

class Superphoton:
    def __init__(self):
        # member variables, set all to zero for first init
        self.x = np.zeros(4)
        self.k = np.zeros(4)
        self.weight = 0.0
        self.bias=0.0
        self.energy=0.0
        self.Inu = 0.0
        self.tau_scatter = 0.0
        self.tau_absorp = 0.0

    # member functions
    def push_photon(self,dl):