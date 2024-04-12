"""Common utility functions/variables"""

# units are in cgs
constants={}
constants['c'] = 29979245800
constants['me'] = 9.1093837015e-28
constants['kb'] = 1.380649e-16
constants['G'] = 6.674015e-8
constants['M0'] = 1.99e33
constants['pc'] = 3.0857e18
constants['h'] = 6.62607015e-27
constants['sigma_thomson'] = 0.665245873e-24
constants['eV'] = 1.602176634e-12


bounds = {}
bounds['thetae_min'] = 1e-4
bounds['photon_energy_min'] = 1e-12
bounds['gammae_max'] = lambda thetae: 1 + 12*thetae
# bounds['gammae_max'] = 12
