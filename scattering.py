"""
Scattering kernel of the toy problem, also includes computation of the absorption from scattering
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import hotcross
import sampling
import utils
import tetrads
import electron_distributions



def compute_alpha_scattering(sigma_hotcross_func,ne=1,**kwargs):
    """
    alpha_scattering = ne * sigma_hotcross
    """
    params = kwargs["params"]
    del(kwargs["params"])
    dist_func = kwargs["dist_func"]
    del(kwargs["dist_func"])
    return ne * hotcross.hotcross_lookup(dist_func,sigma_hotcross_func,params,**kwargs)

def compute_tau_scattering(sigma_hotcross_func,dl=1, ne=1,**kwargs):
    """
    tau_scattering =  alpha_scattering * dl
    """
    return dl* compute_alpha_scattering(sigma_hotcross_func=sigma_hotcross_func,ne=ne,**kwargs)

def compute_scattering_distance(alpha_scattering,bias,**kwargs):
    """
    Given the absorption coefficient,
    randomly sample between 0 and 1 (call it p). This gives the probability for
    a superphoton to scatter p = 1 - exp(-bias * tau_s)
    then dl = tau_s/alpha_scattering = -log(1-p)/bias/alpha_scattering
    alternatively approximate p \approx tau_s => dl = p / alpha_scattering
    """
    rand_num = sampling.sample_1d()
    # alpha_scattering = compute_alpha_scattering(sigma_hotcross_func,ne=ne,**kwargs)
    return -np.log(1-rand_num)/bias/alpha_scattering
    # return rand_num/(alpha_scattering)

def sample_e_momentum(dist_func: electron_distributions.DistFunc,kcona,**kwargs):
    """
    Samples the momentum of an electron based on an edf, in the plasma tetrad frame
    1. Pick a gamma and direction from the edf
    2. Define a coorinate system by specifying 3 vectors, 
        first one is superphoton k vector (kcona)
        one should be the zero angle (probably e_mu_1, which is aligned with B?),
            wouldn't project out the first vector from second make it misaligned with B tho
        third one is their cross product
        why not just use the tetrad basis vector -> Better to avoid introducing bias
    3. Project scattered direction
    """
    
    scattering_basis_vectors = np.zeros((3,3))
    # rejection sampling to get relevant momentum for scattering
    count=0
    while (count<10000000):
        # get sampled electron lorentz factor in plasma tetrad frame
        ln_gammae = dist_func.sample_distribution(**kwargs)
        gammae = np.exp(ln_gammae)
        betae = utils.beta(gammae)

        # sample direction this is chosen to be wrt to e^mu_(1) which is the B aligned vector (?)
        # better to move this and the gamma sampling to a separate function
        mu = np.cos(sample_e_mu(betae,**kwargs))

        photon_energy_eframe = gammae * (1 - betae * mu) * kcona[0]
        sigma_kn = hotcross.sigma_kn(photon_energy_eframe)

        x1 = sampling.sample_1d()

        if(x1<sigma_kn):
            break

        count+=1

    # first unit vector for system along kcona
    scattering_basis_vectors[0] = kcona[1:]
    scattering_basis_vectors[0]/= np.sqrt(np.sum(scattering_basis_vectors[0]**2))
    
    # take random direction as the 0 angle vector
    theta,phi = dist_func.sample_direction()
    n0 = [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]
    n0_dot_v0 = np.dot(n0,scattering_basis_vectors[0])

    scattering_basis_vectors[1] = n0 - n0_dot_v0*scattering_basis_vectors[0]

    # last unit vector from cross product of first two
    scattering_basis_vectors[2] = np.cross(scattering_basis_vectors[0],scattering_basis_vectors[1])

    # now get four-vector p along unit vectors
    phi = sampling.sample_1d(0,2*np.pi)
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    cth=mu
    sth = np.sqrt(1 - mu**2)

    electron_p = np.zeros(4)
    electron_p[0] = gammae
    electron_p[1:] = gammae * betae * (cth * scattering_basis_vectors[0] + sth * (cphi*scattering_basis_vectors[1] + sphi * scattering_basis_vectors[2]))

    if(betae<0):
        print("Error, betae < 0!")

    return electron_p

def sample_e_mu(betae,**kwargs):
    """
    given a gammae, use inverse sampling to get mu_e from the (1-mu*beta)/2 distribution
    
    """
    x1 = sampling.sample_1d()
    det = 1 +  2*betae + betae**2 - 4*betae*x1
    if (det<0):
        print(f"error, det<0 in mu sampling {betae,x1}")
    mu = (1 - np.sqrt(det))/betae
    return mu

def sample_scattered_superphoton(kcona, electron_p,**kwargs):
    
    new_ph_kcona = np.zeros(4)
    incoming_ph_eframe = tetrads.boost(kcona,electron_p)

    if(incoming_ph_eframe[0]>1e-4):
        k0p = sample_klein_nishina(incoming_ph_eframe[0])
        cth = 1 - 1/k0p + 1/incoming_ph_eframe[0]
    else:
        k0p = incoming_ph_eframe[0]
        cth = sample_thomson()
    sth = np.sqrt(abs(1-cth**2))

    scattering_basis_vectors=np.zeros((3,3))
    kemag = np.sqrt(np.sum(incoming_ph_eframe[1:]**2))
    scattering_basis_vectors[0] = incoming_ph_eframe[1:]/kemag

    # take random direction as the 0 angle vector
    theta,phi = (sampling.sample_1d(0,np.pi),sampling.sample_1d(0,2*np.pi))
    n0 = [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]
    n0_dot_v0 = np.dot(n0,scattering_basis_vectors[0])

    scattering_basis_vectors[1] = n0 - n0_dot_v0*scattering_basis_vectors[0]
    scattering_basis_vectors[1]/=np.sqrt(np.sum(scattering_basis_vectors[1]**2))

    # last unit vector from cross product of first two
    scattering_basis_vectors[2] = np.cross(scattering_basis_vectors[0],scattering_basis_vectors[1])

    # now get four-vector p along unit vectors
    phi = sampling.sample_1d(0,2*np.pi)
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    scattered_ph_eframe = np.zeros(4)
    scattered_ph_eframe[0] = k0p
    scattered_ph_eframe[1:] = k0p * (cth * scattering_basis_vectors[0] + sth * (cphi*scattering_basis_vectors[1] + sphi * scattering_basis_vectors[2]))

    return tetrads.boost(scattered_ph_eframe,electron_p)



def sample_klein_nishina(k0):
    
    k0pmin = k0/(1 + 2*k0)
    k0pmax = k0
    ncount=0
    while(1):
        k0p_test = k0pmin + (k0pmax - k0pmin) * sampling.sample_1d()

        x1 = (2 * 1 + 2*k0 + 2 *k0**2)/(k0**2 * (1+2*k0))
        x1*=sampling.sample_1d()
        ncount+=1
        if(x1<klein_nishina_differential(k0,k0p_test)):
            break
    return k0p_test

def klein_nishina_differential(a,ap):
    ch = 1. + 1. / a - 1. / ap
    kn = (a / ap + ap / a - 1. + ch * ch) / (a * a)

    return (kn)

def sample_thomson():
    ncount=0
    while(1):
        x1 = sampling.sample_1d(-1,1)
        x2 = sampling.sample_1d(0,3/4)

        if (x2 < 3/8 * (1 + x1**2)):
            break
    return x1

def thomson_differential(x):
    return 3/8 * (1 + x**2)

def test_sample_thomson():
    nsamples = 1e5
    sampled_values = [sample_thomson() for _ in np.arange(nsamples)]
    plt.hist(sampled_values,bins=100,density=True)
    xvals = np.linspace(-1,1,1000)
    plt.plot(xvals,3/8 * (1+xvals**2))
    plt.xlabel(r"$\cos \theta$")
    plt.tight_layout()
    plt.savefig("tests/plots/scattering_thomson_sampling.png")

def test_sample_thomson_convergence():
    num_superphotons_array = np.logspace(2,5,100,base=10)
    # num_superphotons_array = [1e5]
    errors = []
    solution_sum = 1
    for num_superphotons in map(int,num_superphotons_array):
        # sampled_values = np.random.uniform(-1,1,size=num_superphotons)
        # error=abs(1 - 2/num_superphotons * np.sum(thomson_differential((sampled_values))))
        sampled_values = np.array([sample_thomson() for _ in np.arange(num_superphotons)])
        # error=abs(1 - 2/num_superphotons * np.sum(((sampled_values))))
        sample_cdf,sorted_angles = utils.getCDF((sampled_values))
        cdf_func = lambda x: (x+x**3/3+4/3)*3/8
        # plt.step(sorted_angles,scattering_angles_cdf,label='cdf')
        # plt.plot(sorted_angles,list(map(cdf_func,sorted_angles)))
        # plt.savefig("tests/plots/scattering_thomson_cdf.png")
        # error = np.max(np.abs(np.array(list(map(cdf_func,sorted_angles))) - sample_cdf))
        error,pval= stats.ks_1samp((sampled_values),cdf_func)
        print(num_superphotons,error,pval)
        errors.append(error)
    plt.loglog(num_superphotons_array,errors)
    plt.loglog(num_superphotons_array,0.5*np.max(errors)*num_superphotons_array[0]**0.5*num_superphotons_array**-0.5)
    plt.tight_layout()
    plt.savefig("tests/plots/sampling_thomson_convergence.png")

def test_sample_kn():
    k0=0.18614589527035774
    nsamples = 1e5
    sampled_values = [sample_klein_nishina(k0) for _ in np.arange(nsamples)]
    plt.hist(sampled_values,bins=100,density=True)
    xvals = np.linspace(-1,1,1000)
    # plt.plot(xvals,3/8 * (1+xvals**2))
    # plt.xlabel(r"$\cos \theta$")
    plt.tight_layout()
    plt.savefig("tests/plots/scattering_kleinnishina_sampling.png")

if __name__ == "__main__":
    test_sample_thomson_convergence()