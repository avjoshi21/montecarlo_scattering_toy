"""
Generate tetrads at a particular position x^mu
"""
import numpy as np

import geometry

class Tetrad:
    def __init__(self,geom: geometry.Geometry):
        self.geom=geom

    def normalize(self,k,gcov):
        """normalize a vector k^mu located at x^mu, k^mu k_mu"""
        norm = np.sqrt(np.abs(np.einsum("i,i",k,self.geom.flip_index(k,gcov))))
        return k/norm
    
    def project_out(self,kcon,lcon,gcov):
        """project out lcon from kcon"""
        return kcon - self.geom.inner(lcon,kcon,gcov)*lcon

    def make_tetrad(self,X,u=None,b=None) -> tuple([np.ndarray,np.ndarray]):
        """
        generate an orthonormal tetrad e^mu_a at a point x^mu given some geometry.
        1. e^mu_0 is aligned with the 4-velocity u (fluid or electron)
        2. e^mu_1 is aligned with the magnetic field 4-vector b (or a radial vector if no b)
        Two other trial vectors selected and their projections on the other two tetrad vectors are subtracted to complete the set
        
        econ/ecov index explanation:
        Econ[k][l]
        k: index attached to tetrad basis
        index down
        l: index attached to coordinate basis 
        index up
        Ecov[k][l]
        k: index attached to tetrad basis
        index up
        l: index attached to coordinate basis 
        index down

        Parameters:
        X -- x^mu position to construct tetrad
        u -- u^mu of the fluid, or trial vector 0 (1,0,0,0)
        b -- b^mu of simulation, or trial vector 1 (0,1,0,0)

        Returns:
        econ -- 4 4-vectors representing e^mu_a (greek index (mu) coordinate basis, latin index (a) tetrad basis index)
        ecov -- same as econ but e^a_mu (flipping greek index with coordinate metric, flipping latin index by minkowski in cartesian)
        """
        if(u is None):
            u = np.array([1,0,0,0])
        if(b is None):
            b = np.array([0,1,0,0])

        gcov = self.geom.gcov(X)
        econ = np.eye(4)
        econ[0] = u.copy()
        econ[1] = self.project_out(b.copy(),econ[0],gcov)
        econ[1] = self.normalize(econ[1],gcov)
        # econ[2] = [0,0,1,0]
        econ[2] = self.project_out(econ[2],econ[0],gcov)
        econ[2] = self.project_out(econ[2],econ[1],gcov)
        econ[2] = self.normalize(econ[2],gcov)
        # econ[3] = [0,0,0,1]
        econ[3] = self.project_out(econ[3],econ[0],gcov)
        econ[3] = self.project_out(econ[3],econ[1],gcov)
        econ[3] = self.project_out(econ[3],econ[2],gcov)
        econ[3] = self.normalize(econ[3],gcov)
        
        ecov = np.eye(4)
        # lower coordinate index mu
        for i in range(4):
            ecov[i] = self.geom.flip_index(econ[i],gcov)
        # raise latin index a
        ecov[0,:]*=-1

        return econ,ecov

def boost(vconb,ucona):
    """
    boosts vector v^b along u^a. assumes both vectors are given in orthonormal tetrad basis
    here velocities are in units of c (I think)
    """
    gamma = ucona[0]
    u_total = np.sqrt(1 - 1/gamma**2)
    # if u^a stationary in original frame, return vconb unchanged
    if (u_total==0):
        return vconb

    ns = ucona[:]/(gamma* u_total)
    g_min_one = gamma-1

    lorentz_matrix = np.array( [
        [ucona[0], -ucona[1], -ucona[2], -ucona[3]],
        [-ucona[1], 1+(g_min_one*ns[1]**2), g_min_one*ns[1]*ns[2], g_min_one*(ns[1]*ns[3])],
        [-ucona[2], g_min_one*ns[1]*ns[2], 1+(g_min_one*ns[2]**2), g_min_one*ns[2]*ns[3]],
        [-ucona[3], g_min_one*(ns[1]*ns[3]), g_min_one*ns[2]*ns[3], 1+(g_min_one*ns[3]**2)]
    ])
    return lorentz_matrix @ vconb
