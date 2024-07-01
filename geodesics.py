import numpy as np
import geometry
import sampling

class FlatspaceSphericalGeodesics:
    def __init__(self):
        self.geom = geometry.MinkowskiSpherical()
        pass

    def normalize_null(self, K:np.array) -> np.array:
        """
        Given a vector K (assumed 4-velocity), return a vector that is appropriately normalized: i.e., k^mu k_mu = 0 
        by normalizing all the spatial coefficients
        """
        spatial_norm = np.sqrt(np.sum(K[1:]**2))

        return np.concatenate(K[0],K[1:]/spatial_norm)

    def evolve_geodesic(self, initX: np.array, initK: np.array, dlambda, debug=False) -> list:
        """
        Since flatspace geodesics can be computed analytically, no need for the metric. 
        Best done by converting to cartesian and converting back.
        
        Parameters
        initX: initial x^mu of the superphoton in spherical
        initK: initial k^mu of the superphoton in spherical
        
        Returns
        finalX: final x^mu of superphoton in spherical
        finalK: final k^mu of superphoton in spherical
        """
        jacobi_spher_to_cart  = self.geom.dx_cartesian_dx_spherical(initX)
        # print(jacobi_spher_to_cart)
        initX_cart = self.geom.spher_to_cart(initX)
        initK_cart = np.einsum("ij,j->i",jacobi_spher_to_cart,initK)

        finalX_cart = np.zeros(4)
        finalK_cart = np.zeros(4)

        finalX_cart = initX_cart + initK_cart*dlambda
        finalK_cart = initK_cart
        jacobi_cart_to_spher = self.geom.dx_spherical_dx_cartesian(finalX_cart)
        finalX = self.geom.cart_to_spher(finalX_cart)
        finalK = np.einsum("ij,j->i",jacobi_cart_to_spher,finalK_cart)

        if debug:
            print("initX_cart",initX_cart)
            print("initK_cart",initK_cart)
            print("finalX_cart",finalX_cart)
        return [finalX,finalK]


def test_flat_geodesic():
    dl = 10;
    init_X = [0,1e-5,np.pi/2,0]
    # init_K = np.array([1,1,0,sampling.sample_1d(0,2*np.pi)])
    init_K = np.array([1,1,0,0])
    geom = geometry.MinkowskiSpherical()
    flatspace_geodesic = FlatspaceSphericalGeodesics()
    flatspace_geodesic.geom = geom
    final_X,final_K = flatspace_geodesic.evolve_geodesic(init_X,init_K,dl);
    print(geom.spher_to_cart(init_X),init_K)
    print(geom.spher_to_cart(final_X),final_K)

if __name__ == "__main__":
    test_flat_geodesic()