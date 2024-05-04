import numpy as np
import geometry

class FlatspaceSphericalGeodesics:
    def __init__(self):
        self.flat_geom = geometry.MinkowskiSpherical()
        pass

    

    def evolve_geodesic(self,initX: np.array,initK: np.array,dl,debug=False) -> list:
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
        jacobi_spher_to_cart  = self.flat_geom.dx_cartesian_dx_spherical(initX)
        # print(jacobi_spher_to_cart)
        initX_cart = self.flat_geom.spher_to_cart(initX)
        initK_cart = np.einsum("ij,j->i",jacobi_spher_to_cart,initK)

        finalX_cart = initX_cart + initK_cart*dl
        finalK_cart = initK_cart
        jacobi_cart_to_spher = self.flat_geom.dx_spherical_dx_cartesian(finalX_cart)
        finalX = self.flat_geom.cart_to_spher(finalX_cart)
        finalK = np.einsum("ij,j->i",jacobi_cart_to_spher,finalK_cart)

        if debug:
            print("initX_cart",initX_cart)
            print("initK_cart",initK_cart)
            print("finalX_cart",finalX_cart)
        return [finalX,finalK]


def test_flat_geodesic():
    dl = 10;
    init_X = [0,1,np.pi/2,np.pi/2]
    init_K = [1,1,1,0]
    flatspace_geodesic = FlatspaceSphericalGeodesics()
    final_X,final_K = flatspace_geodesic.evolve_geodesic(init_X,init_K,dl);
    print(init_X,init_K)
    print(final_X,final_K)

if __name__ == "__main__":
    test_flat_geodesic()