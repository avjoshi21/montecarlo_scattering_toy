import numpy as np


class FlatspaceSphericalGeodesics:
    def __init__(self):
        pass

    def cart_to_spher(self,X: np.array) -> np.array:
        """
        Should definitely not be a function here but in geometry.py
        """
        x = X[1]
        y = X[2]
        z = X[3]
        r=np.sqrt(x**2+y**2+z**2)
        return np.array([
            X[0],
            r,
            np.arccos(z/r),
            np.arctan(y/x)
        ])
    
    def spher_to_cart(self,X: np.array) -> np.array:
        """
        Should also be in geometry.py, not here
        """
        r = X[1]
        sth = np.sin(X[2])
        cth = np.cos(X[2])
        sphi = np.sin(X[3])
        cphi = np.cos(X[3])

        return np.array([
            X[0],
            r * sth * cphi,
            r * sth * sphi,
            r * cth
        ])
    
    def dx_cartesian_dx_spherical(self,X: np.array) -> list:
        """
        dx^mu(cartesian)/dx^mu(spherical) for flatspace
        Parameters:
        X -- x^mu in spherical
        Returns:
        dx^mu(cartesian)/dx^mu(spherical) for flatspace
        """
        r = X[1]
        sth = np.sin(X[2])
        cth = np.cos(X[2])
        sphi = np.sin(X[3])
        cphi = np.cos(X[3])
        return np.array([
            [1,0,0,0],
            [0,sth*cphi,X[1]*cth*cphi,-r*sth*sphi],
            [0,sth*sphi,r*cth*sphi,r*sth*cphi],
            [0,cth,-r*sth,0]
            ])

    def dx_spherical_dx_cartesian(self,Xcart: np.array) -> list:
        """
        dx^mu(spherical)/dx^mu(cartesian) for flatspace
        Parameters:
        Xcart -- x^mu in cartesian
        Returns:
        dx^mu(spherical)/dx^mu(cartesian) for flatspace
        """
        x = Xcart[1]
        y = Xcart[2]
        z = Xcart[3]
        r = np.sqrt(x**2+y**2+z**2)
        return np.array([
            [1,0,0,0],
            [0,x/r,y/r,z/r],
            [0, x*z/(np.sqrt(x**2+y**2)*r**2), y*z/(np.sqrt(x**2+y**2)*r**2), -np.sqrt(x**2+y**2)/r**2],
            [0, -y/(x**2+y**2), x/(x**2+y**2), 0]
        ])

    

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
        jacobi_spher_to_cart  = self.dx_cartesian_dx_spherical(initX)
        # print(jacobi_spher_to_cart)
        initX_cart = self.spher_to_cart(initX)
        initK_cart = np.einsum("ij,j->i",jacobi_spher_to_cart,initK)

        finalX_cart = initX_cart + initK_cart*dl
        finalK_cart = initK_cart
        jacobi_cart_to_spher = self.dx_spherical_dx_cartesian(finalX_cart)
        finalX = self.cart_to_spher(finalX_cart)
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
    print(final_X,final_K)

if __name__ == "__main__":
    test_flat_geodesic()