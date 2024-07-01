import numpy as np
import utils

class Geometry:
    def __init__(self):
        return
    def gcov(self,x):
        """return gcov (g_munu), setting to Minkowski cartesian as default"""
        identity = np.eye(4)
        identity[0][0]=-1
        return identity
    def gcon(self,x):
        """compute g^munu at x^mu"""
        return np.linalg.inv(self.gcov(x))
    def flip_index(self,vcon,gcov):
        """flip index high to low or low to high"""
        return gcov @ vcon
    def inner(self,kcon,lcon,gcov):
        """compute inner product k^mu g_munu l^nu at x^mu"""
        # gcov = self.gcov(x)
        lcov = self.flip_index(lcon,gcov)
        return np.inner(kcon,lcov)

    
class MinkowskiSpherical(Geometry):
    def gcov(self,x):
        gcov = np.eye(4)
        gcov[0][0] = -1
        gcov[2][2] = x[1]**2
        gcov[3][3] = x[1]**2 * np.sin(x[2])**2
        return gcov
    
    def gcon(self,x):
        return np.linalg.inv(self.gcov(x))

    # @utils.operate_along_axis_member_function
    def cart_to_spher(self,X: np.array,axis=None) -> np.array:
        """
        Should this be in a coordinates.py?
        Parameters:
        X -- position four-vector in cartesian
        Returns:
        Position four-vector in spherical
        """
        x = X[1]
        y = X[2]
        z = X[3]
        r=np.sqrt(x**2+y**2+z**2)
        return np.array([
            X[0],
            r,
            np.arccos(z/r),
            np.arctan2(y,x)
        ])
    
    # @utils.operate_along_axis_member_function
    def spher_to_cart(self,X: np.array, axis=None) -> np.array:
        """
        Should be in coordinates.py?
        Parameters:
        X -- position four-vector in spherical
        Returns:
        Position four-vector in cartesian
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
    
    # @utils.operate_along_axis_member_function
    def dx_cartesian_dx_spherical(self,X: np.array,axis=None) -> np.array:
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
    
    # @utils.operate_along_axis_member_function
    def dx_spherical_dx_cartesian(self,Xcart: np.array, axis=None) -> np.array:
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
    
    def vector_cart_to_spher(self,Xcart,Kcart):
        """
        Transforms vector from cartesian to spherical
        """
        return np.einsum("ij,j->i",self.dx_spherical_dx_cartesian(Xcart),Kcart)

    def vector_spher_to_cart(self,Xspher,Kspher):
        """
        Transforms vector from cartesian to spherical
        """
        return np.einsum("ij,j->i",self.dx_cartesian_dx_spherical(Xspher),Kspher)



def test_coord_transformations():
    flat_geom = MinkowskiSpherical()

    X_spher = np.array([1,45.4211,0.6,2])
    X_cart = flat_geom.spher_to_cart(X_spher)

    a = flat_geom.dx_cartesian_dx_spherical(X_spher)
    b = flat_geom.dx_spherical_dx_cartesian(X_cart)
    print(a,b)
    dx_mults = np.einsum("ij,jk->ik",a,b)
    print(dx_mults)

if __name__ == "__main__":
    test_coord_transformations()