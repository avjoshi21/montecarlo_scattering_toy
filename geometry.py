import numpy as np


class Geometry:
    def __init__(self):
        return
    def gcov(self,x):
        return
    def gcon(self,x):
        return np.linalg.inv(self.gcov(x))
    def flipIndex(self,vcon,gcov):
        """flip index high to low or low to high"""
        return np.einsum("ij,j->i",gcov,vcon)

class MinkowskiSpherical(Geometry):
    def gcov(self,x):
        gcov = np.eye(4)
        gcov[0][0] = -1
        gcov[2][2] = x[1]**2
        gcov[3][3] = x[1]**2 * np.sin(x[2])**2
        return gcov

    def cart_to_spher(self,X: np.array) -> np.array:
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
            np.arctan(y/x)
        ])
    
    def spher_to_cart(self,X: np.array) -> np.array:
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
   