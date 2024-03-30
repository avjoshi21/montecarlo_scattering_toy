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
        