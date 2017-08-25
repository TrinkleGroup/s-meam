# Note: this imports a generic 1D spline obect that does not have smoothing
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

class Spline(CubicSpline):

    def __init__(self,x,y):
        super(Spline,self).__init__(x,y,bc_type='natural')#,bc_type=((1,d0),(1,dN)))
        self.cutoff = (x[0],x[len(x)-1])
        self.d0 = self(self.x[0],1)
        self._dN = self(self.x[-1],1)

    @property
    def cutoff(self):
        """A tuple of (left cutoff, right cutoff)"""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, tup):
        self._cutoff = tup

    @property
    def d0(self):
        """First derivative at first knot"""
        return self._d0

    @d0.setter
    def d0(self,val):
        self._d0 = val

    @property
    def dN(self):
        """First derivative at last knot"""
        return self._dN

    @dN.setter
    def dN(self,val):
        self._dN = val

    def in_range(self, x):
        """Checks if a given value is within the spline's cutoff range"""

        return (x > self.cutoff[0]) and (x < self.cutoff[1])

    def extrap(self, x):
        """Performs linear extrapolation past the endpoints of the spline"""

        if x < self.cutoff[0]:
            return self(self.x[0]) - self.d0*(self.x[0]-x)
        elif x > self.cutoff[0]:
            return self(self.x[-1]) + self.dN*(x-self.x[-1])

    def plot(self):
        """Plots the spline"""

        low,high = self.cutoff
        low -= abs(0.2*low)
        high += abs(0.2*high)

        x = np.linspace(low,high,1000)
        y = map(lambda e: self(e) if self.in_range(e) else self.extrap(e), x)
        yi =map(lambda e: self(e), self.x)

        plt.plot(self.x, yi, 'o', x, y)
        plt.show()

    #def __call__(self,x,i=None):
    #    if i:
    #        return super(Spline,self).__call__(x,i)

    #    if self.in_range(x):
    #        return super(Spline,self).__call__(x)
    #    else:
    #        if x < self.cutoff[0]:
    #            return self(self.x[0]) - self.d0*(self.x[0]-x)
    #        else:
    #            return 0.0
