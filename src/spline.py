# Note: this imports a generic 1D spline obect that does not have smoothing
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

# TODO: binning for non-grid knot values

class Spline(CubicSpline):

    def __init__(self,x,y,bc_type='natural', end_derivs=(0,0)):

        super(Spline,self).__init__(x,y,bc_type=bc_type)#,bc_type=((1,d0),(1,dN)))
        self.cutoff = (x[0],x[len(x)-1])

        self.d0, self.dN = end_derivs
        self.h = x[1]-x[0]

    def __eq__(self, other):
        x_eq = np.allclose(self.x, other.x)
        d_eq = ((self.d0 == other.d0) and (self.dN == other.dN))

        return x_eq and d_eq

    def in_range(self, x):
        """Checks if a given value is within the spline's cutoff range"""

        return (x >= self.cutoff[0]) and (x <= self.cutoff[1])

    def extrap(self, x):
        """Performs linear extrapolation past the endpoints of the spline"""

        if x < self.cutoff[0]:
            return self(self.x[0]) - self.d0*(self.x[0]-x)
        elif x > self.cutoff[1]:
            return self(self.x[-1]) + self.dN*(x-self.x[-1])

    def plot(self,xr=None,yr=None,xl=None,yl=None,saveName=None):
        """Plots the spline"""

        low,high = self.cutoff
        low -= abs(0.2*low)
        high += abs(0.2*high)

        x = np.linspace(low,high,1000)
        y = list(map(lambda e: self(e) if self.in_range(e) else self.extrap(
            e), x))
        yi = list(map(lambda e: self(e), self.x))

        plt.figure()
        plt.plot(self.x, yi, 'o', x, y)

        if xr: plt.xlim(xr)
        if yr: plt.ylim(yr)
        if xl: plt.xlabel(xl)
        if yl: plt.ylabel(yl)

        if saveName: plt.savefig(saveName)
        else: plt.show()

    def __call__(self,x,i=None):
        """Evaluates the spline at the given point, linearly extrapolating if
        outside of the spline cutoff. If 'i' is specified, evaluates the ith
        derivative instead."""

        if i:
            if x <= self.cutoff[0]:
                return super(Spline,self).__call__(self.x[0],i)
            elif x >= self.cutoff[1]:
                return super(Spline,self).__call__(self.x[-1],i)
            else:
                return super(Spline,self).__call__(x,i)

        if self.in_range(x):
            return super(Spline,self).__call__(x)
        else:
            return self.extrap(x)
            #return super(Spline,self).__call__(x,extrapolate=True)

    # TODO: add a to_matrix() function for matrix form?

class ZeroSpline(Spline):
    """Used to easily create a zero spline"""

    def __init__(self,knotsx):

        knotsx = np.array(knotsx)

        super(ZeroSpline,self).__init__(knotsx,np.zeros(knotsx.shape[0]),\
                bc_type=((1,0),(1,0)), end_derivs=(0,0))
