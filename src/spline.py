# Note: this imports a generic 1D spline obect that does not have smoothing
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

# TODO: binning for non-grid knot values

class Spline(CubicSpline):

    def __init__(self, x=[], y=[], end_derivs=None):
        """Assumes that boundary conditions are either 'natural' on both
        sides, or fixed with given 1st derivatives"""

        if end_derivs is None:
            bc = ('natural', 'natural')
        elif len(end_derivs) == 2:
            bc = ((1,end_derivs[0]), (1,end_derivs[1]))
        else:
            raise ValueError("Must specify exactly 2 end derivatives OR leave"
                             "blank for natural boundary conditions")

        super(Spline,self).__init__(x,y,bc_type=bc, extrapolate=True)

        self.d0 = self(x[0], 1)
        self.dN = self(x[-1], 1)
        self.cutoff = (x[0],x[len(x)-1])
        self.h = x[1]-x[0]

    def in_range(self, x):
        """Checks if a given value is within the spline's cutoff range"""

        return (x >= self.cutoff[0]) and (x <= self.cutoff[1])

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

    # TODO: add a to_matrix() function for matrix form?

class ZeroSpline(Spline):
    """Used to easily create a zero spline"""

    def __init__(self, knotsx):

        knotsx = np.array(knotsx)

        super(ZeroSpline, self).__init__(knotsx, np.zeros(knotsx.shape[0]),\
                derivs=(0.,0.))
