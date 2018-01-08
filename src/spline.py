# Note: this imports a generic 1D spline obect that does not have smoothing
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

# TODO: binning for non-grid knot values

class Spline(CubicSpline):

    def __init__(self,x,y,bc_type='natural', derivs=(0,0)):

        super(Spline,self).__init__(x,y,bc_type=bc_type)#,bc_type=((1,d0),(1,dN)))
        self.cutoff = (x[0],x[len(x)-1])
        self.bc_type = bc_type

        self.d0, self.dN = derivs
        self.h = x[1]-x[0]
        self.knotsx = x
        self.knotsy = y

        self.knotsy1 = np.array([super(Spline,self).__call__(z,1) for z in x])
        self.knotsy2 = np.array([super(Spline,self).__call__(z,2) for z in x])

        # Addcoefficients for extrapolating splines; can't edit self.c itself
        self.cmat = np.copy(self.c)
        self.cmat = np.insert(self.c,0,np.array([0,0,self.d0,self.knotsy[
            0]]),axis=1)
        self.cmat = np.insert(self.cmat,self.cmat.shape[1],np.array([0,0,\
                        self.dN, self.knotsy[-1]]),axis=1)

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
                return self.d0
            elif x >= self.cutoff[1]:
                return self.dN
            else:
                return super(Spline,self).__call__(x,i)

        if type(x) == np.ndarray: # use CubicSpline.__call__()
            return super(Spline, self).__call__(x)
        else:
            if self.in_range(x):
                return super(Spline,self).__call__(x)
            else:
                return self.extrap(x)

    # TODO: add a to_matrix() function for matrix form?

class ZeroSpline(Spline):
    """Used to easily create a zero spline"""

    def __init__(self,knotsx):

        knotsx = np.array(knotsx)

        super(ZeroSpline,self).__init__(knotsx,np.zeros(knotsx.shape[0]),\
                derivs=(0,0))
