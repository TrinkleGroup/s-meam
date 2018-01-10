# Note: this imports a generic 1D spline obect that does not have smoothing
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import numbers

# TODO: binning for non-grid knot values

class Spline(CubicSpline):

    def __init__(self, x=[], y=[], end_derivs=None):
        """Assumes that boundary conditions are either 'natural' on both
        sides, or fixed with given 1st derivatives"""

        if end_derivs is None:
            bc = ('natural', 'natural')
        elif (len(end_derivs)) == 2:
            d0 = end_derivs[0]; dN = end_derivs[1]
            if ((d0=='natural') or (isinstance(d0, numbers.Real))) and\
                    ((dN=='natural') or (isinstance(dN, numbers.Real))):
                bc = ((1,d0), (1,dN))
            else:
                raise ValueError("Invalid boundary condition")
        else:
            raise ValueError("Must specify exactly 2 end derivatives OR leave"
                             "blank for natural boundary conditions")

        super(Spline,self).__init__(x,y,bc_type=bc, extrapolate=True)

        self.cutoff = (x[0],x[len(x)-1])
        self.h = x[1]-x[0]

    def __eq__(self, other):

        x_eq = np.allclose(self.x, other.x)
        y_eq = np.allclose(self(self.x), other(other.x))
        y1_eq = np.allclose(self(self.x,1), other(other.x,1))
        y2_eq = np.allclose(self(self.x,2), other(other.x,2))
        c_eq = (self.cutoff == other.cutoff)
        h_eq = (self.h == other.h)

        return x_eq and y_eq and y1_eq and y2_eq and c_eq and h_eq

    def get_interval(self, x):
        """Extracts the interval corresponding to a given value of x. Assumes
        linear extrapolation outside of knots and fixed knot positions.

        Args:
            x (float):
                the point used to find the spline interval

        Returns:
            interval_num (int):
                index of spline interval
            knot_num (int):
                knot index used for value shifting; LHS knot for internal"""

        h = self.x[1] - self.x[0]

        # Find spline interval; +1 to account for extrapolation
        interval_num = int(np.floor((x-self.x[0])/h)) + 1

        if interval_num <= 0:
            interval_num = 0
            knot_num = 0
        elif interval_num > len(self.x):
            interval_num = len(self.x)
            knot_num = interval_num - 1
        else:
            knot_num = interval_num - 1

        # TODO do you need the knot_num?
        # return interval_num, knot_num
        return interval_num

    def in_range(self, x):
        """Checks if a given value is within the spline's cutoff range"""

        return (x >= self.cutoff[0]) and (x <= self.cutoff[1])

    def plot(self,xr=None,yr=None,xl=None,yl=None,saveName=None):
        """Plots the spline"""

        low,high = self.cutoff
        low -= abs(0.2*low)
        high += abs(0.2*high)

        x = np.linspace(low,high,1000)
        y = self(x)
        yi = self(self.x)

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
                end_derivs=(0.,0.))
