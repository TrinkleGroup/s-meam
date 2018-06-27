import numpy as np
import logging
import h5py

from scipy.sparse import diags, lil_matrix, csr_matrix
from src.numba_functions import onepass_min_max, outer_prod, outer_prod_simple

logger = logging.getLogger(__name__)

class WorkerSpline:
    """A representation of a cubic spline specifically tailored to meet the
    needs of a Worker object. In this implementation, all WorkerSpline
    objects are design to linearly extrapolate to points outside of their
    range.
    """

    def __init__(self, knots, bc_type=None, natoms=0, M=None):
        self.knots = np.array(knots, dtype=float)
        self.n_knots = len(knots)
        self.bc_type = bc_type
        self.natoms = natoms

        """
        Extrapolation is done by building a spline between the end-point
        knot and a 'ghost' knot that is separated by a distance of
        extrap_dist. This distance is updated later if extrapolation must be
        done at an even greater range
        """

        self.extrap_dist = (x[-1] - x[0]) / 2.
        self.lhs_extrap_dist = self.extrap_dist
        self.rhs_extrap_dist = self.extrap_dist

        """
        A 'structure vector' is an array of coefficients defined by the
        Hermitian form of cubic splines that will evaluate a spline for a
        set of points when dotted with a vector of knot y-coordinates and
        boundary conditions. Some important definitions:
        
        M: the matrix corresponding to the system of equations for y'
        alpha: the set of coefficients corresponding to knot y-values
        beta: the set of coefficients corresponding to knot y'-values
        gamma: the matrix product of beta*M (doesn't need to be stored)
        structure vector: the summation of alpha+gamma
        
        Note that the extrapolation structure vectors are NOT of the same 
        form as the rest; they rely on the results of previous y' 
        calculations, whereas the others only depend on y values. In short, 
        they can't be treated the same mathematically
        
        Apologies for the vague naming conventions... see README for explanation
        """

        if M is None:
            self.M = build_M(len(knots), knots[1] - knots[0], bc_type)
        else:
            self.M = M

        self.alphas = {}
        self.alphas['energy'] = np.zeros(self.n_knots + 2)
        self.alphas['forces'] = np.zeros(natoms, self.n_knots + 2, 3)

        self.betas = {}
        self.betas['energy'] = np.zeros(self.n_knots + 2)
        self.betas['forces'] = np.zeros(natoms, self.n_knots + 2, 3)

        self.structure_vectors = {}
        self.structure_vectors['energy'] = np.zeros(self.n_knots + 2)
        self.structure_vectors['forces'] = np.zeros(natoms, self.n_knots + 2, 3)

        self.extrap_struct_vecs = {}

        self.extrap_struct_vecs['energy'] = {}
        self.extrap_struct_vecs['energy']['lhs'] = np.zeros(4)
        self.extrap_struct_vecs['energy']['lhs'] = np.zeros(4)

        self.extrap_struct_vecs['forces'] = {}
        self.extrap_struct_vecs['forces']['lhs'] = np.zeros(natoms, 4, 3)
        self.extrap_struct_vecs['forces']['rhs'] = np.zeros(natoms, 4, 3)

    # def get_abcd(self, x, deriv=0):
    #     """Calculates the spline coefficients for a given evaluation point, x
    #
    #     Args:
    #         x (float): point to be evaluated
    #         deriv (int): optionally compute the 1st derivative at point x
    #
    #     Returns:
    #         alpha: vector of coefficients to be added to alpha
    #         beta: vector of coefficients to be added to betas
    #         lhs_extrap: vector of coefficients to be added to lhs_extrap vector
    #         rhs_extrap: vector of coefficients to be added to rhs_extrap vector
    #     """
    #
    #     self.lhs_extrap_dist = max(self.extrap_dist, self.x[0] - x)
    #     self.rhs_extrap_dist = max(self.extrap_dist, x - self.x[-1])
    #
    #     # add ghost knots
    #     knots = [self.knots[0] - self.lhs_extrap_dist] + self.knots.tolist() + \
    #             [self.knots[-1] + self.rhs_extrap_dist]
    #
    #     knots = np.array(knots)
    #
    #     # indicates the splines that the points fall into
    #     k = np.digitize(x, knots) - 1
    #     k = np.clip(k, 0, len(knots) - 2)
    #
    #     prefactor = knots[k + 1] - knots[k]
    #
    #     t = (x - knots[k]) / prefactor
    #     t2 = t*t
    #     t3 = t2*t
    #
    #     if deriv == 0:
    #         scaling = 1
    #
    #         A = 2*t3 - 3*t2 + 1
    #         B = t3 - 2*t2 + t
    #         C = -2*t3 + 3*t2
    #         D = t3 - t2
    #
    #     elif deriv == 1:
    #         scaling =  1 / (prefactor * deriv)
    #
    #         A = 6*t2 - 6*t
    #         B = 3*t2 - 4*t + 1
    #         C = -6*t2 + 6*t
    #         D = 3*t2 - 2*t
    #
    #     else:
    #         raise ValueError("Only allowed derivative values are 0 and 1")
    #
    #     B *= prefactor
    #     D *= prefactor
    #
    #     alpha = np.zeros(self.n_knots + 2)
    #     betas = np.zeros(self.n_knots)
    #     lhs_extrap = np.zeros(4)
    #     rhs_extrap = np.zeros(4)
    #
    #     if k == 0:
    #         lhs_extrap = np.array([A, B, C, D])
    #         elif k =

def build_M(num_x, dx, bc_type):
    """Builds the A and B matrices that are needed to find the function
    derivatives at all knot points. A and B come from the system of equations
    that comes from matching second derivatives at internal spline knots
    (using Hermitian cubic splines) and specifying boundary conditions

        Ap' = Bk

    where p' is the vector of derivatives for the interpolant at each knot
    point and k is the vector of parameters for the spline (y-coordinates of
    knots and second derivatives at endpoints).

    Let N be the number of knot points

    In addition to N equations from internal knots and 2 equations from boundary
    conditions, there are an additional 2 equations for requiring linear
    extrapolation outside of the spline range. Linear extrapolation is
    achieved by specifying a spline who's first derivatives match at each end
    and whose endpoints lie in a line with that derivative.

    With these specifications, A and B are both (N+2, N+2) matrices

    A's core is a tridiagonal matrix with [h''_10(1), h''_11(1)-h''_10(0),
    -h''_11(0)] on the diagonal which is dx*[2, 8, 2] based on their definitions

    B's core is tridiagonal matrix with [-h''_00(1), h''_00(0)-h''_01(1),
    h''_01(0)] on the diagonal which is [-6, 0, 6] based on their definitions

    Note that the dx is a scaling factor defined as dx = x_k+1 - x_k, assuming
    uniform grid points and is needed to correct for the change into the
    variable t, defined below.

    and functions h_ij are defined as:

        h_00 = (1+2t)(1-t)^2
        h_10 = t (1-t)^2
        h_01 = t^2 (3-2t)
        h_11 = t^2 (t-1)

        with t = (x-x_k)/dx

    which means that the h''_ij functions are:

        h''_00 = 12t - 6
        h''_10 = 6t - 4
        h''_01 = -12t + 6
        h''_11 = 6t - 2

    Args:
        num_x (int): the total number of knots

        dx (float): knot spacing (assuming uniform spacing)

        bc_type (tuple): tuple of 'natural' or 'fixed'

    Returns:
        M (np.arr):
            A^(-1)B
    """

    n = num_x - 2

    if n <= 0:
        raise ValueError("the number of knots must be greater than 2")

    # note that values for h''_ij(0) and h''_ij(1) are substituted in
    # TODO: add checks for non-grid x-coordinates

    bc_lhs, bc_rhs = bc_type
    bc_lhs = bc_lhs.lower()
    bc_rhs = bc_rhs.lower()

    A = np.zeros((n + 4, n + 4))
    B = np.zeros((n + 4, n + 4))

    # match 2nd deriv for internal knots
    fillA = diags(np.array([2, 8, 2]), [0, 1, 2], (n, n + 2))
    fillB = diags([-6, 0, 6], [0, 1, 2], (n, n + 2))
    A[2:n+2, 1:n+3] = fillA.toarray()
    B[2:n+2, :n+2] = fillB.toarray()

    # add remaining equations
    A[0,0] = 1/dx; A[0,1] = -1/dx # lhs ghost deriv matching lhs knot deriv
    A[-1,-1] = 1/dx; A[-1,-2] = -1/dx # rhs ghost deriv matching rhs knot deriv

    # equation accounting for lhs bc
    if bc_lhs == 'natural':
        A[1,1] = -4; A[1,2] = -2
        B[1,0] = 6; B[1,1] = -6; B[1,-2] = 1
    elif bc_lhs == 'fixed':
        A[1,1] = 1/dx;
        B[1,-2] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    # equation accounting for rhs bc
    if bc_rhs == 'natural':
        A[-2,-3] = 2; A[-2,-2] = 4
        B[-2,-4] = -6; B[-2,-3] = 6; B[-2,-1] = 1
    elif bc_rhs == 'fixed':
        A[1,-2] = 1/dx
        B[1,-1] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    A *= dx

    # M = A^(-1)B
    return np.dot(np.linalg.inv(A), B)