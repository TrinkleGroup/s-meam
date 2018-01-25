import numpy as np

import logging
logger = logging.getLogger(__name__)

from scipy.sparse import diags

class WorkerSpline:
    """A representation of a cubic spline specifically tailored to meet
    the needs of a Worker object.

    Attributes:
        x (np.arr):
            x-coordinates of knot points

        h (float):
            knot spacing (assumes equally-spaced)

        y (np.arr):
            y-coordinates of knot points; only set by solve_for_derivs()

        y1 (np.arr):
            first derivatives at knot points; only set by solve_for_derivs()

        M (np.arr):
            M = AB where A and B are the matrices from the system of
            equations that comes from matching second derivatives at knot
            points (see build_M() for details)

        cutoff (tuple-like):
            upper and lower bounds of knots (x[0], x[-1])

        index (int):
            the index of the first knot of this spline when the knots of all
            splines in a potential are grouped into a single 1D vector

        end_derivs (tuple-like):
            first derivatives at endpoints

        bc_type (tuple):
            2-element tuple corresponding to boundary conditions at LHS/RHS
            respectively. Options are 'natural' (zero second derivative) and
            'fixed' (fixed first derivative). If 'fixed' on one/both ends,
            self.end_derivatives cannot be None for that end

        struct_vecs (list [np.arr]):
            list of structure vects, where struct_vec[i] is the structure
            vector used to evaluate the function at the i-th derivative.

            each structure vector is a 2D array for evaluating the spline on
            the structure; each row corresponds to a single pair/triplet
            evaluation

    Notes:
        This object is distinct from a spline.Spline since it requires some
        attributes and functionality that a spline.Spline doesn't have."""

    def __init__(self, x, bc_type):

        # Check bad knot coordinates
        if not np.all(x[1:] > x[:-1], axis=0):
            raise ValueError("x must be strictly increasing")

        # Check bad boundary conditions
        if bc_type[0] not in ['natural', 'fixed']:
            raise ValueError("boundary conditions must be one of 'natural' or"
                             "'fixed'")
        if bc_type[1] not in ['natural', 'fixed']:
            raise ValueError("boundary conditions must be one of 'natural' or"
                             "'fixed'")

        # Variables that can be set at beginning
        self.x = x
        self.h = x[1]-x[0]

        self.bc_type = bc_type

        self.cutoff = (x[0], x[-1])
        self.M = build_M(len(x), self.h, self.bc_type)

        # Variables that will be set at some point
        self.index = None
        self.struct_vecs = [[], []]

        # Variables that will be set on evaluation
        self._y = None
        self.y1 = None
        self.end_derivs = None

    def __call__(self, y, deriv=0):

        self.y = y

        z = np.concatenate((self.y, self.y1))

        if self.struct_vecs[deriv] == []:
            return np.array([0.])
        else:
            return self.struct_vecs[deriv] @ z.transpose()

    # self.y made as a property to ensure setting of self.y1 and self.end_derivs
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y, self.end_derivs = np.split(y, [-2])
        self.y1 = self.M @ y.transpose()

    def get_abcd(self, x, deriv=0):
        """Calculates the coefficients needed for spline interpolation.

        Args:
            x (float):
                point at which to evaluate spline

            deriv (int):
                order of derivative to evaluate; default is zero, meaning
                evaluate original function

        Returns:
            vec (np.arr):
                vector of length len(knots)*2; this formatting is used since for
                evaluation, vec will be dotted with a vector consisting of first
                the y-values at the knots, then the y1-values at the knots

        In general, the polynomial p(x) can be interpolated as

            p(x) = A*p_k + B*m_k + C*p_k+1 + D*m_k+1

        where k is the interval that x falls into, p_k and p_k+1 are the
        y-coordinates of the k and k+1 knots, m_k and m_k+1 are the derivatives
        of p(x) at the knots, and the coefficients are defined as:

            A = h_00(t)
            B = h_10(t)(x_k+1 - x_k)
            C = h_01(t)
            D = h_11(t)(x_k+1 - x_k)

        and functions h_ij are defined as:

            h_00 = (1+2t)(1-t)^2
            h_10 = t (1-t)^2
            h_01 = t^2 ( 3-2t)
            h_11 = t^2 (t-1)

            with t = (x-x_k)/(x_k+1 - x_k)"""

        knots = self.x.copy()

        # Find spline interval
        h = knots[1] - knots[0]
        all_k = np.floor((x-knots[0])/h).astype(int)

        nknots = len(knots)
        vec = np.zeros((len(x), 2*nknots))

        for i in range(len(all_k)):  # for every point to be evaluated
            k = all_k[i]

            if k < 0: # LHS extrapolation
                h_00 = np.poly1d([0,0,0,1])
                h_10 = np.poly1d([0,0,1,0])

                t = (x[i] - knots[0])

                h_00 = np.polyder(h_00, deriv)
                h_10 = np.polyder(h_10, deriv)

                A = h_00(t)
                B = h_10(t)

                vec[i][0] = A
                vec[i][nknots] = B

            elif k >= nknots-1: # RHS extrapolation
                k = nknots-1 # centered at second-to-last knot for indexing

                h_00 = np.poly1d([0,0,0,1])
                h_10 = np.poly1d([0,0,1,0])

                t = (x[i] - knots[k])

                h_00 = np.polyder(h_00, deriv)
                h_10 = np.polyder(h_10, deriv)

                A = h_00(t)
                B = h_10(t)

                vec[i][k] = A
                vec[i][-1] = B

            else:
                prefactor  = (knots[k+1] - knots[k])

                h_00 = np.poly1d([2,-3,0,1])
                h_10 = np.poly1d([1,-2,1,0])
                h_01 = np.poly1d([-2,3,0,0])
                h_11 = np.poly1d([1,-1,0,0])

                t = (x[i] - knots[k])/prefactor

                h_00 = np.polyder(h_00, deriv)
                h_10 = np.polyder(h_10, deriv)
                h_01 = np.polyder(h_01, deriv)
                h_11 = np.polyder(h_11, deriv)

                A = h_00(t)
                B = h_10(t)*prefactor
                C = h_01(t)
                D = h_11(t)*prefactor

                vec[i][k] = A
                vec[i][k+nknots] = B
                vec[i][k+1] = C
                vec[i][k+1+nknots] = D

                if deriv == 0:
                    scaling = 1
                else:
                    scaling = 1/(prefactor*deriv)

                vec[i] *= scaling

        return vec

    def add_to_struct_vec(self, val):
        """Builds the ABCD vectors for all elements in val, then adds to
        struct_vec

        Args:
            val (int, float, np.arr, list):
                collection of values to add; converted into a np.arr in this
                function"""

        add = np.atleast_1d(val)

        abcd_0 = self.get_abcd(add, 0)
        abcd_1 = self.get_abcd(add, 1)

        # Add to struct_vec for normal eval
        if self.struct_vecs[0] == []:
            self.struct_vecs[0] = abcd_0
        else:
            self.struct_vecs[0] = np.vstack((self.struct_vecs[0], abcd_0))

        # Add to struct_vec for deriv eval
        if self.struct_vecs[1] == []:
            self.struct_vecs[1] = abcd_1
        else:
            self.struct_vecs[1] = np.vstack((self.struct_vecs[1], abcd_1))

    def plot(self, fname=''):
        raise NotImplementedError("Worker plotting is not ready yet")

        low,high = self.cutoff
        low -= abs(0.2*low)
        high += abs(0.2*high)

        if self.y is None:
            raise ValueError("Must specify y before plotting")

        plt.figure()
        plt.plot(self.x, self.y, 'ro', label='knots')

        tmp_struct = self.struct_vecs
        self.struct_vecs = None

        plot_x = np.linspace(low,high,1000)

        for i in range(len(plot_x)):
            self.add_to_struct_vec(plot_x[i])

        plot_y = self(np.concatenate((self.y, self.end_derivs)))

        self.struct_vecs = tmp_struct

        plt.plot(plot_x, plot_y)
        plt.legend()
        plt.show()

class RhoSpline(WorkerSpline):
    """Special case of a WorkerSpline that is used for rho since it is
    'inside' of the U function and has to keep its per-atom contributions
    separate until it passes the results to U

    Attributes:
        ntypes (int):
            number of atomic types in the system

        struct_vec_dict (list[np.arr]):
            key = original atom id
            value = structure vector corresponding to one atom's neighbors

        natoms (int):
            the number of atoms in the system"""

    def __init__(self, x, bc_type, natoms):
        super(RhoSpline, self).__init__(x, bc_type)

        self.natoms = natoms
        # TODO: could convert to numpy array rather than dict?
        self.struct_vec_dict = {key:[[], []] for key in np.arange(natoms)}

    def compute_for_all(self, y, deriv=0):
        """Computes results for every struct_vec in struct_vec_dict

        Returns:
            ni (np.arr):
                each entry is the set of all ni for a specific atom

            deriv (int):
                which derivative to evaluate"""

        ni = np.zeros(self.natoms)

        for i in self.struct_vec_dict.keys():
            self.struct_vecs = self.struct_vec_dict[i]

            val = super(RhoSpline, self).__call__(y, deriv)
            np.set_printoptions(precision=16)
            #logging.info("WORKER: i, val = {0}, {1}".format(i, val, digits=15))
            ni[i] = np.sum(np.array(super(RhoSpline, self).__call__(y, deriv)))

        return ni

    def update_struct_vec_dict(self, val, atom_id):
        """Updates the structure vector of the <i>th spline of type <type> for a value of r

        Args:
            val (float):
                value to evaluate the spline at

            atom_id (int):
                atom id"""

        add = np.atleast_1d(val)

        abcd_0 = self.get_abcd(add, 0)
        abcd_1 = self.get_abcd(add, 1)

        if self.struct_vec_dict[atom_id][0] == []:
            self.struct_vec_dict[atom_id][0] = abcd_0
        else:
            struct_vec = np.vstack((self.struct_vec_dict[atom_id][0], abcd_0))
            self.struct_vec_dict[atom_id][0] = struct_vec

        if self.struct_vec_dict[atom_id][1] == []:
            self.struct_vec_dict[atom_id][1] = abcd_1
        else:
            struct_vec = np.vstack((self.struct_vec_dict[atom_id][1], abcd_1))
            self.struct_vec_dict[atom_id][1] = struct_vec

class ffgSpline:
    """Spline representation specifically used for building a spline
    representation of the combined funcions f_j*f_k*g_jk used in the
    three-body interactions."""

    def __init__(self, fj, fk, g, natoms):
        """Args:
            fj, fk, g (WorkerSpline):
                fully initialized WorkerSpline objects for each spline

            natoms (int):
                the number of atoms in the system"""

        self.fj = fj
        self.fk = fk
        self.g = g

        # TODO: is this assignment by reference? does it make a NEW spline?
        # does it matter??

        self.natoms = natoms
        # TODO: could convert to numpy array rather than dict?
        self.struct_vec_dict = {key:[[], []] for key in np.arange(natoms)}
        self.struct_vecs = [[], []]

    def __call__(self, y_fj, y_fk, y_g, deriv=0):

        if self.struct_vecs[deriv] == []:
            return 0.

        self.fj.y = y_fj
        fj_vec = np.concatenate((self.fj.y, self.fj.y1))

        self.fk.y = y_fk
        fk_vec = np.concatenate((self.fk.y, self.fk.y1))

        self.g.y = y_g
        g_vec = np.concatenate((self.g.y, self.g.y1))

        self.y = np.prod(cartesian_product(fj_vec, fk_vec, g_vec), axis=1)

        return self.struct_vecs[deriv] @ self.y

    def compute_for_all(self, y_fj, y_fk, y_g, deriv=0):
        """Computes results for every struct_vec in struct_vec_dict

        Returns:
            ni (list [np.arr]):
                each entry is the set of all ni for a specific atom

            deriv (int):
                derivative at which to evaluate the function"""

        ni = np.zeros(self.natoms)

        for i in self.struct_vec_dict.keys():
            self.struct_vecs = self.struct_vec_dict[i]

            val = np.sum(self.__call__(y_fj, y_fk, y_g, deriv))

            ni[i] = val

        return ni

    def get_abcd(self, rij, rik, cos_theta, deriv=0):
        """Ovverrides WorkerSpline.get_abcd()

        Computes the full parameter vector for the multiplication of ffg
        splines

        Args:
            rij (float):
                the value at which to evaluate fj

            fik (float):
                the value at which to evaluate fk

            cos_theta (float):
                the value at which to evaluate g

            deriv (int):
                deriv at which to evaluate the splines"""

        add_rij = np.atleast_1d(rij)
        add_rik = np.atleast_1d(rik)
        add_cos_theta = np.atleast_1d(cos_theta)

        # TODO: using ravel() b/c multi-value is not ready for ffgSpline
        fj_abcd = self.fj.get_abcd(add_rij, deriv)
        fj_abcd = np.ravel(fj_abcd)

        fk_abcd = self.fk.get_abcd(add_rik, deriv)
        fk_abcd = np.ravel(fk_abcd)

        g_abcd = self.g.get_abcd(add_cos_theta, deriv)
        g_abcd = np.ravel(g_abcd)

        full_abcd = np.prod(cartesian_product(fj_abcd, fk_abcd, g_abcd), axis=1)

        return full_abcd

    def update_struct_vec_dict(self, rij, rik, cos_theta, atom_id):
        """Computes the vector of coefficients for evaluating the complete
        product of fj*fk*g

        Args:
            rij (float):
                the value at which to evaluate fj

            rik (float):
                the value at which to evaluate fk

            cos_theta (float):
                the value at which to evaluate g

            atom_id (int):
                atom id"""

        abcd_0 = self.get_abcd(rij, rik, cos_theta, 0)
        abcd_1 = self.get_abcd(rij, rik, cos_theta, 1)

        if self.struct_vec_dict[atom_id][0] == []:
            self.struct_vec_dict[atom_id][0] = abcd_0.reshape((1,len(abcd_0)))
        else:
            struct_vec = np.vstack((self.struct_vec_dict[atom_id][0], abcd_0))
            self.struct_vec_dict[atom_id][0] = struct_vec

        if self.struct_vec_dict[atom_id][1] == []:
            self.struct_vec_dict[atom_id][1] = abcd_1.reshape((1,len(abcd_1)))
        else:
            struct_vec = np.vstack((self.struct_vec_dict[atom_id][1], abcd_1))
            self.struct_vec_dict[atom_id][1] = struct_vec

def build_M(num_x, dx, bc_type):
    """Builds the A and B matrices that are needed to find the function
    derivatives at all knot points. A and B come from the system of equations

        Ap' = Bk

    where p' is the vector of derivatives for the interpolant at each knot
    point and k is the vector of parameters for the spline (y-coordinates of
    knots and second derivatives at endpoints).

    A is a tridiagonal matrix with [h''_10(1), h''_11(1)-h''_10(0), -h''_11(0)]
    on the diagonal which is dx*[2, 8, 2] based on their definitions

    B is a tridiagonal matrix with [-h''_00(1), h''_00(0)-h''_01(1), h''_01(0)]
    on the diagonal which is [-6, 0, 6] based on their definitions

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
        num_x (int):
            the total number of knots

        dx (float):
            knot spacing (assuming uniform spacing)

        bc_type (tuple):
            the type of boundary conditions to be applied to the spline.
            'natural' or 'fixed'

    Returns:
        M (np.arr):
            A^(-1)B"""

    n = num_x - 2

    if n <= 0:
        raise ValueError("the number of knots must be greater than 2")

    # note that values for h''_ij(0) and h''_ij(1) are substituted in
    # TODO: add checks for non-grid x-coordinates
    A = diags(np.array([2,8,2]), [0,1,2], (n,n+2))
    A = A.toarray()

    B = diags([-6, 0, 6], [0,1,2], (n,n+2))
    B = B.toarray()

    bc_lhs, bc_rhs = bc_type
    bc_lhs = bc_lhs.lower()
    bc_rhs = bc_rhs.lower()

    # Determine 1st equation based on LHS boundary condition
    if bc_lhs == 'natural':
        topA = np.zeros(n+2).reshape((1,n+2)); topA[0,0] = -4; topA[0,1] = -2
        topB = np.zeros(n+2).reshape((1,n+2)); topB[0,0] = 6; topB[0,1] = -6
    elif bc_lhs == 'fixed':
        topA = np.zeros(n+2).reshape((1,n+2)); topA[0,0] = 1/dx;
        topB = np.zeros(n+2).reshape((1,n+2));
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' "
                         "or 'fixed'")

    # Determine last equation based on RHS boundary condition
    if bc_rhs == 'natural':
        botA = np.zeros(n+2).reshape((1,n+2)); botA[0,-2] = 2; botA[0,-1] = 4
        botB = np.zeros(n+2).reshape((1,n+2)); botB[0,-2] = -6; botB[0,-1] = 6
    elif bc_rhs == 'fixed':
        botA = np.zeros(n+2).reshape((1,n+2)); botA[0,-1] = 1/dx;
        botB = np.zeros(n+2).reshape((1,n+2));# botB[0,-2] = 6; botB[0,-1] = -6
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' "
                         "or 'fixed'")

    rightB = np.zeros((n+2,2)); rightB[0,0] = rightB[-1,-1] = 1

    # Build matrices
    A = np.concatenate((topA, A), axis=0)
    A = np.concatenate((A, botA), axis=0)

    A *= dx

    B = np.concatenate((topB, B), axis=0)
    B = np.concatenate((B, botB), axis=0)
    B = np.concatenate((B, rightB), axis=1)

    # M = A^(-1)B
    return np.dot(np.linalg.inv(A), B)

def cartesian_product(*arrays):
    """Function for calculating the Cartesian product of any number of input
    arrays.

    Args:
        arrays (np.arr):
            any number of numpy arrays

    Note: this function comes from the stackoverflow user 'senderle' in the
    thread https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/11146645#11146645"""
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a

    return arr.reshape(-1, la)
