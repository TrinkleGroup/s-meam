import numpy as np
import logging

from scipy.sparse import diags, dok_matrix

# from src.numba_functions import jit_add_at_1D, jit_add_at_2D

logger = logging.getLogger(__name__)


class WorkerSpline:
    """A representation of a cubic spline specifically tailored to meet
    the needs of a Worker object.

    Attributes:
        x (np.arr):
            x-coordinates of knot points

        h (float):
            knot spacing (assumes equally-spaced)

        y (np.arr):
            y-coordinates of knot points only set by solve_for_derivs()

        y1 (np.arr):
            first derivatives at knot points only set by solve_for_derivs()

        M (np.arr):
            M = AB where A and B are the matrices from the system of
            equations that comes from matching second derivatives at knot
            points (see build_M() for details)

        cutoff (tuple-like):
            upper and lower bounds of knots (x[0], x[-1])

        end_derivs (tuple-like):
            first derivatives at endpoints

        bc_type (tuple):
            2-element tuple corresponding to boundary conditions at LHS/RHS
            respectively. Options are 'natural' (zero second derivative) and
            'fixed' (fixed first derivative). If 'fixed' on one/both ends,
            self.end_derivatives cannot be None for that end

        struct_vecs (list [list]):
            list of structure vects, where struct_vec[i] is the structure
            vector used to evaluate the function at the i-th derivative.

            each structure vector is a 2D list for evaluating the spline on
            the structure each row corresponds to a single pair/triplet
            evaluation. Converted to NumPy array for calculations

        indices[_f,_b] (list [tuple-like]):
            indices for matching values to atoms needed for force
            calculations, which require per-atom grouping. Tuple needed to do
            forwards/backwards directions. For U, this is just a single ID.
            The _f or _b denotes forwards/backwards directions

    Notes:
        This object is distinct from a spline.Spline since it requires some
        attributes and functionality that a spline.Spline doesn't have.

        By default, splines are designed to extrapolate accurately to double
        the effective range (half of the original range on each side)

        It is assumed that the U potential will modify the extrapolation
        parameters outside of the WorkerSpline based on calculated ni values
    """

    def __init__(self, x, bc_type, natoms):

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
        self.x = np.array(x, dtype=float)
        self.h = x[1] - x[0]

        self.bc_type = bc_type

        self.cutoff = (x[0], x[-1])
        self.M = build_M(len(x), self.h, self.bc_type)

        self.natoms = natoms

        self.extrap_dist = (self.cutoff[1] - self.cutoff[0]) / 2.

        self.lhs_extrap_dist = self.extrap_dist
        self.rhs_extrap_dist = self.extrap_dist

        # Variables that will be set at some point
        self.energy_struct_vec = np.zeros(2*len(x)+4)
        self.forces_struct_vec = np.zeros((natoms, 2*len(x)+4, 3))

        # Variables that will be set on evaluation
        self._y = None
        self.y1 = None
        self.eval_y = [0]*(2*len(x) + 4)
        self.end_derivs = None
        self.len_x = len(self.x)

    @property
    def y(self):
        """Made as a property to ensure setting of self.y1 and
        self.end_derivs
        """
        return self._y

    @y.setter
    def y(self, y):
        self._y = y[:-2]; self.end_derivs = y[-2:]
        self.y1 = (self.M @ y)

    # @profile
    def get_abcd(self, x, deriv=0):
        """Calculates the coefficients needed for spline interpolation.

        Args:
            x (ndarray):
                point at which to evaluate spline

            deriv (int):
                order of derivative to evaluate default is zero, meaning
                evaluate original function

        Returns:
            vec (np.arr):
                vector of length len(knots)*2 this formatting is used since for
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

            with t = (x-x_k)/(x_k+1 - x_k)
        """

        x = np.atleast_1d(x)
        if x.shape[0] < 1:
            n_eval = 2*len(self.x) + 4
            return np.zeros(n_eval).reshape((1, n_eval))

        self.lhs_extrap_dist = max(self.extrap_dist, np.min(x) - self.x[0])
        self.rhs_extrap_dist = max(self.extrap_dist, np.max(x) - self.x[-1])

        # add ghost knots
        knots = [self.x[0] - self.lhs_extrap_dist] + list(self.x) + [self.x[-1]\
                 + self.rhs_extrap_dist]
        knots = np.array(knots)

        nknots = len(knots)

        # Perform interval search and prepare prefactors
        all_k = np.digitize(x, knots, right=True) - 1

        prefactors = knots[all_k + 1] - knots[all_k]

        all_t = (x - knots[all_k]) / prefactors

        t = all_t
        t2 = all_t*all_t
        t3 = t2*all_t

        if deriv == 0:
            scaling = np.ones(len(prefactors))

            all_A = 2*t3 - 3*t2 + 1
            all_B = t3 - 2*t2 + t
            all_C = -2*t3 + 3*t2
            all_D = t3 - t2

        elif deriv == 1:
            scaling =  1 / (prefactors * deriv)

            all_A = 6*t2 - 6*t
            all_B = 3*t2 - 4*t + 1
            all_C = -6*t2 + 6*t
            all_D = 3*t2 - 2*t

        else:
            raise ValueError("Only allowed derivative values are 0 and 1")

        all_B *= prefactors
        all_D *= prefactors

        vec = np.zeros((len(x), 2*(nknots)))

        tmp_indices = np.arange(len(vec))

        vec[tmp_indices, all_k] += all_A
        vec[tmp_indices, all_k + nknots] += all_B
        vec[tmp_indices, all_k + 1] += all_C
        vec[tmp_indices, all_k + 1 + nknots] += all_D

        scaling = scaling.reshape((len(scaling), 1))

        vec *= scaling

        return vec

    # @profile
    def calc_energy(self, y):
        self.y = y

        z = [self.y[0] - self.y1[0]*self.lhs_extrap_dist] +\
            self.y.tolist() +\
            [self.y[-1] + self.y1[-1]*self.rhs_extrap_dist, self.y1[0]] +\
            self.y1.tolist() + [self.y1[-1]]

        z = z

        return self.energy_struct_vec @ z

    def calc_forces(self, y):
        self.y = y

        z = [self.y[0] - self.y1[0]*self.lhs_extrap_dist] +\
            self.y.tolist() +\
            [self.y[-1] + self.y1[-1]*self.rhs_extrap_dist, self.y1[0]] +\
            self.y1.tolist() + [self.y1[-1]]

        z = z

        return np.einsum('ijk,j->ik', self.forces_struct_vec, z)

    def add_to_forces_struct_vec(self, values, dirs, atom_id):
        dirs = np.array(dirs)

        abcd = self.get_abcd(values, 1)

        for a in range(3):
            abcd_3d = np.einsum('ij,i->ij', abcd, dirs[:,a])

            self.forces_struct_vec[atom_id, :, a] += np.sum(abcd_3d, axis=0)

    def add_to_energy_struct_vec(self, values):
        self.energy_struct_vec += np.sum(self.get_abcd(values, 0), axis=0)


class USpline(WorkerSpline):

    def __init__(self, x, bc_type, natoms):
        super(USpline, self).__init__(x, bc_type, natoms)

        self.deriv_struct_vec = np.zeros((natoms, 2*len(self.x)+4))

        self.zero_abcd = self.get_abcd([0])

        self.atoms_embedded = 0

    def add_to_energy_struct_vec(self, values):
        values = np.atleast_1d(values)

        super(USpline, self).add_to_energy_struct_vec(values)

        self.atoms_embedded += values.shape[0]

    def add_to_deriv_struct_vec(self, ni, atom_id):
        abcd = self.get_abcd(ni, 1)

        self.deriv_struct_vec[atom_id, :] = np.sum(abcd, axis=0)

    def calc_deriv(self, y):
        self.y = y

        z = [self.y[0] - self.y1[0]*self.lhs_extrap_dist] +\
            self.y.tolist() +\
            [self.y[-1] + self.y1[-1]*self.rhs_extrap_dist, self.y1[0]] +\
            self.y1.tolist() + [self.y1[-1]]

        z = z

        return self.deriv_struct_vec @ z

    def compute_zero_potential(self, y):
        """Calculates the value of the potential as if every entry in the
        structure vector was a zero.

        Args:
            y (np.arr):
                array of parameter vectors

        Returns:
            the value evaluated by the spline using num_zeros zeros"""

        y1 = self.M @ y.transpose()

        y = y[:-2]

        z = [0] + y.tolist() + [0, 0] + y1.tolist() + [0]

        return (self.zero_abcd @ z)*self.atoms_embedded


class RhoSpline(WorkerSpline):

    def __init__(self, x, bc_type, natoms):
        super(RhoSpline, self).__init__(x, bc_type, natoms)

        self.energy_struct_vec = np.zeros((self.natoms, 2*len(x)+4))

        self.forces_struct_vec = np.zeros((self.natoms, self.natoms,
                                           2*len(x)+4, 3))

    def calc_forces(self, y):
        self.y = y

        z = [self.y[0] - self.y1[0]*self.lhs_extrap_dist] +\
            self.y.tolist() +\
            [self.y[-1] + self.y1[-1]*self.rhs_extrap_dist, self.y1[0]] +\
            self.y1.tolist() + [self.y1[-1]]

        z = z

        return np.einsum('ijkl,k->ijl', self.forces_struct_vec, z)

    def add_to_energy_struct_vec(self, values, atom_id):
        self.energy_struct_vec[atom_id, :] += np.sum(self.get_abcd(values,0),
                                                     axis=0)

    def add_to_forces_struct_vec(self, value, dir, atom_id, neighbor_id):
        """Single add for neighbor"""
        abcd_3d = np.einsum('i,j->ij', self.get_abcd(value, 1).ravel(), dir)

        self.forces_struct_vec[atom_id, neighbor_id, :, :] += abcd_3d


class ffgSpline:
    """Spline representation specifically used for building a spline
    representation of the combined funcions f_j*f_k*g_jk used in the
    three-body interactions.
    """

    def __init__(self, fj, fk, g, natoms):
        """Args:
            fj, fk, g (WorkerSpline):
                fully initialized WorkerSpline objects for each spline

            natoms (int):
                the number of atoms in the system
        """

        self.fj = fj
        self.fk = fk
        self.g = g

        self.natoms = natoms

        # Variables that will be set at some point
        nknots_cartesian = (len(fj.x)*2+4)*(len(fk.x)*2+4)*(len(g.x)*2+4)

        self.energy_struct_vec = np.zeros((natoms, nknots_cartesian))

        sv_x = self.energy_struct_vec.copy()
        sv_y = self.energy_struct_vec.copy()
        sv_z = self.energy_struct_vec.copy()

        self.forces_struct_vec = [sv_x, sv_y, sv_z]
        # self.forces_struct_vec_j = self.forces_struct_vec_i.copy()

    # @profile
    def calc_energy(self, y_fj, y_fk, y_g):
        fj = self.fj
        fk = self.fk
        g = self.g

        fj.y = y_fj
        fk.y = y_fk
        g.y = y_g

        z_fj = [fj.y[0] - fj.y1[0]*fj.lhs_extrap_dist] + fj.y.tolist() +\
               [fj.y[-1] + fj.y1[-1]*fj.rhs_extrap_dist, fj.y1[0]] +\
               fj.y1.tolist() + [fj.y1[-1]]

        z_fk = [fk.y[0] - fk.y1[0]*fk.lhs_extrap_dist] + fk.y.tolist() +\
               [fk.y[-1] + fk.y1[-1]*fk.rhs_extrap_dist, fk.y1[0]] +\
               fk.y1.tolist() + [fk.y1[-1]]

        z_g = [g.y[0] - g.y1[0]*g.lhs_extrap_dist] + g.y.tolist() +\
               [g.y[-1] + g.y1[-1]*g.rhs_extrap_dist, g.y1[0]] +\
               g.y1.tolist() + [g.y1[-1]]

        z_fj = np.array(z_fj)
        z_fk = np.array(z_fk)
        z_g = np.array(z_g)

        # z_cart = np.product(cartesian_product(z_fj, z_fk, z_g), axis=1)
        z_cart = np.outer(np.outer(z_fj, z_fk), z_g).ravel()

        return self.energy_struct_vec @ z_cart

    def calc_forces(self, y_fj, y_fk, y_g):

        fj = self.fj
        fk = self.fk
        g = self.g

        fj.y = y_fj
        fk.y = y_fk
        g.y = y_g

        z_fj = [fj.y[0] - fj.y1[0]*fj.lhs_extrap_dist] + fj.y.tolist() +\
               [fj.y[-1] + fj.y1[-1]*fj.rhs_extrap_dist, fj.y1[0]] +\
               fj.y1.tolist() + [fj.y1[-1]]

        z_fk = [fk.y[0] - fk.y1[0]*fk.lhs_extrap_dist] + fk.y.tolist() +\
               [fk.y[-1] + fk.y1[-1]*fk.rhs_extrap_dist, fk.y1[0]] +\
               fk.y1.tolist() + [fk.y1[-1]]

        z_g = [g.y[0] - g.y1[0]*g.lhs_extrap_dist] + g.y.tolist() +\
               [g.y[-1] + g.y1[-1]*g.rhs_extrap_dist, g.y1[0]] +\
               g.y1.tolist() + [g.y1[-1]]

        z_fj = np.array(z_fj)
        z_fk = np.array(z_fk)
        z_g = np.array(z_g)

        # z_cart = np.product(cartesian_product(z_fj, z_fk, z_g), axis=1)
        z_cart = np.outer(np.outer(z_fj, z_fk), z_g).ravel()

        results = np.zeros((self.natoms, 3))

        results[:,0] += self.forces_struct_vec[0] @ z_cart
        results[:,1] += self.forces_struct_vec[1] @ z_cart
        results[:,2] += self.forces_struct_vec[2] @ z_cart

        # logging.info("WORKER: results =\n{0}".format(results))

        return results

    def add_to_energy_struct_vec(self, rij, rik, cos, atom_id):
        """Updates structure vectors with given values"""
        abcd_fj = self.fj.get_abcd(rij, 0).ravel()
        abcd_fk = self.fk.get_abcd(rik, 0).ravel()
        abcd_g = self.g.get_abcd(cos, 0).ravel()

        # cart = cartesian_product(abcd_fj, abcd_fk, abcd_g)
        cart = np.outer(np.outer(abcd_fj, abcd_fk), abcd_g).ravel()

        # self.energy_struct_vec[atom_id, :] += np.product(cart, axis=1)
        self.energy_struct_vec[atom_id, :] += cart

    def add_to_forces_struct_vec(self, rij, rik, cos, dirs, atom_id):
        """Adds all 6 directional information to the struct vector for the
        given triplet values

        Args:
            rij : single rij value
            rik : single rik value
            cos : single cos_theta value
            dirs : ordered list of the six terms of the triplet deriv directions
        """

        fj_1, fk_1, g_1 = self.get_abcd(rij, rik, cos, [1, 0, 0])
        fj_2, fk_2, g_2 = self.get_abcd(rij, rik, cos, [0, 1, 0])
        fj_3, fk_3, g_3 = self.get_abcd(rij, rik, cos, [0, 0, 1])

        # v1 = np.product(cartesian_product(fj_1, fk_1, g_1), axis=1)
        # v2 = np.product(cartesian_product(fj_2, fk_2, g_2), axis=1)
        # v3 = np.product(cartesian_product(fj_3, fk_3, g_3), axis=1)

        v1 = np.outer(np.outer(fj_1, fk_1), g_1).ravel()
        v2 = np.outer(np.outer(fj_2, fk_2), g_2).ravel()
        v3 = np.outer(np.outer(fj_3, fk_3), g_3).ravel()

        for a in range(3):

            self.forces_struct_vec[a][atom_id, :] += v1*dirs[0][a]
            self.forces_struct_vec[a][atom_id, :] += v3*dirs[1][a]
            self.forces_struct_vec[a][atom_id, :] += v3*dirs[2][a]
            self.forces_struct_vec[a][atom_id, :] += v2*dirs[3][a]
            self.forces_struct_vec[a][atom_id, :] += v3*dirs[4][a]
            self.forces_struct_vec[a][atom_id, :] += v3*dirs[5][a]

    def get_abcd(self, rij, rik, cos_theta, deriv=[0,0,0]):
        """Computes the full parameter vector for the multiplication of ffg
        splines

        Args:
            rij (float):
                the value at which to evaluate fj

            rik (float):
                the value at which to evaluate fk

            cos_theta (float):
                the value at which to evaluate g

            deriv (list):
                derivatives at which to evaluate the splines, ordered as [
                fj_deriv, fk_deriv, g_deriv]
        """

        fj_deriv, fk_deriv, g_deriv = deriv

        add_rij = np.atleast_1d(rij)
        add_rik = np.atleast_1d(rik)
        add_cos_theta = np.atleast_1d(cos_theta)

        fj_abcd = self.fj.get_abcd(add_rij, fj_deriv)
        fk_abcd = self.fk.get_abcd(add_rik, fk_deriv)
        g_abcd = self.g.get_abcd(add_cos_theta, g_deriv)

        return fj_abcd.squeeze(), fk_abcd.squeeze(), g_abcd.squeeze()


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
            A^(-1)B
    """

    n = num_x - 2

    if n <= 0:
        raise ValueError("the number of knots must be greater than 2")

    # note that values for h''_ij(0) and h''_ij(1) are substituted in
    # TODO: add checks for non-grid x-coordinates
    A = diags(np.array([2, 8, 2]), [0, 1, 2], (n, n + 2))
    A = A.toarray()

    B = diags([-6, 0, 6], [0, 1, 2], (n, n + 2))
    B = B.toarray()

    bc_lhs, bc_rhs = bc_type
    bc_lhs = bc_lhs.lower()
    bc_rhs = bc_rhs.lower()

    # Determine 1st equation based on LHS boundary condition
    if bc_lhs == 'natural':
        topA = np.zeros(n + 2).reshape((1, n + 2))
        topA[0, 0] = -4
        topA[0, 1] = -2
        topB = np.zeros(n + 2).reshape((1, n + 2))
        topB[0, 0] = 6
        topB[0, 1] = -6
    elif bc_lhs == 'fixed':
        topA = np.zeros(n + 2).reshape((1, n + 2))
        topA[0, 0] = 1 / dx
        topB = np.zeros(n + 2).reshape((1, n + 2))
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' "
                         "or 'fixed'")

    # Determine last equation based on RHS boundary condition
    if bc_rhs == 'natural':
        botA = np.zeros(n + 2).reshape((1, n + 2))
        botA[0, -2] = 2
        botA[0, -1] = 4
        botB = np.zeros(n + 2).reshape((1, n + 2))
        botB[0, -2] = -6
        botB[0, -1] = 6
    elif bc_rhs == 'fixed':
        botA = np.zeros(n + 2).reshape((1, n + 2))
        botA[0, -1] = 1 / dx
        botB = np.zeros(n + 2).reshape(
            (1, n + 2))  # botB[0,-2] = 6 botB[0,-1] = -6
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' "
                         "or 'fixed'")

    rightB = np.zeros((n + 2, 2))
    rightB[0, 0] = rightB[-1, -1] = 1

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
    thread https://stackoverflow.com/questions/11144513/numpy-cartesian-product
    -of-x-and-y-array-points-into-single-array-of-2d-points/11146645#11146645
    """

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a

    return arr.reshape(-1, la)
