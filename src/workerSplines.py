import numpy as np
import logging

from scipy.sparse import diags

from src.numba_functions import jit_add_at_1D, jit_add_at_2D

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

        # self.lhs_extrap_dist = extrap_distance
        # self.rhs_extrap_dist = extrap_distance

        # Variables that will be set at some point
        self.struct_vecs = [[], []]
        self.energy_struct_vec = []
        self.forces_struct_vec = []
        # self.energy_struct_vec = []
        # self.force_struct_vec = [[] for i in range(natoms)]

        # self.indices = []
        self.indices_f = []
        self.indices_b = []
        self.max_num_evals_eng = 1
        self.max_num_evals_fcs = 1

        # Variables that will be set on evaluation
        self._y = None
        self.y1 = None
        self.eval_y = [0]*(2*len(x) + 4)
        self.end_derivs = None
        self.len_x = len(self.x)

        self.zero_abcd = self.get_abcd([0])

    def __call__(self, y, deriv=0):

        self.y = y

        z = [self.y[0] - self.y1[0]*self.lhs_extrap_dist] +\
            self.y.tolist() +\
            [self.y[-1] + self.y1[-1]*self.rhs_extrap_dist, self.y1[0]] +\
            self.y1.tolist() + [self.y1[-1]]

        z = z*self.max_num_evals_eng

        return (self.struct_vecs[deriv] @ z)

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

        return (self.zero_abcd @ z)*self.struct_vecs[0].shape[0]

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

    def get_extrap_range(self, x):
        """Calculates the maximum distance needed for LHS/RHS extrapolation
        given a set of x points

        Args:
            x (np.arr):
                set of points to be checked for extrapolation

        Returns:
            max_lhs_extrap_distance (float):
                maximum distance from any point in x to the leftmost knot

            max_rhs_extrap_distance (float):
                maximum distance from any point in x to the rightmost knot
        """
        return np.min(x) - self.x[0], np.max(x) - self.x[-1]

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
        if x.shape[0] < 1: return np.zeros(self.x.shape)

        # knots = self.x.copy()

        extrap_distances = self.get_extrap_range(x)

        max_lhs_extrap_distance = extrap_distances[0]
        max_rhs_extrap_distance = extrap_distances[1]

        self.lhs_extrap_dist = max(self.extrap_dist,max_lhs_extrap_distance)
        self.rhs_extrap_dist = max(self.extrap_dist,max_rhs_extrap_distance)

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

    def calc_energy(self, y):
        self.y = y

        z = [self.y[0] - self.y1[0]*self.lhs_extrap_dist] +\
            self.y.tolist() +\
            [self.y[-1] + self.y1[-1]*self.rhs_extrap_dist, self.y1[0]] +\
            self.y1.tolist() + [self.y1[-1]]

        z = z*self.max_num_evals_eng

        return np.einsum('ij,j->', self.energy_struct_vec, z)

    def calc_forces(self, y):

        z = [self.y[0] - self.y1[0]*self.lhs_extrap_dist] +\
            self.y.tolist() +\
            [self.y[-1] + self.y1[-1]*self.rhs_extrap_dist, self.y1[0]] +\
            self.y1.tolist() + [self.y1[-1]]

        z = z*self.max_num_evals_fcs

        return np.einsum('ijk,j->ik', self.forces_struct_vec, z)

    def add_to_forces_struct_vec(self, values, dirs, atom_id):
        dirs = np.array(dirs)

        abcd = self.get_abcd(values, 1)

        for a in range(3):
            abcd_3d = np.einsum('ij,i->ij', abcd, dirs[:,a]).ravel()
            if abcd_3d.shape[0] == 286:
                print()
            self.forces_struct_vec[atom_id, :abcd_3d.shape[0], a] = abcd_3d

        # abcd_3d = np.einsum('ij,jk->ikj', dirs, abcd)
        # abcd_3d = abcd_3d.reshape((1, abcd.ravel().shape[0], 3))

        # self.forces_struct_vec[atom_id, :abcd_3d.shape[1], :] = abcd_3d

    def add_to_energy_struct_vec(self, values, atom_id):
        abcd = self.get_abcd(values, 0).ravel()

        self.energy_struct_vec[atom_id, :len(abcd)] = abcd

    def add_to_struct_vec_new(self, values, deriv):
        """
        Adds the values to the atom_id row in the struct vec; assumes added
        in order of atomic id

        Args:
            values: set of values to add (e.g. all rij values)
            deriv: which struct vec to add to

        Returns:
            None; updates instance variable
        """

        if deriv == 0:
            abcd = self.get_abcd(values, 0).ravel()
            diff = (2*len(self.x) + 4)*self.max_num_evals - abcd.shape[0]

            if diff > 0:
                abcd = np.concatenate((abcd, np.zeros(diff)))

        elif deriv == 1:
            abcd_x = self.get_abcd(values[:,0], 1).ravel()
            abcd_y = self.get_abcd(values[:,1], 1).ravel()
            abcd_z = self.get_abcd(values[:,2], 1).ravel()

            diff = (2*len(self.x) + 4)*self.max_num_evals - abcd_x.shape[0]

            if diff > 0:
                abcd_x = np.concatenate((abcd_x, np.zeros(diff)))
                abcd_y = np.concatenate((abcd_y, np.zeros(diff)))
                abcd_z = np.concatenate((abcd_z, np.zeros(diff)))

            abcd = np.vstack((abcd_x, abcd_y, abcd_z))
        else:
            raise ValueError("derivative must be 0 or 1")

        self.struct_vecs[deriv].append(abcd)
        # abcd_1 = self.get_abcd(values, 0).ravel()
        # diff = (2*len(self.x) + 4)*self.max_num_evals - abcd_1.shape[0]
        #
        # if diff > 0:
        #     abcd_1 = np.concatenate((abcd_1, np.zeros(diff)))
        #     # abcd_1 = np.pad(abcd_1, diff, 'constant', constant_values=0)
        #
        # self.struct_vecs[1].append(abcd_1)

    def add_to_struct_vec(self, val, indices):
        """Builds the ABCD vectors for all elements in val, then adds to
        struct_vec

        Args:
            val (int, float, np.arr, list):
                collection of values to add converted into a np.arr in this
                function

            indices (tuple-like):
                index values to append to self.indices
        """
        if len(val) < 1: return

        abcd_0 = self.get_abcd(val, 0)
        abcd_1 = self.get_abcd(val, 1)

        # self.struct_vecs[0] += [abcd_0.squeeze()]
        # self.struct_vecs[1] += [abcd_1.squeeze()]

        self.struct_vecs[0].append(abcd_0.squeeze())
        self.struct_vecs[1].append(abcd_1.squeeze())

        self.struct_vecs[0] = np.vstack(self.struct_vecs[0])
        self.struct_vecs[1] = np.vstack(self.struct_vecs[1])

        # Reshape indices replicate if adding an array of values
        indices_0, indices_1 = zip(*indices)
        indices_0 = list(indices_0)
        indices_1 = list(indices_1)

        # self.indices += indices
        self.indices_f += indices_0
        self.indices_b += indices_1

        # self.indices_f.append(indices_0)
        # self.indices_b.append(indices_1)

    def call_with_args(self, y, new_sv, new_extrap, deriv):
        """Uses the spline to evaluate a structure vector with the given
        extrapolation range without overwriting any necessary class
        variables.

        Args:
            y (np.arr):
                y value parameter vector

            new_sv (np.arr):
                structure vector to be evaluated

            new_extrap (tuple [float]):
                min/max extrapolation values to be used

        Returns:
            values (np.arr):
                results of computation
        """

        tmp_sv = self.struct_vecs[deriv]
        tmp_lhs_extrap = self.lhs_extrap_dist
        tmp_rhs_extrap = self.rhs_extrap_dist

        self.struct_vecs[deriv] = new_sv
        self.lhs_extrap_dist = new_extrap[0]
        self.rhs_extrap_dist = new_extrap[1]

        results = self.__call__(y, deriv)

        self.struct_vecs[deriv] = tmp_sv
        self.lhs_extrap_dist = tmp_lhs_extrap
        self.rhs_extrap_dist = tmp_rhs_extrap

        return results


class RhoSpline(WorkerSpline):
    """Special case of a WorkerSpline that is used for rho since it is
    'inside' of the U function and has to keep its per-atom contributions
    separate until it passes the results to U

    Attributes:
        natoms (int):
            the number of atoms in the system
    """

    def __init__(self, x, bc_type, natoms):
        super(RhoSpline, self).__init__(x, bc_type, natoms)

        self.natoms = natoms

    def compute_for_all(self, y, deriv=0):
        """Computes results for every struct_vec in struct_vec_dict

        Returns:
            ni (np.arr):
                each entry is the set of all ni for a specific atom

            deriv (int):
                which derivative to evaluate

        Note:
            for RhoSplines, indices can be interpreted as indices[0] embedded in
            indices[1]"""

        ni = np.zeros(self.natoms)

        if len(self.struct_vecs[deriv]) == 0: return ni

        results = super(RhoSpline, self).__call__(y, deriv)

        # ni += np.bincount(self.indices_f, weights=results,
        #                   minlength=self.natoms)
        ni += jit_add_at_1D(self.indices_f, results, self.natoms)

        return np.array(ni)


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
        self.fj_struct_vecs = [[], []]
        self.fk_struct_vecs = [[], []]
        self.g_struct_vecs  = [[], []]

        self.fj_extrap_distances = [0., 0.]
        self.fk_extrap_distances = [0., 0.]
        self.g_extrap_distances = [0., 0.]

        self.indices_f = [[], []]
        self.indices_b = [[], []]

    # @profile
    def __call__(self, y_fj, y_fk, y_g, deriv=[0,0,0]):

        if self.fj_struct_vecs[deriv] is None:
            return np.array([0.])

        fj  = self.fj
        fk  = self.fk
        g   = self.g

        fj_results = fj.call_with_args(y_fj, self.fj_struct_vecs[deriv],
                                       self.fj_extrap_distances, deriv)

        fk_results = fk.call_with_args(y_fk, self.fk_struct_vecs[deriv],
                                       self.fk_extrap_distances, deriv)

        g_results = g.call_with_args(y_g, self.g_struct_vecs[deriv],
                                       self.g_extrap_distances, deriv)

        val = np.multiply(np.multiply(fj_results, fk_results), g_results)

        return val.ravel()

    def compute_for_all(self, y_fj, y_fk, y_g, deriv=0):
        """Computes results for every struct_vec in struct_vec_dict

        Returns:
            ni (list [np.arr]):
                each entry is the set of all ni for a specific atom

            deriv (int):
                derivative at which to evaluate the function
        """

        ni = np.zeros(self.natoms)

        if len(self.fj_struct_vecs[deriv]) < 1: return ni

        results = self.__call__(y_fj, y_fk, y_g, deriv)

        indices = self.indices_f[deriv]

        ni += jit_add_at_1D(indices, results, self.natoms)

        return ni

    def get_abcd(self, rij, rik, cos_theta, deriv=0):
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

        # TODO: using ravel() b/c multi-value is not ready for ffgSpline
        fj_abcd = self.fj.get_abcd(add_rij, fj_deriv)
        # fj_abcd = np.ravel(fj_abcd)

        fk_abcd = self.fk.get_abcd(add_rik, fk_deriv)
        # fk_abcd = np.ravel(fk_abcd)

        g_abcd = self.g.get_abcd(add_cos_theta, g_deriv)
        # g_abcd = np.ravel(g_abcd)

        # full_abcd = np.prod(cartesian_product(fj_abcd, fk_abcd, g_abcd), axis=1)

        # return full_abcd
        return fj_abcd.squeeze(), fk_abcd.squeeze(), g_abcd.squeeze()

    def add_to_struct_vec(self, rij, rik, cos_theta, indices):
        """To the first structure vector (direct evaluation), adds one row for
        corresponding to the product of fj*fk*g

        To the second structure vector (first derivative), adds all 6 terms
        associated with the derivative of fj*fk*g (these can be derived by
        using chain/product rule on the derivative of fj*fk*g, keeping in
        mind that they are functions of rij/rik/cos_theta respectively)

        Note that factors (like negatives signs and the additional cos_theta)
        are ignored here, and will be multiplied with the direction later

        Args:
            rij (float):
                the value at which to evaluate fj

            rik (float):
                the value at which to evaluate fk

            cos_theta (float):
                the value at which to evaluate g

            indices (tuple-like):
                [i,j,k] atom tags where i is the center atom, and i and j are
                the neighbors
        """

        if not rij: return

        fj_0, fk_0, g_0 = self.get_abcd(rij, rik, cos_theta, [0, 0, 0])
        fj_1, fk_1, g_1 = self.get_abcd(rij, rik, cos_theta, [1, 0, 0])
        fj_2, fk_2, g_2 = self.get_abcd(rij, rik, cos_theta, [0, 1, 0])
        fj_3, fk_3, g_3 = self.get_abcd(rij, rik, cos_theta, [0, 0, 1])

        self.fj_struct_vecs[0].append(fj_0)
        self.fk_struct_vecs[0].append(fk_0)
        self.g_struct_vecs[0].append(g_0)

        self.fj_struct_vecs[0] = np.vstack(self.fj_struct_vecs[0])
        self.fk_struct_vecs[0] = np.vstack(self.fk_struct_vecs[0])
        self.g_struct_vecs[0] = np.vstack(self.g_struct_vecs[0])

        self.fj_struct_vecs[1]  += [fj_1, fj_3, fj_3, fj_2, fj_3, fj_3]
        self.fk_struct_vecs[1]  += [fk_1, fk_3, fk_3, fk_2, fk_3, fk_3]
        self.g_struct_vecs[1]   += [g_1, g_3, g_3, g_2, g_3, g_3]

        self.fj_struct_vecs[1] = np.vstack(self.fj_struct_vecs[1])
        self.fk_struct_vecs[1] = np.vstack(self.fk_struct_vecs[1])
        self.g_struct_vecs[1] = np.vstack(self.g_struct_vecs[1])

        tmp_indices = np.array(indices)
        ij = tmp_indices[:,[0,1]]
        ik = tmp_indices[:,[0,2]]

        ij = ij.tolist()
        ik = ik.tolist()

        deriv_indices = ij*3 + ik*3

        self.indices_f[0] += [el[0] for el in indices]
        self.indices_b[0] += [el[1] for el in indices]

        self.indices_f[1] += [el[0] for el in deriv_indices]
        self.indices_b[1] += [el[1] for el in deriv_indices]

        self.fj_extrap_distances = self.fj.get_extrap_range(rij)
        self.fk_extrap_distances = self.fj.get_extrap_range(rik)
        self.g_extrap_distances = self.fj.get_extrap_range(cos_theta)


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
