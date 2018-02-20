import numpy as np
import logging

from scipy.sparse import diags

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

        indices (list [tuple-like]):
            indices for matching values to atoms needed for force
            calculations, which require per-atom grouping. Tuple needed to do
            forwards/backwards directions. For U, this is just a single ID

    Notes:
        This object is distinct from a spline.Spline since it requires some
        attributes and functionality that a spline.Spline doesn't have.

        By default, splines are designed to extrapolate accurately to double
        the effective range (half of the original range on each side)

        It is assumed that the U potential will modify the extrapolation
        parameters outside of the WorkerSpline based on calculated ni values
    """

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
        self.x = np.array(x, dtype=float)
        self.h = x[1] - x[0]

        self.bc_type = bc_type

        self.cutoff = (x[0], x[-1])
        self.M = build_M(len(x), self.h, self.bc_type)

        extrap_distance = (self.cutoff[1] - self.cutoff[0])/2.

        self.lhs_extrap_dist = extrap_distance
        self.rhs_extrap_dist = extrap_distance

        # Variables that will be set at some point
        self.struct_vecs = [[], []]
        self.indices = []

        # Variables that will be set on evaluation
        self._y = None
        self.y1 = None
        self.end_derivs = None

    def __call__(self, y, deriv=0):

        self.y = y


        if self.struct_vecs[deriv]:

            # default extrapolation distance is half of full spline range
            lhs_extrap_y = np.array([self.y[0] - self.y1[0]*self.lhs_extrap_dist])
            rhs_extrap_y = np.array([self.y[-1] + self.y1[-1]*self.rhs_extrap_dist])

            y_with_extrap = np.concatenate((lhs_extrap_y, self.y, rhs_extrap_y))
            y1_with_extrap = np.concatenate(([self.y1[0]], self.y1,
                                             [self.y1[-1]]))

            z = np.concatenate((y_with_extrap, y1_with_extrap))

            joined_struct_vec = np.vstack(self.struct_vecs[deriv])

            return np.atleast_1d(joined_struct_vec @ z.transpose()).ravel()
        else:
            return np.array([0.])

    def compute_zero_potential(self, y):
        """Calculates the value of the potential as if every entry in the
        structure vector was a zero.

        Args:
            y (np.arr):
                array of parameter vectors

        Returns:
            the value evaluated by the spline using num_zeros zeros"""

        if self.struct_vecs[0]:

            y1 = self.M @ y.transpose()

            y = y[:-2]

            zero_abcd = self.get_abcd([0])

            y = np.concatenate((np.zeros(1), y, np.zeros(1)))
            y1 = np.concatenate((np.zeros(1), y1, np.zeros(1)))
            z = np.concatenate((y, y1))

            return np.array(zero_abcd @ z)*len(np.vstack(self.struct_vecs[0]))
        else:
            return 0.

    @property
    def y(self):
        """Made as a property to ensure setting of self.y1 and
        self.end_derivs
        """

        return self._y

    @y.setter
    def y(self, y):
        self._y, self.end_derivs = np.split(y, [-2])
        self.y1 = self.M @ y.transpose()

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

        knots = self.x.copy()

        # compute maximum required extrapolation distance and add ghost knots
        distances_from_lhs_knot = x - knots[0]
        distances_from_rhs_knot = x - knots[-1]

        places_where_lhs_extrap = np.where(distances_from_lhs_knot < 0)
        places_where_rhs_extrap = np.where(distances_from_rhs_knot > 0)

        max_lhs_extrap_distance = 0
        max_rhs_extrap_distance = 0

        # if x outside of knots, update extrapolation variables
        if len(places_where_lhs_extrap[0]) > 0:
            max_lhs_extrap_distance = \
                np.max(np.abs(distances_from_lhs_knot[places_where_lhs_extrap]))

        if len(places_where_rhs_extrap[0]) > 0:
            max_rhs_extrap_distance = \
                np.max(np.abs(distances_from_rhs_knot[places_where_rhs_extrap]))

        return max_lhs_extrap_distance, max_rhs_extrap_distance

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

        # TODO: change worker.__init__() to take advantage of multi-add

        x = np.atleast_1d(x)

        knots = self.x.copy()

        extrap_distances = self.get_extrap_range(x)

        max_lhs_extrap_distance = extrap_distances[0]
        max_rhs_extrap_distance = extrap_distances[1]

        if max_lhs_extrap_distance > self.lhs_extrap_dist:
            self.lhs_extrap_dist = max_lhs_extrap_distance

        if max_rhs_extrap_distance > self.rhs_extrap_dist:
            self.rhs_extrap_dist = max_rhs_extrap_distance

        # add ghost knots
        knots = np.concatenate(\
            (np.array([knots[0] - self.lhs_extrap_dist]), knots))

        knots = np.concatenate(\
            (knots, np.array([knots[-1] + self.rhs_extrap_dist])))

        nknots = len(knots)

        # Perform interval search and prepare prefactors
        # intervals_from_zero = np.floor((x - self.x[0]) / self.h).astype(int)

        all_k = np.digitize(x, knots, right=True) - 1

        prefactors = knots[all_k + 1] - knots[all_k]

        all_t = (x - knots[all_k]) / prefactors

        # TODO: direct Horner's method for polyval

        h_00 = np.poly1d([2, -3, 0, 1])
        h_10 = np.poly1d([1, -2, 1, 0])
        h_01 = np.poly1d([-2, 3, 0, 0])
        h_11 = np.poly1d([1, -1, 0, 0])

        h_00 = np.polyder(h_00, deriv)
        h_10 = np.polyder(h_10, deriv)
        h_01 = np.polyder(h_01, deriv)
        h_11 = np.polyder(h_11, deriv)

        all_A = h_00(all_t)
        all_B = h_10(all_t) * prefactors
        all_C = h_01(all_t)
        all_D = h_11(all_t) * prefactors

        vec = np.zeros((len(x), 2*(nknots)))

        tmp_indices = np.arange(len(vec))

        vec[tmp_indices, all_k] += all_A
        vec[tmp_indices, all_k + nknots] += all_B
        vec[tmp_indices, all_k + 1] += all_C
        vec[tmp_indices, all_k + 1 + nknots] += all_D

        if deriv == 0: scaling = np.ones(len(prefactors))
        else: scaling =  1 / (prefactors * deriv)

        scaling = scaling.reshape((len(scaling), 1))

        vec *= scaling

        return vec

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
        #
        # min_to_add = np.min(val)
        # max_to_add = np.max(val)
        #
        # if min_to_add < self.ghost_lhs_extrap_knot:
        #     self.ghost_lhs_extrap_knot = min_to_add
        #
        # if max_to_add > self.ghost_rhs_extrap_knot:
        #     self.ghost_rhs_extrap_knot = max_to_add

        abcd_0 = self.get_abcd(val, 0)
        abcd_1 = self.get_abcd(val, 1)

        self.struct_vecs[0] += [abcd_0.squeeze()]
        self.struct_vecs[1] += [abcd_1.squeeze()]

        # Reshape indices replicate if adding an array of values
        self.indices += indices

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

    def plot(self):

        # raise NotImplementedError("Worker plotting is not ready yet")
        import matplotlib.pyplot as plt

        low, high = self.cutoff
        low     -= abs(2 * self.h)
        high    += abs(2 * self.h)

        if self.y is None:
            raise ValueError("Must specify y before plotting")

        plt.figure()
        plt.plot(self.x, self.y, 'ro', label='knots')

        tmp_struct = self.struct_vecs
        self.struct_vecs = [[],[]]

        self.add_to_struct_vec(plot_x, [0,0])

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
        natoms (int):
            the number of atoms in the system
    """

    def __init__(self, x, bc_type, natoms):
        super(RhoSpline, self).__init__(x, bc_type)

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

        if len(np.vstack(self.struct_vecs[deriv])) == 0: return ni

        # for i in self.struct_vec_dict.keys():
        results = super(RhoSpline, self).__call__(y, deriv)

        # TODO: store forwards/backwards indices separately to avoid np.arr()

        # for i in range(self.natoms):
        #     ni[i] = np.sum(results[np.array(self.indices)[:, 0] == i])

        np.add.at(ni, np.array(self.indices)[:,0], results)

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

        self.indices = [[], []]

    def __call__(self, y_fj, y_fk, y_g, deriv=[0,0,0]):

        if not self.fj_struct_vecs[deriv]:
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

        if deriv == 1:
            np.set_printoptions(precision=16)
            tmp = np.vstack((fj_results, fk_results, g_results))
            # logging.info("WORKER: fj, fk, g = {0}".format(tmp.T))
            # logging.info("WORKER: fj_results = {0}".format(fj_results))
            # logging.info("WORKER: fk_results = {0}".format(fk_results))
            # logging.info("WORKER: g_results = {0}".format(g_results))

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

        np.add.at(ni, np.array(self.indices[deriv])[:,0], results)

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

        fj_0, fk_0, g_0 = self.get_abcd(rij, rik, cos_theta, [0, 0, 0])
        fj_1, fk_1, g_1 = self.get_abcd(rij, rik, cos_theta, [1, 0, 0])
        fj_2, fk_2, g_2 = self.get_abcd(rij, rik, cos_theta, [0, 1, 0])
        fj_3, fk_3, g_3 = self.get_abcd(rij, rik, cos_theta, [0, 0, 1])

        self.fj_struct_vecs[0].append(fj_0)
        self.fk_struct_vecs[0].append(fk_0)
        self.g_struct_vecs[0].append(g_0)

        self.fj_struct_vecs[1]  += [fj_1, fj_3, fj_3, fj_2, fj_3, fj_3]
        self.fk_struct_vecs[1]  += [fk_1, fk_3, fk_3, fk_2, fk_3, fk_3]
        self.g_struct_vecs[1]   += [g_1, g_3, g_3, g_2, g_3, g_3]

        num_evals = len(indices)

        # for row in indices:
        #     i, j, k = indices
        #     deriv_indices = [[i, j], [i, j], [i, j], [i, k], [i, k], [i, k]]
        # deriv_indices = [[i, j]*num_evals, [i, j]*num_evals,
        #                  [i, j]*num_evals, [i, k]*num_evals,
        #                  [i, k]*num_evals, [i, k]*num_evals]

        tmp_indices = np.array(indices)
        ij = tmp_indices[:,[0,1]]
        ik = tmp_indices[:,[0,2]]

        # deriv_indices = [[list(indices[:,2])]*3, list(indices[:,[0,2]])]
        deriv_indices = list(ij) + list(ij) + list(ij) + list(ik) + list(ik) \
                        + list(ik)

        self.indices[1] += deriv_indices

        self.indices[0] += indices

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
