import numpy as np
import logging
import h5py
from scipy.interpolate import CubicSpline

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

        if not np.all(knots[1:] > knots[:-1], axis=0):
            raise ValueError("knots must be strictly increasing")

        self.knots = np.array(knots, dtype=float)
        self.n_knots = len(knots)
        self.bc_type = bc_type
        self.natoms = natoms
        self.index = 0

        """
        Extrapolation is done by building a spline between the end-point
        knot and a 'ghost' knot that is separated by a distance of
        extrap_dist.
        
        NOTE: the assumption that all extrapolation points are added at once 
        is NOT needed, since get_abcd() scales each point accordingly
        """

        self.extrap_dist = (knots[-1] - knots[0]) / 2.
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
        gamma: the result of M being row-scaled by beta
        structure vector: the summation of alpha + gamma
        
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

        self.structure_vectors = {}
        self.structure_vectors['energy'] = np.zeros(self.n_knots + 2)
        self.structure_vectors['forces'] = np.zeros((natoms, self.n_knots+2, 3))

    def get_abcd(self, x, deriv=0):
        """Calculates the spline coefficients for a set of points x

        Args:
            x (np.arr): list of points to be evaluated
            deriv (int): optionally compute the 1st derivative instead

        Returns:
            alpha: vector of coefficients to be added to alpha
            beta: vector of coefficients to be added to betas
            lhs_extrap: vector of coefficients to be added to lhs_extrap vector
            rhs_extrap: vector of coefficients to be added to rhs_extrap vector
        """
        x = np.atleast_1d(x)

        mn, mx = onepass_min_max(x)

        lhs_extrap_dist = max(self.extrap_dist, self.knots[0] - mn)
        rhs_extrap_dist = max(self.extrap_dist, mx - self.knots[-1])

        # add ghost knots
        knots = [self.knots[0] - lhs_extrap_dist] + self.knots.tolist() + \
                [self.knots[-1] + rhs_extrap_dist]

        knots = np.array(knots)

        # indicates the splines that the points fall into
        spline_bins = np.digitize(x, knots, right=True) - 1
        spline_bins = np.clip(spline_bins, 0, len(knots) - 2)

        if (np.min(spline_bins) < 0) or (np.max(spline_bins) >  self.n_knots+2):
            raise ValueError("Bad extrapolation; a point lies outside of the "
                             "computed extrapolation range")

        prefactor = knots[spline_bins + 1] - knots[spline_bins]

        t = (x - knots[spline_bins]) / prefactor
        t2 = t*t
        t3 = t2*t

        # shifts the extrapolation bin indices to the correct values
        # k = np.clip(spline_bins, 0, len(knots) - 2) - 1

        if deriv == 0:
            scaling = np.ones(len(x))

            A = 2*t3 - 3*t2 + 1
            B = t3 - 2*t2 + t
            C = -2*t3 + 3*t2
            D = t3 - t2

        elif deriv == 1:
            scaling =  1 / (prefactor * deriv)

            A = 6*t2 - 6*t
            B = 3*t2 - 4*t + 1
            C = -6*t2 + 6*t
            D = 3*t2 - 2*t

        else:
            raise ValueError("Only allowed derivative values are 0 and 1")

        B *= prefactor
        D *= prefactor

        A *= scaling
        B *= scaling
        C *= scaling
        D *= scaling

        alpha = np.zeros(self.n_knots)
        beta = np.zeros(self.n_knots)

        # values being extrapolated need to be indexed differently
        lhs_extrap_mask = spline_bins == 0
        rhs_extrap_mask = spline_bins == self.n_knots

        alpha[0] += np.sum(A[lhs_extrap_mask])
        alpha[0] += np.sum(C[lhs_extrap_mask])

        alpha[-1] += np.sum(A[rhs_extrap_mask])
        alpha[-1] += np.sum(C[rhs_extrap_mask])

        beta[0] += np.sum(A[lhs_extrap_mask])*(-lhs_extrap_dist)
        beta[0] += np.sum(B[lhs_extrap_mask])
        beta[0] += np.sum(D[lhs_extrap_mask])

        beta[-1] += np.sum(B[rhs_extrap_mask])
        beta[-1] += np.sum(C[rhs_extrap_mask])*rhs_extrap_dist
        beta[-1] += np.sum(D[rhs_extrap_mask])

        # now add internal knots
        internal_mask = np.logical_not(lhs_extrap_mask + rhs_extrap_mask)

        shifted_indices = spline_bins[internal_mask] - 1

        #
        np.add.at(alpha, shifted_indices, A[internal_mask])
        np.add.at(alpha, shifted_indices + 1, C[internal_mask])

        np.add.at(beta, shifted_indices, B[internal_mask])
        np.add.at(beta, shifted_indices + 1, D[internal_mask])

        big_alpha = np.concatenate([alpha, [0,0]])

        gamma = self.M * beta[:, np.newaxis]

        return big_alpha + np.sum(gamma, axis=0)

    @classmethod
    def from_hdf5(cls, hdf5_file, name, load_sv=True):
        """Builds a new spline from an HDF5 file.

        Args:
            hdf5_file (h5py.File): file to load from
            name (str): name of spline to load
            load_sv (bool): False if struct vec shouldn't be loaded (ffgSpline)

        Notes:
            this does NOT convert Dataset types into Numpy arrays, meaning the
            file must remain open in order to use the worker
        """

        spline_data = hdf5_file[name]

        x = np.array(spline_data['x'])
        bc_type = tuple(spline_data.attrs['bc_type'])
        bc_type = ['fixed' if el==1 else 'natural' for el in bc_type]
        M = np.array(spline_data['M'])
        natoms = int(spline_data.attrs['natoms'])

        ws = cls(x, bc_type, natoms, M)

        ws.extrap_dist = float(spline_data.attrs['extrap_dist'])
        ws.lhs_extrap_dist = float(spline_data.attrs['lhs_extrap_dist'])
        ws.rhs_extrap_dist = float(spline_data.attrs['rhs_extrap_dist'])
        ws.index = int(spline_data.attrs['index'])

        if load_sv:
            ws.structure_vectors['energy'] = np.array(spline_data['energy_struct_vec'])
            ws.structure_vectors['forces'] = np.array(spline_data['forces_struct_vec'])

        ws.n_knots = len(x)

        return ws

    def add_to_hdf5(self, hdf5_file, name, save_sv=True):
        """Adds spline to HDF5 file

        Args:
            hdf5_file (h5py.File): file object to be added to
            name (str): desired group name
            save_sv (bool): False if no need to save struct vecs (for ffgSpline)
        """

        new_group = hdf5_file.create_group(name)

        new_group.attrs["extrap_dist"] = self.extrap_dist
        new_group.attrs["lhs_extrap_dist"] = self.lhs_extrap_dist
        new_group.attrs["rhs_extrap_dist"] = self.rhs_extrap_dist
        new_group.attrs["natoms"] = self.natoms
        new_group.attrs["index"] = self.index
        new_group.attrs["bc_type"] = [1 if el=='fixed' else 0
                for el in self.bc_type]

        new_group.create_dataset("M", data = self.M)
        new_group.create_dataset("x", data = self.knots)

        if save_sv:
            new_group.create_dataset("energy_struct_vec",
                data=self.structure_vectors['energy'])

            new_group.create_dataset("forces_struct_vec",
                data=self.structure_vectors['forces'])

    def add_to_energy_struct_vec(self, values):
        self.structure_vectors['energy'] += self.get_abcd(values)

    def add_to_forces_struct_vec(self, values, dirs, atom_id):
        dirs = np.array(dirs)

        abcd = self.get_abcd(values, 1).ravel()

        self.structure_vectors['forces'][atom_id, :, :] += np.einsum('i,j->ij', abcd, dirs)

    def calc_energy(self, y):
        """Evaluates the energy structure vector for a given y. A second list of
        parameters is created by appending the 'ghost knot' positions to y

        Args:
            y (np.arr): a list of N knot y-coords, plus 2 boundary conditions

        Returns:
            energy (float): the energy of the system
        """

        return self.structure_vectors['energy'] @ y.T

    def calc_forces(self, y):
        return np.einsum('ijk,pj->pik', self.structure_vectors['forces'], y)

class RhoSpline(WorkerSpline):
    """RhoSpline objects are a variant of the WorkerSpline, but that require
    tracking energies for each atom. To account for potential many-atom
    simulation cells, sparse matrices are used for the forces struct vec"""

    def __init__(self, knots, bc_type, natoms, M=None):
        super(RhoSpline, self).__init__(knots, bc_type, natoms, M)

        self.structure_vectors['energy'] = np.zeros((self.natoms, self.n_knots+2))

        N = self.natoms
        self.structure_vectors['forces'] = lil_matrix((3*N*N, self.n_knots+2),dtype=float)

    def add_to_hdf5(self, hdf5_file, name, save_sv=False):
        """The 'save_sv' argument allows saving as a sparse matrix"""
        super().add_to_hdf5(hdf5_file, name, save_sv=False)

        spline_group = hdf5_file[name]

        spline_group.create_dataset("energy_struct_vec",
                data=self.structure_vectors['energy'])

        f_sv = self.structure_vectors['forces']
        spline_group.create_dataset('f_sv.data', data=f_sv.data)
        spline_group.create_dataset('f_sv.indices', data=f_sv.indices)
        spline_group.create_dataset('f_sv.indptr', data=f_sv.indptr)
        spline_group.create_dataset('f_sv.shape', data=f_sv.shape)

    @classmethod
    def from_hdf5(cls, hdf5_file, name, load_sv=True):
        """The 'load_sv' argument is only overloaded for ffgSpline objects"""

        spline_data = hdf5_file[name]

        x = np.array(spline_data['x'])
        bc_type = spline_data.attrs['bc_type']
        bc_type = ['fixed' if el==1 else 'natural' for el in bc_type]
        M = np.array(spline_data['M'])
        natoms = np.array(spline_data.attrs['natoms'])

        rho = cls(x, bc_type, natoms, M)

        rho.extrap_dist = np.array(spline_data.attrs['extrap_dist'])
        rho.lhs_extrap_dist = np.array(spline_data.attrs['lhs_extrap_dist'])
        rho.rhs_extrap_dist = np.array(spline_data.attrs['rhs_extrap_dist'])
        rho.index = int(spline_data.attrs['index'])

        rho.structure_vectors['energy'] = np.array(spline_data['energy_struct_vec'])

        f_sv_data = np.array(spline_data['f_sv.data'])
        f_sv_indices = np.array(spline_data['f_sv.indices'])
        f_sv_indptr = np.array(spline_data['f_sv.indptr'])
        f_sv_shape = np.array(spline_data['f_sv.shape'])

        rho.structure_vectors['forces'] = csr_matrix(
                (f_sv_data, f_sv_indices, f_sv_indptr), shape=f_sv_shape)

        rho.n_knots = len(x)

        return rho

    def add_to_energy_struct_vec(self, values, atom_id):
        self.structure_vectors['energy'][atom_id, :] += self.get_abcd(values)

    def add_to_forces_struct_vec(self, value, dir, i, j):
        """Single add used because need to speciy two atom tags"""
        abcd_3d = np.einsum('i,j->ij', self.get_abcd(value, 1).ravel(), dir)

        N = self.natoms

        for a in range(3):
            self.structure_vectors['forces'][N*N*a + N*i+ j,:] += abcd_3d[:,a]

    def calc_energy(self, y):
        return self.structure_vectors['energy'] @ y.T

    def calc_forces(self, y):
        return (self.structure_vectors['forces'] @ y.T).T

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

        nknots_cartesian = (len(fj.knots)+2)*(len(fk.knots)+2)*(len(g.knots)+2)

        N = self.natoms

        # initialized as array for fast building; converted to sparse later
        self.structure_vectors = {}
        self.structure_vectors['energy'] = np.zeros((natoms, nknots_cartesian))
        self.structure_vectors['forces'] = np.zeros((3*N*N, nknots_cartesian))

    @classmethod
    def from_hdf5(cls, hdf5_file, name):
        ffg_data = hdf5_file[name]

        natoms = int(ffg_data.attrs['natoms'])

        fj = WorkerSpline.from_hdf5(ffg_data, 'fj', load_sv=False)
        fk = WorkerSpline.from_hdf5(ffg_data, 'fk', load_sv=False)
        g = WorkerSpline.from_hdf5(ffg_data, 'g', load_sv=False)

        ffg = cls(fj, fk, g, natoms)

        e_sv_data = ffg_data['e_sv.data']
        e_sv_indices = ffg_data['e_sv.indices']
        e_sv_indptr = ffg_data['e_sv.indptr']
        e_sv_shape = ffg_data['e_sv.shape']

        ffg.structure_vectors['energy'] = csr_matrix(
                (e_sv_data, e_sv_indices, e_sv_indptr), shape=e_sv_shape)

        f_sv_data = ffg_data['f_sv.data']
        f_sv_indices = ffg_data['f_sv.indices']
        f_sv_indptr = ffg_data['f_sv.indptr']
        f_sv_shape = ffg_data['f_sv.shape']

        ffg.structure_vectors['forces'] = csr_matrix(
                (f_sv_data, f_sv_indices, f_sv_indptr), shape=f_sv_shape)

        return ffg

    def add_to_hdf5(self, hdf5_file, name):
        new_group = hdf5_file.create_group(name)

        new_group.attrs['natoms'] = self.natoms

        # assumes scipy sparse CSR
        new_group.create_dataset('e_sv.data', data=self.structure_vectors['energy'].data)
        new_group.create_dataset('e_sv.indices', data=self.structure_vectors['energy'].indices)
        new_group.create_dataset('e_sv.indptr', data=self.structure_vectors['energy'].indptr)
        new_group.create_dataset('e_sv.shape', data=self.structure_vectors['energy'].shape)

        new_group.create_dataset('f_sv.data', data=self.structure_vectors['forces'].data)
        new_group.create_dataset('f_sv.indices', data=self.structure_vectors['forces'].indices)
        new_group.create_dataset('f_sv.indptr', data=self.structure_vectors['forces'].indptr)
        new_group.create_dataset('f_sv.shape', data=self.structure_vectors['forces'].shape)

        self.fj.add_to_hdf5(new_group, 'fj', save_sv=False)
        self.fk.add_to_hdf5(new_group, 'fk', save_sv=False)
        self.g.add_to_hdf5(new_group, 'g', save_sv=False)

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

    def calc_energy(self, y_fj, y_fk, y_g):

        cart1 = np.einsum("ij,ik->ijk", y_fj, y_fk)
        cart1 = cart1.reshape((cart1.shape[0], cart1.shape[1]*cart1.shape[2]))

        cart2 = np.einsum("ij,ik->ijk", cart1, y_g)

        cart_y = cart2.reshape((cart2.shape[0], cart2.shape[1]*cart2.shape[2]))

        return (self.structure_vectors['energy'] @ cart_y.T).T

    def calc_forces(self, y_fj, y_fk, y_g):

        cart1 = np.einsum("ij,ik->ijk", y_fj, y_fk)
        cart1 = cart1.reshape((cart1.shape[0], cart1.shape[1]*cart1.shape[2]))

        cart2 = np.einsum("ij,ik->ijk", cart1, y_g)

        cart_y = cart2.reshape((cart2.shape[0], cart2.shape[1]*cart2.shape[2]))

        return (self.structure_vectors['forces'] @ cart_y.T).T

    def calc_energy_and_forces(self, y_fj, y_fk, y_g):
        # TODO: saves time computing together
        pass

    def add_to_energy_struct_vec(self, rij, rik, cos, atom_id):
        """Updates structure vectors with given values"""

        abcd_fj = self.fj.get_abcd(rij, 0).ravel()
        abcd_fk = self.fk.get_abcd(rik, 0).ravel()
        abcd_g = self.g.get_abcd(cos, 0).ravel()

        cart = outer_prod_simple(
            outer_prod_simple(abcd_fj, abcd_fk), abcd_g).ravel()

        self.structure_vectors['energy'][atom_id, :] += cart

    def add_to_forces_struct_vec(self, rij, rik, cos, dirs, i, j, k):
        """Adds all 6 directional information to the struct vector for the
        given triplet values

        Args:
            rij : single rij value
            rik : single rik value
            cos : single cos_theta value
            dirs : ordered list of the six terms of the triplet deriv directions
            i : atom i tag
            j : atom j tag
            k : atom k tag
        """

        fj_1, fk_1, g_1 = self.get_abcd(rij, rik, cos, [1, 0, 0])
        fj_2, fk_2, g_2 = self.get_abcd(rij, rik, cos, [0, 1, 0])
        fj_3, fk_3, g_3 = self.get_abcd(rij, rik, cos, [0, 0, 1])

        v1 = outer_prod_simple(outer_prod_simple(fj_1, fk_1), g_1) # fj' fk g
        v2 = outer_prod_simple(outer_prod_simple(fj_2, fk_2), g_2) # fj fk' g
        v3 = outer_prod_simple(outer_prod_simple(fj_3, fk_3), g_3) # fj fk g' -> PF

        # all 6 terms to be added
        t0 = np.einsum('i,k->ik', v1, dirs[0])
        t1 = np.einsum('i,k->ik', v3, dirs[1])
        t2 = np.einsum('i,k->ik', v3, dirs[2])

        t3 = np.einsum('i,k->ik', v2, dirs[3])
        t4 = np.einsum('i,k->ik', v3, dirs[4])
        t5 = np.einsum('i,k->ik', v3, dirs[5])

        # condensed versions
        fj = t0 + t1 + t2
        fk = t3 + t4 + t5

        N = self.natoms
        for a in range(3):
            self.structure_vectors['forces'][N*N*a + N*i + i, :] += fj[:, a]
            self.structure_vectors['forces'][N*N*a + N*j + i, :] -= fj[:, a]

            self.structure_vectors['forces'][N*N*a + N*i + i, :] += fk[:, a]
            self.structure_vectors['forces'][N*N*a + N*k + i, :] -= fk[:, a]

class USpline(CubicSpline):
    """The U functions are unable to take advantage of pre-computing spline
    coefficients, so it may be faster to use a scipy CubicSpline object with
    added functionality for linear extrapolation"""

    def __init__(self, x, y, end_derivs=(0,0)):

        self.d0, self.dN = end_derivs

        super(USpline,self).__init__(x, y, bc_type=((1, self.d0),(1, self.dN)))
        self.cutoff = (x[0],x[len(x)-1])

        self.h = x[1]-x[0]

    def in_range(self, x):
        """Checks if a given value is within the spline's cutoff range"""

        return (x >= self.cutoff[0]) and (x <= self.cutoff[1])

    def extrap(self, x):
        """Performs linear extrapolation past the endpoints of the spline"""

        if x < self.cutoff[0]:
            return self(self.x[0]) - self.d0*(self.x[0]-x)
        elif x > self.cutoff[1]:
            return self(self.x[-1]) + self.dN*(x-self.x[-1])

    def __call__(self, x, i):
        if self.in_range(x):
            return super(USpline, self).__call__(x)
        else:
            return self.extrap(x)

    def __call__(self, x, i=None):
        """Evaluates the spline at the given point, linearly extrapolating if
        outside of the spline cutoff. If 'i' is specified, evaluates the ith
        derivative instead.
        """

        if i:
            if x < self.cutoff[0]:
                return self.d0
            elif x > self.cutoff[1]:
                return self.dN
            else:
                return super(USpline, self).__call__(x, i)

        if self.in_range(x):
            return super(USpline, self).__call__(x)
        else:
            return self.extrap(x)

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

    A = np.zeros((n + 2, n + 2))
    B = np.zeros((n + 2, n + 4))

    # match 2nd deriv for internal knots
    fillA = diags(np.array([2, 8, 2]), [0, 1, 2], (n, n + 2))
    fillB = diags([-6, 0, 6], [0, 1, 2], (n, n + 2))
    A[1:n+1, :n+2] = fillA.toarray()
    B[1:n+1, :n+2] = fillB.toarray()

    # equation accounting for lhs bc
    if bc_lhs == 'natural':
        A[0,0] = -4; A[0,1] = -2
        B[0,0] = 6; B[0,1] = -6; B[0,-2] = 1
    elif bc_lhs == 'fixed':
        A[0,0] = 1/dx;
        B[0,-2] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    # equation accounting for rhs bc
    if bc_rhs == 'natural':
        A[-1,-2] = 2; A[-1,-1] = 4
        B[-1,-4] = -6; B[-1,-3] = 6; B[-1,-1] = 1
    elif bc_rhs == 'fixed':
        A[-1,-1] = 1/dx
        B[-1,-1] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    A *= dx

    # M = A^(-1)B
    return np.dot(np.linalg.inv(A), B)