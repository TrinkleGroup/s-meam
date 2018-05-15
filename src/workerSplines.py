import numpy as np
import logging
import h5py

from scipy.sparse import diags, lil_matrix, csr_matrix

# from src.numba_functions import jit_add_at_1D, jit_add_at_2D
from src.numba_functions import onepass_min_max, outer_prod, outer_prod_simple
# from src.fast import onepass_min_max, mat_vec_mult

logger = logging.getLogger(__name__)

class WorkerSpline:
    """A representation of a cubic spline specifically tailored to meet
    the needs of a Worker object.

    Attributes:
        x (np.arr): x-coordinates of knot points

        h (float): knot spacing (assumes equally-spaced)

        y (np.arr): y-coordinates of knot points only set by solve_for_derivs()

        y1 (np.arr): first derivatives at knot points only set by
            solve_for_derivs()

        M (np.arr): M = AB where A and B are the matrices from the system of
            equations that comes from matching second derivatives at knot
            points (see build_M() for details)

        cutoff (tuple-like): upper and lower bounds of knots (x[0], x[-1])

        end_derivs (tuple-like): first derivatives at endpoints

        bc_type (tuple): 2-element tuple corresponding to boundary conditions at LHS/RHS
            respectively. Options are 'natural' (zero second derivative) and
            'fixed' (fixed first derivative). If 'fixed' on one/both ends,
            self.end_derivatives cannot be None for that end

        energy_struct_vec (np.arr): NxZ matrix where N is the number of atoms in
            the system, Z is the number of knot points being used. The i-th
            row represents the sum of the contributions to the i-th atom. The
            first half of each row is the coefficients in front of the knot
            value terms in the Cubic Hermitian Spline format; the second half of
            each row is the coefficients in front of the knot derivative values.

        forces_struct_vec (np.arr): same as energy_struct_vec format,
            but with a third dimension for xyz cartesian directions
    """

    def __init__(self, x=None, bc_type=None, natoms=0, M=None):

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
        # self.h = x[1] - x[0]

        self.n_knots = len(x)

        self.bc_type = bc_type

        if M is None:
            self.M = build_M(len(x), x[1] - x[0], bc_type)
        else:
            self.M = M

        self.natoms = natoms

        cutoff = (x[0], x[-1])
        self.extrap_dist = (cutoff[1] - cutoff[0]) / 2.

        self.lhs_extrap_dist = self.extrap_dist
        self.rhs_extrap_dist = self.extrap_dist

        # Variables that will be set at some point
        self.energy_struct_vec = np.zeros(2*len(x)+4)
        self.forces_struct_vec = np.zeros((natoms, 2*len(x)+4, 3))
        self.index = 0

        # Variables that will be set on evaluation
        self._y = None
        self.y1 = None
        self.end_derivs = None

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
        # print(spline_data.attrs['index'])
        ws.index = int(spline_data.attrs['index'])

        if load_sv:
            ws.energy_struct_vec = np.array(spline_data['energy_struct_vec'])
            ws.forces_struct_vec = np.array(spline_data['forces_struct_vec'])

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
        # print(self.index)
        new_group.attrs["index"] = self.index
        new_group.attrs["bc_type"] = [1 if el=='fixed' else 0
                for el in self.bc_type]

        new_group.create_dataset("M", data = self.M)
        new_group.create_dataset("x", data = self.x)

        if save_sv:
            e_s = self.energy_struct_vec.shape
            f_s = self.forces_struct_vec.shape

            new_group.create_dataset("energy_struct_vec",
                    data=self.energy_struct_vec)
            new_group.create_dataset("forces_struct_vec",
                    data=self.forces_struct_vec)
        # new_group.create_dataset("forces_struct_vec", data =
        #         self.forces_struct_vec.reshape(f_s[0]*f_s[2], f_s[1]))

    @property
    def y(self):
        """Made as a property to ensure setting of self.y1 and
        self.end_derivs
        """
        return self._y

    @y.setter
    def y(self, y):
        # TODO: get rid of this setter; or just make a y1 setter?
        # NOTE: end derivs come into play only during y1 evalution
        self._y = y[:, :-2]; self.end_derivs = y[:, -2:]
        self.y1 = (self.M @ y.T).T

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

        mn, mx = onepass_min_max(x)
        self.lhs_extrap_dist = max(self.extrap_dist, abs(mn - self.x[0]))
        self.rhs_extrap_dist = max(self.extrap_dist, abs(mx - self.x[-1]))

        # add ghost knots
        knots = [self.x[0] - self.lhs_extrap_dist] + self.x.tolist() + \
                [self.x[-1] + self.rhs_extrap_dist]

        knots = np.array(knots)

        nknots = knots.shape[0]

        # Perform interval search and prepare prefactors
        all_k = np.digitize(x, knots, right=True) - 1
        all_k = np.clip(all_k, 0, len(knots) - 1)

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

        # z = [self.y[:, 0] - self.y1[:, 0]*self.lhs_extrap_dist] +\
        #     self.y.tolist() +\
        #     [self.y[:, -1] + self.y1[:, -1]*self.rhs_extrap_dist,
        #      self.y1[:, 0]] + self.y1.tolist() + [self.y1[:, -1]]

        z = np.zeros((self.y.shape[0], 2*self.y.shape[1]+4))

        z[:, 0] = self.y[:, 0] - self.y1[:, 0]*self.lhs_extrap_dist
        z[:, 1:1+self.n_knots] = self.y
        z[:, 1+self.n_knots] = self.y[:,-1] + self.y1[:,-1]*self.rhs_extrap_dist
        z[:, 2+self.n_knots] = self.y1[:, 0]
        z[:, 3+self.n_knots:3+2*self.n_knots] = self.y1
        z[:, 3+2*self.n_knots] = self.y1[:, -1]

        return (self.energy_struct_vec @ z.T)

    def calc_forces(self, y):
        self.y = y

        z = np.zeros((self.y.shape[0], 2*self.y.shape[1]+4))
        z[:, 0] = self.y[:, 0] - self.y1[:, 0]*self.lhs_extrap_dist
        z[:, 1:1+self.n_knots] = self.y
        z[:, 1+self.n_knots] = self.y[:,-1] + self.y1[:,-1]*self.rhs_extrap_dist
        z[:, 2+self.n_knots] = self.y1[:, 0]
        z[:, 3+self.n_knots:3+2*self.n_knots] = self.y1
        z[:, 3+2*self.n_knots] = self.y1[:, -1]

        return np.einsum('ijk,pj->pik', self.forces_struct_vec, z)

    def add_to_forces_struct_vec(self, values, dirs, atom_id):
        dirs = np.array(dirs)

        abcd = self.get_abcd(values, 1).ravel()

        self.forces_struct_vec[atom_id, :, :] +=np.einsum('i,j->ij', abcd, dirs)

    def add_to_energy_struct_vec(self, values):
        self.energy_struct_vec += np.sum(self.get_abcd(values, 0), axis=0)


class USpline(WorkerSpline):

    def __init__(self, x, bc_type, natoms, M=None):
        super(USpline, self).__init__(x, bc_type, natoms, M)

        self.deriv_struct_vec = np.zeros((natoms, 2*len(self.x)+4))

        self.zero_abcd = self.get_abcd([0])

        self.atoms_embedded = 0

    def add_to_hdf5(self, hdf5_file, name):
        super().add_to_hdf5(hdf5_file, name)

        uspline_group = hdf5_file[name]

        uspline_group.create_dataset('deriv_struct_vec',
                data=self.deriv_struct_vec)

        uspline_group.create_dataset('zero_abcd', data=self.zero_abcd)
        uspline_group.attrs['atoms_embedded'] = self.atoms_embedded

    @classmethod
    def from_hdf5(cls, hdf5_file, name):

        spline_data = hdf5_file[name]

        x = np.array(spline_data['x'])
        bc_type = spline_data.attrs['bc_type']
        bc_type = ['fixed' if el==1 else 'natural' for el in bc_type]
        M = np.array(spline_data['M'])
        natoms = int(spline_data.attrs['natoms'])

        us = cls(x, bc_type, natoms, M)

        us.extrap_dist = np.array(spline_data.attrs['extrap_dist'])
        us.lhs_extrap_dist = np.array(spline_data.attrs['lhs_extrap_dist'])
        us.rhs_extrap_dist = np.array(spline_data.attrs['rhs_extrap_dist'])
        us.index = int(spline_data.attrs['index'])

        us.energy_struct_vec = np.array(spline_data['energy_struct_vec'])
        us.deriv_struct_vec = np.array(spline_data['deriv_struct_vec'])
        us.forces_struct_vec = np.array(spline_data['forces_struct_vec'])

        us.zero_abcd = np.array(spline_data['zero_abcd'])
        us.atoms_embedded = np.array(spline_data.attrs['atoms_embedded'])

        us.n_knots = len(x)

        return us

    def reset(self):
        self.atoms_embedded = 0
        self.deriv_struct_vec[:] = 0
        self.energy_struct_vec[:] = 0

    def calc_energy(self, y):
        self.y = y

        z = np.zeros((self.y.shape[0], 2*self.y.shape[1]+4))

        z[:, 0] = self.y[:, 0] - self.y1[:, 0]*self.lhs_extrap_dist
        z[:, 1:1+self.n_knots] = self.y
        z[:, 1+self.n_knots] = self.y[:,-1] + self.y1[:,-1]*self.rhs_extrap_dist
        z[:, 2+self.n_knots] = self.y1[:, 0]
        z[:, 3+self.n_knots:3+2*self.n_knots] = self.y1
        z[:, 3+2*self.n_knots] = self.y1[:, -1]

        return np.einsum("ij,ij->i", self.energy_struct_vec, z)

    def add_to_energy_struct_vec(self, values):
        num_new_atoms = values.shape[1]

        if num_new_atoms > 0:
            self.atoms_embedded += num_new_atoms

            values = np.atleast_1d(values)
            org_shape = values.shape
            flat_values = values.ravel()

            abcd = self.get_abcd(flat_values, 0)
            abcd = abcd.reshape(list(org_shape) + [abcd.shape[1]])

            self.energy_struct_vec += np.sum(abcd, axis=1)

    def add_to_deriv_struct_vec(self, values, indices):
        if values.shape[0] > 0:

            values = np.atleast_1d(values)
            org_shape = values.shape
            flat_values = values.ravel()

            abcd = self.get_abcd(flat_values, 1)
            abcd = abcd.reshape(list(org_shape) + [abcd.shape[1]])

            self.deriv_struct_vec[:, indices, :] = abcd

    def calc_deriv(self, y):
        self.y = y

        z = np.zeros((self.y.shape[0], 2*self.y.shape[1]+4))
        z[:, 0] = self.y[:, 0] - self.y1[:, 0]*self.lhs_extrap_dist
        z[:, 1:1+self.n_knots] = self.y
        z[:, 1+self.n_knots] = self.y[:,-1] + self.y1[:,-1]*self.rhs_extrap_dist
        z[:, 2+self.n_knots] = self.y1[:, 0]
        z[:, 3+self.n_knots:3+2*self.n_knots] = self.y1
        z[:, 3+2*self.n_knots] = self.y1[:, -1]

        # return self.deriv_struct_vec @ z
        return np.einsum('ijk,ik->ij', self.deriv_struct_vec, z)

    def compute_zero_potential(self, y):
        """Calculates the value of the potential as if every entry in the
        structure vector was a zero.

        Args:
            y (np.arr):
                array of parameter vectors

        Returns:
            the value evaluated by the spline using num_zeros zeros"""

        y1 = (self.M @ y.T).T

        y = y[:, :-2]

        z = np.zeros((y.shape[0], 2*y.shape[1]+4))

        z[:, 0] = y[:, 0] - y1[:, 0]*self.lhs_extrap_dist
        z[:, 1:1+self.n_knots] = y
        z[:, 1+self.n_knots] = 0
        z[:, 2+self.n_knots] = 0
        z[:, 3+self.n_knots:3+2*self.n_knots] = y1
        z[:, 3+2*self.n_knots] = y1[:, -1]

        return (self.zero_abcd @ z.T).T*self.atoms_embedded


class RhoSpline(WorkerSpline):

    def __init__(self, x, bc_type, natoms, M=None):
        super(RhoSpline, self).__init__(x, bc_type, natoms, M)

        self.energy_struct_vec = np.zeros((self.natoms, 2*len(x)+4))

        N = self.natoms
        self.forces_struct_vec = lil_matrix((3*N*N, 2*len(x)+4), dtype=float)

    def add_to_hdf5(self, hdf5_file, name):
        super().add_to_hdf5(hdf5_file, name, save_sv=False)

        spline_group = hdf5_file[name]

        spline_group.create_dataset("energy_struct_vec",
                data=self.energy_struct_vec)

        f_sv = self.forces_struct_vec
        spline_group.create_dataset('f_sv.data', data=f_sv.data)
        spline_group.create_dataset('f_sv.indices', data=f_sv.indices)
        spline_group.create_dataset('f_sv.indptr', data=f_sv.indptr)
        spline_group.create_dataset('f_sv.shape', data=f_sv.shape)

    @classmethod
    def from_hdf5(cls, hdf5_file, name):

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

        rho.energy_struct_vec = np.array(spline_data['energy_struct_vec'])

        f_sv_data = np.array(spline_data['f_sv.data'])
        f_sv_indices = np.array(spline_data['f_sv.indices'])
        f_sv_indptr = np.array(spline_data['f_sv.indptr'])
        f_sv_shape = np.array(spline_data['f_sv.shape'])

        rho.forces_struct_vec = csr_matrix(
                (f_sv_data, f_sv_indices, f_sv_indptr), shape=f_sv_shape)

        rho.n_knots = len(x)

        return rho

    def calc_forces(self, y):
        self.y = y

        z = np.zeros((self.y.shape[0], 2*self.y.shape[1]+4))
        z[:, 0] = self.y[:, 0] - self.y1[:, 0]*self.lhs_extrap_dist
        z[:, 1:1+self.n_knots] = self.y
        z[:, 1+self.n_knots] = self.y[:,-1] + self.y1[:,-1]*self.rhs_extrap_dist
        z[:, 2+self.n_knots] = self.y1[:, 0]
        z[:, 3+self.n_knots:3+2*self.n_knots] = self.y1
        z[:, 3+2*self.n_knots] = self.y1[:, -1]

        return (self.forces_struct_vec @ z.T).T

    def add_to_energy_struct_vec(self, values, atom_id):
        self.energy_struct_vec[atom_id, :] += np.sum(self.get_abcd(values, 0),
                                                     axis=0)

    def add_to_forces_struct_vec(self, value, dir, i, j):
        """Single add for neighbor"""
        abcd_3d = np.einsum('i,j->ij', self.get_abcd(value, 1).ravel(), dir)

        N = self.natoms

        for a in range(3):
            self.forces_struct_vec[N*N*a + N*i+ j,:] += abcd_3d[:,a]

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

        N = self.natoms

        # initialized as array for fast building; converted to sparse later
        self.energy_struct_vec = np.zeros((natoms, nknots_cartesian))
        self.forces_struct_vec = np.zeros((3*N*N, nknots_cartesian))

    def add_to_hdf5(self, hdf5_file, name):
        new_group = hdf5_file.create_group(name)

        new_group.attrs['natoms'] = self.natoms

        # assumes scipy sparse CSR
        new_group.create_dataset('e_sv.data', data=self.energy_struct_vec.data)
        new_group.create_dataset('e_sv.indices',
                data=self.energy_struct_vec.indices)
        new_group.create_dataset('e_sv.indptr',
                data=self.energy_struct_vec.indptr)
        new_group.create_dataset('e_sv.shape',
                data=self.energy_struct_vec.shape)

        new_group.create_dataset('f_sv.data', data=self.forces_struct_vec.data)
        new_group.create_dataset('f_sv.indices',
                data=self.forces_struct_vec.indices)
        new_group.create_dataset('f_sv.indptr',
                data=self.forces_struct_vec.indptr)
        new_group.create_dataset('f_sv.shape',
                data=self.forces_struct_vec.shape)

        self.fj.add_to_hdf5(new_group, 'fj', save_sv=False)
        self.fk.add_to_hdf5(new_group, 'fk', save_sv=False)
        self.g.add_to_hdf5(new_group, 'g', save_sv=False)

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

        ffg.energy_struct_vec = csr_matrix(
                (e_sv_data, e_sv_indices, e_sv_indptr), shape=e_sv_shape)

        f_sv_data = ffg_data['f_sv.data']
        f_sv_indices = ffg_data['f_sv.indices']
        f_sv_indptr = ffg_data['f_sv.indptr']
        f_sv_shape = ffg_data['f_sv.shape']

        ffg.forces_struct_vec= csr_matrix(
                (f_sv_data, f_sv_indices, f_sv_indptr), shape=f_sv_shape)

        return ffg

    # @profile
    def calc_energy(self, y_fj, y_fk, y_g):
        fj = self.fj
        fk = self.fk
        g = self.g

        fj.y = y_fj
        fk.y = y_fk
        g.y = y_g

        z_fj = np.zeros((fj.y.shape[0], 2*fj.y.shape[1]+4))
        z_fj[:, 0] = fj.y[:, 0] - fj.y1[:, 0]*fj.lhs_extrap_dist
        z_fj[:, 1:1+fj.n_knots] = fj.y
        z_fj[:, 1+fj.n_knots] = fj.y[:,-1] + fj.y1[:,-1]*fj.rhs_extrap_dist
        z_fj[:, 2+fj.n_knots] = fj.y1[:, 0]
        z_fj[:, 3+fj.n_knots:3+2*fj.n_knots] = fj.y1
        z_fj[:, 3+2*fj.n_knots] = fj.y1[:, -1]

        z_fk = np.zeros((fk.y.shape[0], 2*fk.y.shape[1]+4))
        z_fk[:, 0] = fk.y[:, 0] - fk.y1[:, 0]*fk.lhs_extrap_dist
        z_fk[:, 1:1+fk.n_knots] = fk.y
        z_fk[:, 1+fk.n_knots] = fk.y[:,-1] + fk.y1[:,-1]*fk.rhs_extrap_dist
        z_fk[:, 2+fk.n_knots] = fk.y1[:, 0]
        z_fk[:, 3+fk.n_knots:3+2*fk.n_knots] = fk.y1
        z_fk[:, 3+2*fk.n_knots] = fk.y1[:, -1]

        z_g = np.zeros((g.y.shape[0], 2*g.y.shape[1]+4))
        z_g[:, 0] = g.y[:, 0] - g.y1[:, 0]*g.lhs_extrap_dist
        z_g[:, 1:1+g.n_knots] = g.y
        z_g[:, 1+g.n_knots] = g.y[:,-1] + g.y1[:,-1]*g.rhs_extrap_dist
        z_g[:, 2+g.n_knots] = g.y1[:, 0]
        z_g[:, 3+g.n_knots:3+2*g.n_knots] = g.y1
        z_g[:, 3+2*g.n_knots] = g.y1[:, -1]

        z_fj = np.atleast_2d(z_fj)
        z_fk = np.atleast_2d(z_fk)
        z_g = np.atleast_2d(z_g)

        # z_cart = outer_prod(outer_prod(z_fj, z_fk), z_g)

        cart1 = np.einsum("ij,ik->ijk", z_fj, z_fk)
        cart1 = cart1.reshape((cart1.shape[0], cart1.shape[1]*cart1.shape[2]))
        cart2 = np.einsum("ij,ik->ijk", cart1, z_g)
        z_cart = cart2.reshape((cart2.shape[0], cart2.shape[1]*cart2.shape[2]))

        return (self.energy_struct_vec @ z_cart.T).T

    # @profile
    def calc_forces(self, y_fj, y_fk, y_g):

        fj = self.fj
        fk = self.fk
        g = self.g

        fj.y = y_fj
        fk.y = y_fk
        g.y = y_g

        z_fj = np.zeros((fj.y.shape[0], 2*fj.y.shape[1]+4))
        z_fj[:, 0] = fj.y[:, 0] - fj.y1[:, 0]*fj.lhs_extrap_dist
        z_fj[:, 1:1+fj.n_knots] = fj.y
        z_fj[:, 1+fj.n_knots] = fj.y[:,-1] + fj.y1[:,-1]*fj.rhs_extrap_dist
        z_fj[:, 2+fj.n_knots] = fj.y1[:, 0]
        z_fj[:, 3+fj.n_knots:3+2*fj.n_knots] = fj.y1
        z_fj[:, 3+2*fj.n_knots] = fj.y1[:, -1]

        z_fk = np.zeros((fk.y.shape[0], 2*fk.y.shape[1]+4))
        z_fk[:, 0] = fk.y[:, 0] - fk.y1[:, 0]*fk.lhs_extrap_dist
        z_fk[:, 1:1+fk.n_knots] = fk.y
        z_fk[:, 1+fk.n_knots] = fk.y[:,-1] + fk.y1[:,-1]*fk.rhs_extrap_dist
        z_fk[:, 2+fk.n_knots] = fk.y1[:, 0]
        z_fk[:, 3+fk.n_knots:3+2*fk.n_knots] = fk.y1
        z_fk[:, 3+2*fk.n_knots] = fk.y1[:, -1]

        z_g = np.zeros((g.y.shape[0], 2*g.y.shape[1]+4))
        z_g[:, 0] = g.y[:, 0] - g.y1[:, 0]*g.lhs_extrap_dist
        z_g[:, 1:1+g.n_knots] = g.y
        z_g[:, 1+g.n_knots] = g.y[:,-1] + g.y1[:,-1]*g.rhs_extrap_dist
        z_g[:, 2+g.n_knots] = g.y1[:, 0]
        z_g[:, 3+g.n_knots:3+2*g.n_knots] = g.y1
        z_g[:, 3+2*g.n_knots] = g.y1[:, -1]

        z_fj = np.array(z_fj)
        z_fk = np.array(z_fk)
        z_g = np.array(z_g)

        z_cart = outer_prod(outer_prod(z_fj, z_fk), z_g)

        return (self.forces_struct_vec @ z_cart.T).T

    def add_to_energy_struct_vec(self, rij, rik, cos, atom_id):
        """Updates structure vectors with given values"""
        abcd_fj = self.fj.get_abcd(rij, 0).ravel()
        abcd_fk = self.fk.get_abcd(rik, 0).ravel()
        abcd_g = self.g.get_abcd(cos, 0).ravel()

        cart = outer_prod_simple(
            outer_prod_simple(abcd_fj, abcd_fk), abcd_g).ravel()

        self.energy_struct_vec[atom_id, :] += cart

    # @profile
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
            self.forces_struct_vec[N*N*a + N*i + i, :] += fj[:, a]
            self.forces_struct_vec[N*N*a + N*j + i, :] -= fj[:, a]

            self.forces_struct_vec[N*N*a + N*i + i, :] += fk[:, a]
            self.forces_struct_vec[N*N*a + N*k + i, :] -= fk[:, a]

    # @profile
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
        botB = np.zeros(n + 2).reshape((1, n + 2))
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
