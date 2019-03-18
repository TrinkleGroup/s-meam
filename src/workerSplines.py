import numpy as np
import logging
import h5py
from scipy.interpolate import CubicSpline

from scipy.sparse import diags, lil_matrix, csr_matrix
# from src.numba_functions import onepass_min_max, outer_prod_1d

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
        self.structure_vectors['forces'] = np.zeros((natoms, 3, self.n_knots+2))
        # self.structure_vectors['forces'] = np.zeros((natoms, self.n_knots+2, 3))

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

        # mn, mx = onepass_min_max(x)
        mn = np.min(x)
        mx = np.max(x)

        lhs_extrap_dist = max(float(self.extrap_dist), self.knots[0] - mn)
        rhs_extrap_dist = max(float(self.extrap_dist), mx - self.knots[-1])

        # add ghost knots
        knots = list([self.knots[0] - lhs_extrap_dist]) + self.knots.tolist() +\
                list([self.knots[-1] + rhs_extrap_dist])

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

        if deriv == 0:

            A = 2*t3 - 3*t2 + 1
            B = t3 - 2*t2 + t
            C = -2*t3 + 3*t2
            D = t3 - t2

        elif deriv == 1:

            A = 6*t2 - 6*t
            B = 3*t2 - 4*t + 1
            C = -6*t2 + 6*t
            D = 3*t2 - 2*t

        elif deriv == 2:

            A = 12*t - 6
            B = 6*t - 4
            C = -12*t + 6
            D = 6*t - 2
        else:
            raise ValueError("Only allowed derivative values are 0, 1, and 2")

        scaling = 1 / prefactor
        scaling = scaling**deriv

        B *= prefactor
        D *= prefactor

        A *= scaling
        B *= scaling
        C *= scaling
        D *= scaling

        alpha = np.zeros((len(x), self.n_knots))
        beta = np.zeros((len(x), self.n_knots))

        # values being extrapolated need to be indexed differently
        lhs_extrap_mask = spline_bins == 0
        rhs_extrap_mask = spline_bins == self.n_knots

        lhs_extrap_indices = np.arange(len(x))[lhs_extrap_mask]
        rhs_extrap_indices = np.arange(len(x))[rhs_extrap_mask]

        if True in lhs_extrap_mask:
            alpha[lhs_extrap_indices, 0] += A[lhs_extrap_mask]
            alpha[lhs_extrap_indices, 0] += C[lhs_extrap_mask]

            beta[lhs_extrap_indices, 0] += A[lhs_extrap_mask]*(-lhs_extrap_dist)
            beta[lhs_extrap_indices, 0] += B[lhs_extrap_mask]
            beta[lhs_extrap_indices, 0] += D[lhs_extrap_mask]

        if True in rhs_extrap_mask:
            alpha[rhs_extrap_indices, -1] += A[rhs_extrap_mask]
            alpha[rhs_extrap_indices, -1] += C[rhs_extrap_mask]

            beta[rhs_extrap_indices, -1] += B[rhs_extrap_mask]
            beta[rhs_extrap_indices, -1] += C[rhs_extrap_mask]*rhs_extrap_dist
            beta[rhs_extrap_indices, -1] += D[rhs_extrap_mask]

        # now add internal knots
        internal_mask = np.logical_not(lhs_extrap_mask + rhs_extrap_mask)

        shifted_indices = spline_bins[internal_mask] - 1

        np.add.at(alpha, (np.arange(len(x))[internal_mask], shifted_indices),
                  A[internal_mask])
        np.add.at(alpha, (np.arange(len(x))[internal_mask], shifted_indices + 1),
                  C[internal_mask])

        np.add.at(beta, (np.arange(len(x))[internal_mask], shifted_indices),
                  B[internal_mask])
        np.add.at(beta, (np.arange(len(x))[internal_mask], shifted_indices + 1),
                  D[internal_mask])

        big_alpha = np.concatenate([alpha, np.zeros((len(x), 2))], axis=1)

        gamma = np.einsum('ij,ik->kij', self.M, beta.T)

        return big_alpha + np.sum(gamma, axis=1)

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
        # self.structure_vectors['energy'] += self.get_abcd(values)
        self.structure_vectors['energy'] += np.sum(self.get_abcd(values), axis=0)

    def add_to_forces_struct_vec(self, values, dirs, atom_id):
        dirs = np.array(dirs)

        # abcd = self.get_abcd(values, 1).ravel()
        abcd = np.sum(self.get_abcd(values, 1), axis=0).ravel()

        self.structure_vectors['forces'][atom_id, :, :] += np.einsum('i,j->ji', abcd, dirs)

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
        return np.einsum('ijk,pk->pij', self.structure_vectors['forces'], y)

class RhoSpline(WorkerSpline):
    """RhoSpline objects are a variant of the WorkerSpline, but that require
    tracking energies for each atom. To account for potential many-atom
    simulation cells, sparse matrices are used for the forces struct vec"""

    def __init__(self, knots, bc_type, natoms, M=None):
        super(RhoSpline, self).__init__(knots, bc_type, natoms, M)

        self.structure_vectors['energy'] = np.zeros((self.natoms, self.n_knots+2))

        N = self.natoms
        self.structure_vectors['forces'] = np.zeros((3*N*N, self.n_knots+2))
        # self.structure_vectors['forces'] = lil_matrix((3*N*N, self.n_knots+2),dtype=float)

    def add_to_hdf5(self, hdf5_file, name, save_sv=False):
        """The 'save_sv' argument allows saving as a sparse matrix"""
        super().add_to_hdf5(hdf5_file, name, save_sv=False)

        spline_group = hdf5_file[name]

        spline_group.create_dataset("e_sv",
                data=self.structure_vectors['energy'])

        f_sv = self.structure_vectors['forces']
        spline_group.create_dataset('f_sv', data=f_sv)
        # spline_group.create_dataset('f_sv.data', data=f_sv.data)
        # spline_group.create_dataset('f_sv.indices', data=f_sv.indices)
        # spline_group.create_dataset('f_sv.indptr', data=f_sv.indptr)
        # spline_group.create_dataset('f_sv.shape', data=f_sv.shape)

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

        rho.structure_vectors['energy'] = np.array(spline_data['e_sv'])

        # f_sv_data = np.array(spline_data['f_sv.data'])
        # f_sv_indices = np.array(spline_data['f_sv.indices'])
        # f_sv_indptr = np.array(spline_data['f_sv.indptr'])
        # f_sv_shape = np.array(spline_data['f_sv.shape'])

        # rho.structure_vectors['forces'] = csr_matrix(
        #         (f_sv_data, f_sv_indices, f_sv_indptr), shape=f_sv_shape)
        rho.structure_vectors['forces'] = np.array(spline_data['f_sv'])

        rho.n_knots = len(x)

        return rho

    def add_to_energy_struct_vec(self, values, atom_id):
        # self.structure_vectors['energy'][atom_id, :] += self.get_abcd(values)
        self.structure_vectors['energy'][atom_id, :] += \
            np.sum(self.get_abcd(values), axis=0)

    def add_to_forces_struct_vec(self, value, dir, i, j):
        """Single add used because need to speciy two atom tags"""
        # abcd_3d = np.einsum('i,j->ij', self.get_abcd(value, 1).ravel(), dir)
        abcd_3d = np.einsum('i,j->ij',
            np.sum(self.get_abcd(value, 1), axis=0).ravel(), dir)

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

        ffg.structure_vectors['energy'] = ffg_data['e_sv']

        ffg.structure_vectors['forces'] = ffg_data['f_sv']

        return ffg

    def add_to_hdf5(self, hdf5_file, name):
        new_group = hdf5_file.create_group(name)

        new_group.attrs['natoms'] = self.natoms

        new_group.create_dataset('e_sv', data=self.structure_vectors['energy'])

        new_group.create_dataset('f_sv', data=self.structure_vectors['forces'])

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

        # abcd_fj = self.fj.get_abcd(rij, 0).ravel()
        # abcd_fk = self.fk.get_abcd(rik, 0).ravel()
        # abcd_g = self.g.get_abcd(cos, 0).ravel()
        abcd_fj = np.sum(self.fj.get_abcd(rij, 0), axis=0).ravel()
        abcd_fk = np.sum(self.fk.get_abcd(rik, 0), axis=0).ravel()
        abcd_g  = np.sum(self.g.get_abcd(cos, 0), axis=0).ravel()

        n_fj = abcd_fj.shape[0]
        n_fk = abcd_fk.shape[0]
        n_g = abcd_g.shape[0]

        cart = np.outer(np.outer(abcd_fj, abcd_fk), abcd_g).ravel()
        # cart = np.zeros(n_fj*n_fk*n_g)
        # outer_prod_1d(abcd_fj, abcd_fk, abcd_g, n_fj, n_fk, n_g, cart)

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

        # fj_1, fk_1, g_1 = self.get_abcd(rij, rik, cos, [1, 0, 0])
        # fj_2, fk_2, g_2 = self.get_abcd(rij, rik, cos, [0, 1, 0])
        # fj_3, fk_3, g_3 = self.get_abcd(rij, rik, cos, [0, 0, 1])
        fj_1, fk_1, g_1 = self.get_abcd(rij, rik, cos, [1, 0, 0])
        fj_2, fk_2, g_2 = self.get_abcd(rij, rik, cos, [0, 1, 0])
        fj_3, fk_3, g_3 = self.get_abcd(rij, rik, cos, [0, 0, 1])

        v1 = np.outer(np.outer(fj_1, fk_1), g_1).ravel()
        v2 = np.outer(np.outer(fj_2, fk_2), g_2).ravel()
        v3 = np.outer(np.outer(fj_3, fk_3), g_3).ravel()

        # v1 = np.zeros(fj_1.shape[0]*fk_1.shape[0]*g_1.shape[0])
        # v2 = np.zeros(fj_2.shape[0]*fk_2.shape[0]*g_2.shape[0])
        # v3 = np.zeros(fj_3.shape[0]*fk_3.shape[0]*g_3.shape[0])
        #
        # outer_prod_1d(fj_1, fk_1, g_1, fj_1.shape[0], fk_1.shape[0],
        #                    g_1.shape[0], v1) # fj' fk g
        # outer_prod_1d(fj_2, fk_2, g_2, fj_2.shape[0], fk_2.shape[0],
        #                    g_2.shape[0], v2) # fj' fk g
        # outer_prod_1d(fj_3, fk_3, g_3, fj_3.shape[0], fk_3.shape[0],
        #                    g_3.shape[0], v3) # fj' fk g

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
        N2 = N*N
        for a in range(3):
            self.structure_vectors['forces'][N2*a + N*i + i, :] += fj[:, a]
            self.structure_vectors['forces'][N2*a + N*j + i, :] -= fj[:, a]

            self.structure_vectors['forces'][N2*a + N*i + i, :] += fk[:, a]
            self.structure_vectors['forces'][N2*a + N*k + i, :] -= fk[:, a]

class USpline(WorkerSpline):
    """Although U splines can't take as much advantage of pre-computing
    values, it is still useful to use a similar representation as a
    WorkerSpline because it is ideal for evaluating many pvecs simultaneously"""

    def __init__(self, knots, bc_type, natoms, M=None):
        super(USpline, self).__init__(knots, bc_type, natoms, M)

        self.structure_vectors['deriv'] = np.zeros((1, natoms, len(self.knots)+2))
        self.structure_vectors['2nd_deriv'] = np.zeros((1, natoms, len(self.knots)+2))
        self.structure_vectors['energy'] = None
        # self.structure_vectors['energy'] = np.zeros(len(self.knots) + 2)

        # distance for extrapolating to 0; saved separately to avoid overwrite
        self.zero_extrap_dist = self.extrap_dist

        if self.knots[0] > 0:
            self.zero_extrap_dist = self.knots[0]
        elif self.knots[-1] < 0:
            self.zero_extrap_dist = abs(self.knots[-1])

        # self.zero_abcd = self.get_abcd([0])
        self.atoms_embedded = 0

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

        us.zero_abcd = np.array(spline_data['zero_abcd'])
        us.atoms_embedded = np.array(spline_data.attrs['atoms_embedded'])

        us.n_knots = len(x)

        return us

    def add_to_hdf5(self, hdf5_file, name):
        super().add_to_hdf5(hdf5_file, name, save_sv=False)

        uspline_group = hdf5_file[name]

        uspline_group.create_dataset('deriv_struct_vec',
             data=self.structure_vectors['deriv'])

        uspline_group.create_dataset('zero_abcd', data=self.zero_abcd)
        uspline_group.attrs['atoms_embedded'] = self.atoms_embedded

    def update_knot_positions(self, lhs_knot, rhs_knot, npots):

        # TODO: add a check to only update if noticeably different from current

        self.knots = np.linspace(lhs_knot, rhs_knot, self.n_knots)
        self.M = build_M(self.n_knots, self.knots[1] - self.knots[0], self.bc_type)
        self.extrap_dist = (rhs_knot - lhs_knot) / 2

        self.structure_vectors['energy'] = np.zeros((npots, self.n_knots + 2))

    def reset(self):
        self.atoms_embedded = 0
        self.structure_vectors['energy'][:] = 0
        self.structure_vectors['deriv'][:] = 0
        self.structure_vectors['2nd_deriv'][:] = 0

    def add_to_energy_struct_vec(self, values, lhs_knot, rhs_knot):
        num_new_atoms = values.shape[1]

        self.update_knot_positions(lhs_knot, rhs_knot, values.shape[0])

        if num_new_atoms > 0:
            self.atoms_embedded += num_new_atoms

            values = np.atleast_1d(values)
            org_shape = values.shape
            flat_values = values.ravel()

            # abcd = self.get_abcd(flat_values, lhs_knot, rhs_knot, nknots)
            abcd = self.get_abcd(flat_values)
            abcd = abcd.reshape(list(org_shape) + [abcd.shape[1]])

            self.structure_vectors['energy'] += np.sum(abcd, axis=1)
            # self.structure_vectors['energy'] -= self.zero_abcd * num_new_atoms

    def add_to_deriv_struct_vec(self, values, indices, lhs_knot, rhs_knot):
        self.update_knot_positions(lhs_knot, rhs_knot, values.shape[0])

        if values.shape[0] > 0:

            values = np.atleast_1d(values)
            org_shape = values.shape
            flat_values = values.ravel()

            abcd = self.get_abcd(flat_values, 1)
            abcd = abcd.reshape(list(org_shape) + [abcd.shape[1]])

            self.structure_vectors['deriv'][:, indices, :] = abcd

    def add_to_2nd_deriv_struct_vec(self, values, indices, lhs_knot, rhs_knot):
        self.update_knot_positions(lhs_knot, rhs_knot, values.shape[0])

        if values.shape[0] > 0:

            values = np.atleast_1d(values)
            org_shape = values.shape
            flat_values = values.ravel()

            abcd = self.get_abcd(flat_values, 2)
            abcd = abcd.reshape(list(org_shape) + [abcd.shape[1]])

            self.structure_vectors['2nd_deriv'][:, indices, :] = abcd

    def calc_energy(self, y):
        return np.einsum("ij,ij->i", self.structure_vectors['energy'], y)

    def calc_deriv(self, y):
        return np.einsum('ijk,ik->ij', self.structure_vectors['deriv'], y)

    def calc_2nd_deriv(self, y):
        return np.einsum('ijk,ik->ij', self.structure_vectors['2nd_deriv'], y)

    def compute_zero_potential(self, y, n):
        """Calculates the value of the potential as if every entry in the
        structure vector was a zero.

        Args:
            y (np.arr): array of parameter vectors
            n (int): number of embedded atoms

        Returns:
            the value evaluated by the spline using num_zeros zeros"""

        return (self.zero_abcd @ y.T).T*n

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
