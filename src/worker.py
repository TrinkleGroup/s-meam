import numpy as np
np.set_printoptions(precision=16)
import logging
import scipy.sparse
from scipy.sparse import lil_matrix
import itertools

from ase.neighborlist import NeighborList
from pympler import muppy, summary

import src.lammpsTools
import src.meam
from src.workerSplines import WorkerSpline, RhoSpline, ffgSpline, USpline

from src.numba_functions import outer_prod_simple

logger = logging.getLogger(__name__)


class Worker:
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials

    Assumptions:
        - knot x-coordinates never change
        - the atomic structure never changes
    """

    # TODO: in general, need more descriptive variable/function names

    # @profile
    def __init__(self, atoms, knot_xcoords, x_indices, types, load_file=False):
        """Organizes data structures and pre-computes structure information.

        Args:
            atoms (ASE.Atoms):
                an ASE representation of an atomic system

            knot_xcoords (np.arr):
                a 1D array of knots points where points are ordered by spline
                type. Spline groups are ordered as [phi, rho, u, f, g] where
                each group has a number of splines depending on the number of
                elements in the system, and each spline has an arbitrary
                number of knots depending on how the potential was created

                e.g. if the system is Ti-O, the potential looks as follows:
                    phi_Ti-Ti, phi_Ti-O, phi_O-O, rho_Ti, rho_O, U_Ti, U_O,
                    f_Ti, f_O, g_Ti-Ti, g_Ti-O, g_O-O

                    where phi_Ti-Ti could have 5 knots, phi_Ti-O 9, rho_O 13,
                    etc.

                This array will be broken apart into individual splines (
                phi_Ti-Ti, phi_Ti-O, ...), then grouped according to spline
                type (phi, rho, u, f, g)

            x_indices (list):
                starting index of each spline. since each spline does not
                necessarily have the same number of knots, a list of indices
                must be provided to deliminate each spline in the 1D vector.

            types (list):
                set of atomic types described by the potential. note: this
                cannot be inferred from 'atoms' since the structure may not
                have every atom type in it.

            load_file (bool): True if loading from HDF5
        """

        if load_file: return

        ntypes          = len(types)
        self.ntypes     = ntypes
        self.natoms     = len(atoms)
        self.pvec_indices = x_indices

        self.len_param_vec = len(knot_xcoords) + 2*len(x_indices)

        f = lambda t: src.lammpsTools.symbol_to_type(t, types)
        self.type_of_each_atom = list(map(f, atoms.get_chemical_symbols()))

        # TODO: rename self.nphi to be more clear
        # there are nphi phi functions and nphi g fxns
        nphi            = int((self.ntypes+1)*self.ntypes/2)
        self.nphi       = nphi

        all_splines = self.build_spline_lists(knot_xcoords, x_indices)

        self.phis = list(all_splines[0])
        self.rhos = list(all_splines[1])
        self.us = list(all_splines[2])
        self.fs = list(all_splines[3])
        self.gs = list(all_splines[4])

        self.ffgs = self.build_ffg_list(self.fs, self.gs)

        # Compute full potential cutoff distance (based only on radial fxns)
        radial_fxns = self.phis + self.rhos + self.fs
        cutoff = np.max([max(s.knots) for s in radial_fxns])

        # Build neighbor lists

        # No double counting of bonds; needed for pair interactions
        nl_noboth = NeighborList(np.ones(self.natoms) * (cutoff / 2.),
                                 self_interaction=False, bothways=False,
                                 skin=0.0)
        nl_noboth.update(atoms)

        # Allows double counting bonds; needed for embedding energy calculations
        nl = NeighborList(np.ones(self.natoms) * (cutoff / 2.),
                          self_interaction=False, bothways=True, skin=0.0)
        nl.update(atoms)

        for i, atom in enumerate(atoms):
            # Record atom type info
            itype = self.type_of_each_atom[i]
            ipos = atom.position

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # Stores pair information for phi
            for j, offset in zip(neighbors_noboth, offsets_noboth):

                jtype = self.type_of_each_atom[j]

                # Find displacement vector (with periodic boundary conditions)
                jpos = atoms[j].position + np.dot(offset, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0]**2 + jvec[1]**2 + jvec[2]**2)
                jvec /= rij

                # Add distance/index/direction information to necessary lists
                phi_idx = src.meam.ij_to_potl(itype, jtype, self.ntypes)

                # phi
                self.phis[phi_idx].add_to_energy_struct_vec(rij)

                self.phis[phi_idx].add_to_forces_struct_vec(rij, jvec, i)
                self.phis[phi_idx].add_to_forces_struct_vec(rij, -jvec, j)

                # rho
                self.rhos[jtype-1].add_to_energy_struct_vec(rij, i)
                self.rhos[itype-1].add_to_energy_struct_vec(rij, j)

                self.rhos[jtype-1].add_to_forces_struct_vec(rij, jvec, i, i)
                self.rhos[jtype-1].add_to_forces_struct_vec(rij, -jvec, j, i)

                self.rhos[itype-1].add_to_forces_struct_vec(rij, jvec, i, j)
                self.rhos[itype-1].add_to_forces_struct_vec(rij, -jvec, j, j)

            # Store distances, angle, and index info for embedding terms
            # TODO: rename j_idx to be more clear
            j_idx = 0  # for tracking neighbor
            for j, offsetj in zip(neighbors, offsets):

                jtype = self.type_of_each_atom[j]

                # offset accounts for periodic images
                jpos = atoms[j].position + np.dot(offsetj, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0]**2 + jvec[1]**2 + jvec[2]**2)
                jvec /= rij

                # prepare for angular calculations
                a = jpos - ipos
                na = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

                fj_idx = jtype - 1

                j_idx += 1
                for k, offsetk in zip(neighbors[j_idx:], offsets[j_idx:]):
                    if k != j:
                        ktype = self.type_of_each_atom[k]
                        kpos = atoms[k].position + np.dot(offsetk,
                                                          atoms.get_cell())

                        kvec = kpos - ipos
                        rik = np.sqrt(kvec[0]**2 + kvec[1]**2 + kvec[2]**2)
                        kvec /= rik

                        b = kpos - ipos
                        nb = np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)

                        cos_theta = np.dot(a, b) / na / nb

                        fk_idx = ktype - 1

                        d0 = jvec
                        d1 = -cos_theta * jvec / rij
                        d2 = kvec / rij
                        d3 = kvec
                        d4 = -cos_theta * kvec / rik
                        d5 = jvec / rik

                        dirs = np.vstack([d0, d1, d2, d3, d4, d5])

                        self.ffgs[fj_idx][fk_idx].add_to_energy_struct_vec(
                            rij, rik, cos_theta, i)

                        self.ffgs[fj_idx][fk_idx].add_to_forces_struct_vec(
                            rij, rik, cos_theta, dirs, i, j, k)

        # convert arrays to avoid having to convert on call
        self.type_of_each_atom = np.array(self.type_of_each_atom)

        for rho in self.rhos:
            # rho.energy_struct_vec = lil_matrix(rho.energy_struct_vec).tocsr()
            rho.structure_vectors['forces'] = rho.structure_vectors['forces'].tocsr()

        for ffg_list in self.ffgs:
            for ffg in ffg_list:

                # ffg.energy_struct_vec =lil_matrix(ffg.energy_struct_vec).tocsr()
                # ffg.forces_struct_vec =lil_matrix(ffg.forces_struct_vec).tocsr()

                ffg.structure_vectors['energy'] =\
                    lil_matrix(ffg.structure_vectors['energy']).tocsr()

                ffg.structure_vectors['forces'] =\
                    lil_matrix(ffg.structure_vectors['forces']).tocsr()

    @classmethod
    def from_hdf5(cls, hdf5_file, name):
        worker_data = hdf5_file[name]

        w = Worker(None, None, None, None, load_file=True)

        w.natoms = int(worker_data.attrs['natoms'])
        w.ntypes = int(worker_data.attrs['ntypes'])
        w.nphi = int(worker_data.attrs['nphi'])
        w.len_param_vec = int(worker_data.attrs['len_param_vec'])

        w.type_of_each_atom = np.array(worker_data['type_of_each_atom'])

        w.phis = [WorkerSpline.from_hdf5(worker_data["phis"], str(i)) for i in
                range(w.nphi)]

        w.rhos = [RhoSpline.from_hdf5(worker_data["rhos"], str(i)) for i in
                range(w.ntypes)]

        w.us = [USpline.from_hdf5(worker_data["us"], str(i)) for i in
                range(w.ntypes)]

        w.fs = [WorkerSpline.from_hdf5(worker_data["fs"], str(i)) for i in
                range(w.ntypes)]

        w.gs = [WorkerSpline.from_hdf5(worker_data["gs"], str(i)) for i in
                range(w.nphi)]

        w.ffgs = [[ffgSpline.from_hdf5(worker_data["ffgs"][str(i)],
                str(j)) for j in range(w.ntypes)] for i in range(w.ntypes)]

        return w

    def add_to_hdf5(self, hdf5_file, name):
        """Adds a worker to an existing HDF5 file

        Args:
            hdf5_file (h5py.File): file to write to
            name (str): name of worker
        """

        new_group = hdf5_file.create_group(name)

        new_group.attrs['natoms'] = self.natoms
        new_group.attrs['ntypes'] = self.ntypes
        new_group.attrs['nphi'] = self.nphi
        new_group.attrs['len_param_vec'] = self.len_param_vec

        new_group.create_dataset("type_of_each_atom",
                data=self.type_of_each_atom)

        phis_group = new_group.create_group("phis")
        for i,sp in enumerate(self.phis): sp.add_to_hdf5(phis_group, str(i))

        rhos_group = new_group.create_group("rhos")
        for i,sp in enumerate(self.rhos): sp.add_to_hdf5(rhos_group, str(i))

        us_group = new_group.create_group("us")
        for i,sp in enumerate(self.us): sp.add_to_hdf5(us_group, str(i))

        fs_group = new_group.create_group("fs")
        for i,sp in enumerate(self.fs): sp.add_to_hdf5(fs_group, str(i))

        gs_group = new_group.create_group("gs")
        for i,sp in enumerate(self.gs): sp.add_to_hdf5(gs_group, str(i))

        ffgs_group = new_group.create_group("ffgs")
        for i,ffg_list in enumerate(self.ffgs):
            mini_group = ffgs_group.create_group(str(i))
            for j,sp in enumerate(ffg_list):
                sp.add_to_hdf5(mini_group, str(j))

    # @profile
    def build_spline_lists(self, knot_xcoords, x_indices):
        """
        Builds lists of phi, rho, u, f, and g WorkerSpline objects

        Args:
            knot_xcoords: joined array of knot coordinates for all splines
            x_indices: starting index in knot_xcoords of each spline

        Returns:
            splines: list of lists of splines; [phis, rhos, us, fs, gs]
        """

        knots_split = np.split(knot_xcoords, x_indices[1:])

        # TODO: could specify bc outside of Worker and pass in
        # bc_type = ('natural', 'natural')
        bc_type = ('fixed', 'fixed')

        splines = []

        for i, knots in enumerate(knots_split):
            if (i < self.nphi):
                s = WorkerSpline(knots, bc_type, self.natoms)
            elif (self.nphi + self.ntypes <= i < self.nphi + 2 *self.ntypes):
                s = USpline(knots, bc_type, self.natoms)
            elif (i >= self.nphi + 2*self.ntypes):
                s = WorkerSpline(knots, bc_type, self.natoms)
            else:
                s = RhoSpline(knots, bc_type, self.natoms)

            s.index = x_indices[i]
            splines.append(s)

        split_indices = [self.nphi, self.nphi + self.ntypes,
                         self.nphi + 2*self.ntypes, self.nphi + 3*self.ntypes]

        return np.split(splines, split_indices)

    def build_ffg_list(self, fs, gs):
        """
        Creates all combinations of f*f*g splines for use with triplet
        calculations.

        Args:
            fs : list of all f WorkerSpline objects
            gs : list of all g WorkerSpline objects

        Returns:
            ffg_list: 2D list where ffg_list[i][j] is the ffgSpline built
                using splines f_i, f_k, and g_ij
        """

        if not self.fs:
            raise ValueError("f splines have not been set yet")
        elif not self.gs:
            raise ValueError("g splines have not been set yet")

        ffg_list = [[] for _ in range(len(self.fs))]
        for j, fj in enumerate(fs):
            for k, fk in enumerate(fs):
                g = gs[src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                ffg_list[j].append(ffgSpline(fj, fk, g, self.natoms))

        return ffg_list

    # @profile
    def compute_energy(self, parameters):
        """Calculates energies for all potentials using information
        pre-computed during initialization.

        Args:
            parameters (np.arr):
                2D list of all parameters for all splines; each row
                corresponds to a unique potential. Each group in a
                single row should have K+2 elements where K is the number
                of knot points, and the 2 additional are boundary conditions.
                The first K in each group are the knot y-values
        """
        parameters = np.array(parameters)
        parameters = np.atleast_2d(parameters)
        # print(parameters)

        self.n_pots = parameters.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        energy = np.zeros(self.n_pots)

        # Pair interactions
        for y, phi in zip(phi_pvecs, self.phis, ):
            energy += phi.calc_energy(y)

        # Embedding terms
        energy += self.embedding_energy(self.compute_ni(rho_pvecs, f_pvecs,\
                                                        g_pvecs), u_pvecs)

        return energy

    # @profile
    def compute_ni(self, rho_pvecs, f_pvecs, g_pvecs):
        """
        Computes ni values for all atoms

        Args:
            rho_pvecs: parameter vectors for rho splines
            f_pvecs: parameter vectors for f splines
            g_pvecs: parameter vectors for g splines

        Returns:
            ni: potential energy
        """
        ni = np.zeros((self.n_pots, self.natoms))

        # Rho contribution
        for y, rho in zip(rho_pvecs, self.rhos):
            ni += rho.calc_energy(y).T

        # Three-body contribution
        for j, (y_fj,ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            for k, (y_fk,ffg) in enumerate(zip(f_pvecs, ffg_list)):

                y_g = g_pvecs[src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                ni += ffg.calc_energy(y_fj, y_fk, y_g)

        return ni

    # @profile
    def embedding_energy(self, ni, u_pvecs):
        """
        Computes embedding energy

        Args:
            ni: per-atom ni values
            u_pvecs: parameter vectors for U splines

        Returns:
            u_energy: total embedding energy
        """

        # TODO: extrap ranges need to be independent of each other
        # TODO: OR -- rescale so that things fit into [0,1]

        # print("WORKER: ni values: {}".format(ni), flush=True)

        u_energy = np.zeros(self.n_pots)
        # print(ni)

        # Evaluate U, U', and compute zero-point energies
        for i,(y,u) in enumerate(zip(u_pvecs, self.us)):
            u.structure_vectors['energy'] = np.zeros((self.n_pots, u.knots.shape[0]+2))

            ni_sublist = ni[:, self.type_of_each_atom - 1 == i]

            num_embedded = ni_sublist.shape[1]

            if num_embedded > 0:
                u.add_to_energy_struct_vec(ni_sublist)
                u_energy += u.calc_energy(y)

            u.reset()

        return u_energy

    # @profile
    def evaluate_uprimes(self, ni, u_pvecs, second=False):
        """
        Computes U' values for every atom

        Args:
            ni: per-atom ni values
            u_pvecs: parameter vectors for U splines
            second (bool): also compute second derivatives

        Returns:
            uprimes: per-atom U' values
        """

        tags = np.arange(self.natoms)

        # -1 because LAMMPS-style atom type is 1 indexed
        shifted_types = self.type_of_each_atom - 1

        for i, u in enumerate(self.us):

            # get atom ids of type i
            indices = tags[shifted_types == i]

            u.structure_vectors['deriv'] = np.zeros(
                (self.n_pots, self.natoms, u.knots.shape[0]+2))

            if second:
                u.structure_vectors['2nd_deriv'] = np.zeros(
                    (self.n_pots, self.natoms, u.knots.shape[0]+2))

            if indices.shape[0] > 0:
                u.add_to_deriv_struct_vec(ni[:, shifted_types == i], indices)

                if second:
                    u.add_to_2nd_deriv_struct_vec(ni[:, shifted_types == i], indices)

        # Evaluate U, U', and compute zero-point energies
        uprimes = np.zeros((self.n_pots, self.natoms))

        if second: uprimes_2 = np.zeros((self.n_pots, self.natoms))

        for y, u in zip(u_pvecs, self.us):
            uprimes += u.calc_deriv(y)

            if second: uprimes_2 += u.calc_2nd_deriv(y)

        if second: return uprimes, uprimes_2
        else: return uprimes

    # @profile
    def compute_forces(self, parameters):
        """Calculates the force vectors on each atom using the given spline
        parameters.

        Args:
            parameters (np.arr): the 1D array of concatenated parameter
                vectors for all splines in the system
        """
        parameters = np.atleast_2d(parameters)
        self.n_pots = parameters.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        forces = np.zeros((self.n_pots, self.natoms, 3))

        # Pair forces (phi)
        for phi_idx, (phi, y) in enumerate(zip(self.phis, phi_pvecs)):
            forces += phi.calc_forces(y)

        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(ni, u_pvecs)

        # Electron density embedding (rho)

        embedding_forces = np.zeros((self.n_pots, 3*self.natoms*self.natoms))

        for rho_idx, (rho, y) in enumerate(zip(self.rhos, rho_pvecs)):

            rho_forces = rho.calc_forces(y)

            embedding_forces += rho_forces

        # Angular terms (ffg)
        for j, ffg_list in enumerate(self.ffgs):
            for k, ffg in enumerate(ffg_list):

                y_fj = f_pvecs[j]
                y_fk = f_pvecs[k]
                y_g = g_pvecs[src.meam.ij_to_potl(j+1, k+1, self.ntypes)]

                ffg_forces = ffg.calc_forces(y_fj, y_fk, y_g)

                embedding_forces += ffg_forces

        N = self.natoms

        # replaces einsum, but for a 2D matrix; logic needed for sparse matrices
        # for atom_id in range(self.natoms):
            # embedding_forces[:, 0*N*N + N*atom_id: 0*N*N + N*atom_id + self.natoms] = \
            #     np.multiply(embedding_forces[:, 0*N*N + N*atom_id: 0*N*N + N*atom_id + self.natoms], uprimes)
            # embedding_forces[:, 1*N*N + N*atom_id: 1*N*N + N*atom_id + self.natoms] = \
            #     np.multiply(embedding_forces[:, 1*N*N + N*atom_id: 1*N*N + N*atom_id + self.natoms], uprimes)
            # embedding_forces[:, 2*N*N + N*atom_id: 2*N*N + N*atom_id + self.natoms] = \
            #     np.multiply(embedding_forces[:, 2*N*N + N*atom_id: 2*N*N + N*atom_id + self.natoms], uprimes)

        embedding_forces = embedding_forces.reshape((self.n_pots, 3, N, N))
        # embedding_forces = np.sum(embedding_forces, axis=3)
        # embedding_forces = embedding_forces.T.reshape((self.n_pots, self.natoms, 3))

        embedding_forces = np.einsum('pijk,pk->pji', embedding_forces, uprimes)

        return forces + embedding_forces

    # @profile
    def parse_parameters(self, parameters):
        """Separates the pre-ordered array of vectors of all spline parameters
        into groups.

        Args:
            parameters (np.arr):
                2D array of knot points and boundary conditions for ALL
                splines for ALL intervals for ALL potentials

        Returns:
            *_pvecs (np.arr):
                each return is a list of arrays of parameters. e.g.
                phi_pvecs[0] is the parameters for the first phi spline for
                every potential
        """

        splines = self.phis + self.rhos + self.us + self.fs + self.gs

        # Parse parameter vector
        x_indices = [s.index for s in splines]
        y_indices = [x_indices[i] + 2 * i for i in range(len(x_indices))]

        params_split = np.split(parameters, y_indices[1:], axis=1)

        nphi = self.nphi
        ntypes = self.ntypes

        phi_pvecs = params_split[:nphi]
        rho_pvecs = params_split[nphi: nphi + ntypes]
        u_pvecs = params_split[nphi + ntypes:nphi + 2*ntypes]
        f_pvecs = params_split[nphi + 2*ntypes:nphi + 3*ntypes]
        g_pvecs = params_split[nphi + 3*ntypes:]

        return phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs

    def energy_gradient_wrt_pvec(self, pvec):
        parameters = np.atleast_2d(pvec)

        gradient = np.zeros(self.len_param_vec)

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(np.atleast_2d(parameters))

        self.n_pots = parameters.shape[0]

        grad_index = 0
        # gradients of phi are just their structure vectors
        for y, phi in zip(phi_pvecs, self.phis):
            gradient[grad_index:grad_index + y.shape[1]] += \
                phi.structure_vectors['energy']

            grad_index += y.shape[1]

        # chain rule on U functions means dU/dn values are needed
        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(ni, u_pvecs)

        for y, rho in zip(rho_pvecs, self.rhos):

            partial_ni = rho.structure_vectors['energy']

            gradient[grad_index:grad_index + y.shape[1]] += \
                (uprimes @ partial_ni).ravel()

            grad_index += y.shape[1]

        # add in first term of chain rule
        for i,(y,u) in enumerate(zip(u_pvecs, self.us)):
            u.structure_vectors['energy'] = np.zeros((self.n_pots, u.knots.shape[0]+2))

            ni_sublist = ni[:, self.type_of_each_atom - 1 == i]

            num_embedded = ni_sublist.shape[1]

            if num_embedded > 0:
                u.add_to_energy_struct_vec(ni_sublist)

                gradient[grad_index:grad_index + y.shape[1]] += \
                    u.structure_vectors['energy'].ravel()

            grad_index += y.shape[1]

            u.reset()

        # build list of indices for later use
        tmp_index = grad_index
        ffg_indices = [grad_index]

        for y_fj in f_pvecs:
            ffg_indices.append(tmp_index + y_fj.shape[1])
            tmp_index += y_fj.shape[1]

        for y_g in g_pvecs:
            ffg_indices.append(tmp_index + y_g.shape[1])
            tmp_index += y_g.shape[1]

        # add in second term of chain rule
        for j, (y_fj, ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            n_fj = y_fj.shape[1]
            y_fj = y_fj.ravel()

            for k, (y_fk, ffg) in enumerate(zip(f_pvecs, ffg_list)):
                g_idx = src.meam.ij_to_potl(j+1, k+1, self.ntypes)

                y_g = g_pvecs[g_idx]

                n_fk = y_fk.shape[1]
                n_g = y_g.shape[1]

                y_fk = y_fk.ravel()
                y_g = y_g.ravel()

                cart_y = my_outer_prod(y_fj, y_fk, y_g)

                # every ffgSpline affects grad(f_j), grad(f_k), and grad(g)

                # grad(f_j) contribution
                tmp = scipy.sparse.spdiags(cart_y, 0, len(cart_y), len(cart_y))

                scaled_sv = ffg.structure_vectors['energy'] * tmp
                scaled_sv = uprimes * scaled_sv

                fj_contrib = np.split(scaled_sv, n_fj, axis=1)

                for l in range(n_fj):
                    block = fj_contrib[l] / y_fj[l]
                    block = np.nan_to_num(block)

                    gradient[ffg_indices[j] + l] += np.sum(block)

                # grad(f_k) contribution

                fk_contrib = np.split(scaled_sv, n_fj*n_fk, axis=1)
                fk_contrib = np.array(fk_contrib)

                for l in range(n_fk):
                    sample_indices = np.arange(l, n_fj*n_fk, n_fk)

                    block = fk_contrib[sample_indices, :, :] / y_fk[l]
                    block = np.nan_to_num(block)

                    gradient[ffg_indices[k] + l] += np.sum(block)

                # grad(g) contribution

                g_contrib = np.split(scaled_sv, n_fj*n_fk, axis=1)
                g_contrib = np.array(g_contrib)

                for l in range(n_g):
                    block = g_contrib[:, :, l] / y_g[l]
                    block = np.nan_to_num(block)

                    gradient[ffg_indices[self.ntypes + g_idx] + l] += np.sum(block)

        return gradient

    def forces_gradient_wrt_pvec(self, pvec):
        parameters = np.atleast_2d(pvec)

        gradient = np.zeros((self.natoms, 3, self.len_param_vec))

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(np.atleast_2d(parameters))

        self.n_pots = parameters.shape[0]

        grad_index = 0

        # gradients of phi are just their structure vectors
        for y, phi in zip(phi_pvecs, self.phis):
            gradient[:, :, grad_index:grad_index + y.shape[1]] += \
                phi.structure_vectors['forces'].reshape((self.natoms, 3, y.shape[1]))

            grad_index += y.shape[1]

        # chain rule on U functions means dU/dn values are needed
        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)

        N = self.natoms

        uprimes, uprimes_2 = self.evaluate_uprimes(ni, u_pvecs, second=True)

        uprimes_diag = scipy.sparse.spdiags(uprimes, 0, N, N)
        uprimes2_diag = scipy.sparse.spdiags(uprimes_2, 0, N, N)

        embedding_forces = np.zeros((3, self.natoms, self.natoms))

        # rho gradient term; there is a U'' and a U' term for each rho
        for r_index, (y, rho) in enumerate(zip(rho_pvecs, self.rhos)):

            rho_e_sv = rho.structure_vectors['energy']
            rho_sv = rho.structure_vectors['forces'].toarray()

            rho_sv = rho_sv.reshape(
                (3, self.natoms, self.natoms, y.shape[1]))

            # U'' term
            rho_forces = rho.calc_forces(y).reshape((3,self.natoms,self.natoms))

            uprimes_scaled = np.einsum('i,ij->ij', uprimes_2.ravel(), rho_e_sv)

            for rho2,y2 in zip(self.rhos, rho_pvecs):
                rho2_forces = rho2.calc_forces(y2).reshape((3,self.natoms,self.natoms))

                final = np.einsum('ij,kli->lkj', uprimes_scaled, rho2_forces)

                gradient[:, :, grad_index:grad_index + y.shape[1]] += final

            # U' term

            up_contracted_sv = np.einsum('ijkl,k->ijl', rho_sv, uprimes.ravel())

            gradient[:, :, grad_index:grad_index + y.shape[1]] += \
                np.transpose(up_contracted_sv, axes=(1,0,2))

            # Used for U gradient term
            embedding_forces += rho_forces

            grad_index += y.shape[1]

        # save indices so that embedding_forces can be added later
        tmp_U_indices = []

        # prep for U gradient term
        for i,(y,u) in enumerate(zip(u_pvecs, self.us)):
            tmp_U_indices.append((grad_index, grad_index + y.shape[1]))
            grad_index += y.shape[1]

        # build list of indices for later use
        tmp_index = grad_index
        ffg_indices = [grad_index]

        for y_fj in f_pvecs:
            ffg_indices.append(tmp_index + y_fj.shape[1])
            tmp_index += y_fj.shape[1]

        for y_g in g_pvecs:
            ffg_indices.append(tmp_index + y_g.shape[1])
            tmp_index += y_g.shape[1]

        # ffg gradient terms
        for j, (y_fj, ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            n_fj = y_fj.shape[1]
            y_fj = y_fj.ravel()

            for k, (y_fk, ffg) in enumerate(zip(f_pvecs, ffg_list)):
                g_idx = src.meam.ij_to_potl(j+1, k+1, self.ntypes)

                y_g = g_pvecs[g_idx]

                n_fk = y_fk.shape[1]
                n_g = y_g.shape[1]

                y_fk = y_fk.ravel()
                y_g = y_g.ravel()

                cart_y = my_outer_prod(y_fj, y_fk, y_g)

                # every ffgSpline affects grad(f_j), grad(f_k), and grad(g)

                # grad(f_j) contribution

                ffg_sv = ffg.structure_vectors['forces']
                ffg_e_sv = ffg.structure_vectors['energy']

                ffg_forces = (ffg_sv @ cart_y.T).T
                # ffg_forces = ffg_forces.reshape((3, self.natoms, self.natoms))

                embedding_forces += ffg_forces.reshape((3, self.natoms, self.natoms))

                # U'' term
                final_upp_part = lil_matrix((self.natoms*3, len(cart_y)))

                # for rho2,y2 in zip(self.rhos, rho_pvecs):
                for j2, (y_fj2,ffg_list2) in enumerate(zip(f_pvecs,self.ffgs)):
                    y_fj2 = y_fj2.ravel()

                    for k2, (y_fk2,ffg2) in enumerate(zip(f_pvecs, ffg_list2)):
                        ffg_sv2 = ffg2.structure_vectors['forces']

                        g_idx2 = src.meam.ij_to_potl(j2+1, k2+1, self.ntypes)

                        y_g2 = g_pvecs[g_idx2]

                        y_fk2 = y_fk2.ravel()
                        y_g2 = y_g2.ravel()

                        cart_y2 = my_outer_prod(y_fj2, y_fk2, y_g2)

                        ffg_forces2 = (ffg_sv2 @ cart_y2.T).T
                        # ffg_forces2 = ffg_forces2.reshape((3,self.natoms,
                        #                                    self.natoms))

                        scaled_by_up2 = uprimes2_diag * ffg_e_sv

                        joined_svs = scaled_by_up2.T.dot(
                            lil_matrix(ffg_forces2.reshape(3*N, N).T).tocsr())

                        final_upp_part += joined_svs.T

                # U' term
                tmp = scipy.sparse.spdiags(cart_y, 0, len(cart_y), len(cart_y))

                scaled_by_knots = ffg.structure_vectors['forces'] * tmp

                # element-wise multiply by original sv for U chain rule
                # scaled_sv = scaled_sv.multiply(ffg_sv)

                # replaces einsum, but for a sparse matrix; contracts by U', U''
                contracted_by_up = scipy.sparse.csr_matrix((3*N, len(cart_y)))
                # contracted_by_upp = scipy.sparse.csr_matrix((3*N, len(cart_y)))

                for atom_id in range(self.natoms):
                    tmp0 = N*atom_id
                    tmp1 = N*N + tmp0
                    tmp2 = 2*N*N + tmp0

                    contracted_by_up[0*N + atom_id, :] = (uprimes_diag*
                          scaled_by_knots[tmp0: tmp0 + self.natoms, :]).sum(axis=0)

                    contracted_by_up[1*N + atom_id, :] = (uprimes_diag*
                          scaled_by_knots[tmp1: tmp1 + self.natoms, :]).sum(axis=0)

                    contracted_by_up[2*N + atom_id, :] = (uprimes2_diag*
                          scaled_by_knots[tmp2: tmp2 + self.natoms, :]).sum(axis=0)

                combined = contracted_by_up + final_upp_part

                n_fk_g = n_fk*n_g

                # grad(f_j) contribution
                for l in range(n_fj):
                    block = combined[:, n_fj*l : n_fk_g*(l+1)].sum(axis=1)
                    block = block.reshape((self.natoms, 3))

                    if y_fj[l] != 0: block /= y_fj[l]
                    block = np.nan_to_num(block)

                    gradient[:, :, ffg_indices[j] + l] += block

                # grad(f_k) contribution
                for l in range(n_fk):
                    sample_indices = [e1*e2 for e1,e2 in itertools.product(
                        range(l, n_fj*n_fk, n_fk), range(n_g))]

                    block = combined[:, sample_indices].sum(axis=1)
                    block = block.reshape((self.natoms, 3))

                    if y_fk[l] != 0: block /= y_fk[l]
                    block = np.nan_to_num(block)

                    gradient[:, :, ffg_indices[k] + l] += block

                # grad(g) contribution
                for l in range(n_g):
                    sample_indices = range(l, n_fj*n_fk*n_g, n_g)

                    # block = g_contrib[:, :, l]
                    block = combined[:, sample_indices].sum(axis=1)
                    block = block.reshape((self.natoms, 3))

                    if y_g[l] != 0: block /= y_g[l]
                    block = np.nan_to_num(block)

                    gradient[:,:, ffg_indices[self.ntypes + g_idx] + l] += block

        for i, (indices,u) in enumerate(zip(tmp_U_indices, self.us)):
            tmp_U_indices.append((grad_index, grad_index + y.shape[1]))
            start, stop = indices

            u_term = np.einsum('ij,kli->lkj', u.structure_vectors['deriv'][0],
                        embedding_forces)

            # gradient[self.type_of_each_atom - 1 == i:, :, start:stop] += u_term
            gradient[:, :, start:stop] += u_term

        return gradient

def my_outer_prod(y1, y2, y3):
    cart1 = np.einsum("i,j->ij", y1, y2)
    cart1 = cart1.reshape((cart1.shape[0]*cart1.shape[1]))

    cart2 = np.einsum("i,j->ij", cart1, y3)
    return cart2.reshape((cart2.shape[0]*cart2.shape[1]))

# @profile
def main():
    np.random.seed(42)
    import src.meam
    from tests.testPotentials import get_random_pots, get_constant_potential
    from tests.testStructs import allstructs, dimers

    pot = get_random_pots(1)['meams'][0]
    # pot = get_constant_potential()
    x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    from ase import Atom
    a_new = Atom()
    a_new.position = [20, 0, 0]
    a_new.symbol = "He"

    atoms = allstructs['aba']

    worker = Worker(atoms, x_pvec, indices, pot.types)

    h = 1e-8
    N = y_pvec.ravel().shape[0]

    cd_points = np.array([y_pvec] * N*2)
    # cd_points = np.array([y_pvec] * N)

    for l in range(N):
        # cd_points[l, l] += h
        cd_points[2*l, l] += h
        cd_points[2*l+1, l] -= h

    # cd_evaluated = worker.compute_energy(np.array(cd_points))
    cd_evaluated = worker.compute_forces(np.array(cd_points))

    # fd_gradient = np.zeros(N)
    fd_gradient = np.zeros((worker.natoms, 3, worker.len_param_vec))

    for l in range(N):
        # fd_gradient[l] = (cd_evaluated[l] - fx) / h
        # fd_gradient[l] = (cd_evaluated[2*l] - cd_evaluated[2*l+1]) / h / 2
        fd_gradient[:, :, l] = (cd_evaluated[2*l] - cd_evaluated[2*l+1]) / h / 2

    splines = worker.phis + worker.rhos + worker.us + worker.fs + worker.gs

    x_indices = [s.index for s in splines]
    y_indices = [x_indices[i] + 2 * i for i in range(len(x_indices))]

    # grad = worker.energy_gradient_wrt_pvec(y_pvec)
    grad = worker.forces_gradient_wrt_pvec(y_pvec)

    split = np.array_split(grad, y_indices)[1:]
    split2 = np.array_split(fd_gradient, y_indices)[1:]

    np.set_printoptions(linewidth=np.infty)

    # print("Direct method")
    # for l in split:
    #     print(l)
    #
    # print()
    # print("Finite differences")
    # for l in split2:
    #     print(l)
    #
    # print()
    # print("Difference")
    # diff = np.abs(grad - fd_gradient)
    # split3 = np.array_split(diff, y_indices)[1:]
    #
    # for l in split3:
    #     print(l)
    #
    # print()
    diff = np.abs(grad - fd_gradient)

    # print("Guess:", grad[:,0,39:])
    piece = diff[:,0,39:]

    print(piece)
    print("Max difference:", np.max(piece))

if __name__ == "__main__":
    main()
