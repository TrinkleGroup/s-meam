import sys
sys.path.append('./');

import time
import numpy as np
np.set_printoptions(precision=16)
import logging
import scipy.sparse
from scipy.sparse import lil_matrix
import itertools
from operator import itemgetter
# from memory_profiler import profile
import collections

from ase.neighborlist import NeighborList
from pympler import muppy, summary

import src.lammpsTools
import src.meam
from src.workerSplines import WorkerSpline, RhoSpline, ffgSpline, USpline

# from src.numba_functions import outer_prod_1d, outer_prod_1d_2vecs
# from numba import jit


logger = logging.getLogger(__name__)


class Worker:
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials

    Assumptions:
        - knot x-coordinates never change
        - the atomic structure never changes
    """

    # TODO: in general, need more descriptive variable/function names

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

        self.ffg_grad_indices = self.compute_grad_indices(self.ffgs)

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

        all_rij = []
        all_costheta = []

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

                all_rij.append(rij)

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
                        all_costheta.append(cos_theta)

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

    def compute_energy(self, parameters, u_ranges):
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

        self.n_pots = parameters.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        energy = np.zeros(self.n_pots)

        # Pair interactions
        for y, phi in zip(phi_pvecs, self.phis, ):
            energy += phi.calc_energy(y)

        # Embedding terms
        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)

        tmp_eng, ni_sorted = self.embedding_energy(ni, u_pvecs, u_ranges)
        energy += tmp_eng

        return energy, ni_sorted

    def compute_ni(self, rho_pvecs, f_pvecs, g_pvecs):
        """
        Computes ni values for all atoms

        Args:
            rho_pvecs: parameter vectors for rho splines
            f_pvecs: parameter vectors for f splines
            g_pvecs: parameter vectors for g splines

        Returns:
            ni: embedding values for each potential for each atom
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

        self.all_ni = ni

        return ni

    def embedding_energy(self, ni, u_pvecs, new_range):
        """
        Computes embedding energy

        Args:
            ni: per-atom ni values
            u_pvecs: parameter vectors for U splines

        Returns:
            u_energy: total embedding energy
            ni_sorted: maximum magnitude ni values for each atom *type*
        """

        u_energy = np.zeros(self.n_pots)

        max_ni = np.zeros((self.n_pots, len(u_pvecs)))

        # Evaluate U, U'
        for i, (y, u) in enumerate(zip(u_pvecs, self.us)):
            u.structure_vectors['energy'] = np.zeros((self.n_pots, u.knots.shape[0]+2))

            # extract ni values for atoms of type i
            ni_sublist = ni[:, self.type_of_each_atom - 1 == i]

            max_ni[:, i] = np.abs(np.max(ni_sublist))

            num_embedded = ni_sublist.shape[1]

            if num_embedded > 0:
                u_range = new_range[i]

                u.add_to_energy_struct_vec(
                    ni_sublist, u_range[0], u_range[1]
                    )

                u_energy += u.calc_energy(y)

            u.reset()

        return u_energy, max_ni

    def evaluate_uprimes(self, ni, u_pvecs, u_ranges, second=False):
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
            new_range = u_ranges[i]

            # get atom ids of type i
            indices = tags[shifted_types == i]

            u.structure_vectors['deriv'] = np.zeros(
                (self.n_pots, self.natoms, u.knots.shape[0]+2))

            if second:
                u.structure_vectors['2nd_deriv'] = np.zeros(
                    (self.n_pots, self.natoms, u.knots.shape[0]+2))

            if indices.shape[0] > 0:
                u.add_to_deriv_struct_vec(
                    ni[:, shifted_types == i], indices, new_range[0],
                    new_range[1]
                )

                if second:
                    u.add_to_2nd_deriv_struct_vec(
                        ni[:, shifted_types == i], indices, new_range[0],
                        new_range[1]
                        )

        # Evaluate U, U', and compute zero-point energies
        uprimes = np.zeros((self.n_pots, self.natoms))

        if second: uprimes_2 = np.zeros((self.n_pots, self.natoms))

        for y, u in zip(u_pvecs, self.us):
            uprimes += u.calc_deriv(y)

            if second: uprimes_2 += u.calc_2nd_deriv(y)

        if second: return uprimes, uprimes_2
        else: return uprimes

    def compute_forces(self, parameters, u_ranges):
        """Calculates the force vectors on each atom using the given spline
        parameters.

        Args:
            parameters (np.arr): the 1D array of concatenated parameter
                                 vectors for all splines in the system
            u_ranges (list): list of tuples of (lhs_knot, rhs_knot)
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
        uprimes = self.evaluate_uprimes(ni, u_pvecs, u_ranges)

        # Electron density embedding (rho)

        embedding_forces = np.zeros((self.n_pots, 3*self.natoms*self.natoms))

        for rho_idx, (rho, y) in enumerate(zip(self.rhos, rho_pvecs)):
            embedding_forces += rho.calc_forces(y)

        # Angular terms (ffg)
        for j, ffg_list in enumerate(self.ffgs):
            for k, ffg in enumerate(ffg_list):

                y_fj = f_pvecs[j]
                y_fk = f_pvecs[k]
                y_g = g_pvecs[src.meam.ij_to_potl(j+1, k+1, self.ntypes)]

                embedding_forces += ffg.calc_forces(y_fj, y_fk, y_g)

        N = self.natoms

        embedding_forces = embedding_forces.reshape((self.n_pots, 3, N, N))
        embedding_forces = np.einsum('pijk,pk->pji', embedding_forces, uprimes)

        return forces + embedding_forces

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

    def compute_grad_indices(self, ffgs):
        """Prepares lists of indices for extracting partial derivatives of the
        outer product of f_j*f_k*g
        """

        # TODO: this doubles the memory footprint, but improves performance

        ffg_indices = []

        for ffg_list in ffgs:
            tmp_list = []

            for ffg in ffg_list:
                n_fj = len(ffg.fj.knots) + 2
                n_fk = len(ffg.fk.knots) + 2
                n_g = len(ffg.g.knots) + 2

                fj_indices = np.zeros((n_fj, n_fk*n_g))
                for l in range(n_fj):
                    fj_indices[l] = np.arange(n_fk*n_g) + l*n_fk*n_g

                tmp_indices = np.arange(n_g)

                fk_indices = np.zeros((n_fk, n_fj*n_g))
                for l in range(n_fk):
                    k_indices = np.arange(l*n_g, n_fj*n_fk*n_g, n_fk*n_g)

                    gen = itertools.product(k_indices, tmp_indices)
                    fk_indices[l] = [e1+e2 for e1,e2 in gen]

                g_indices = np.zeros((n_g, n_fj*n_fk))
                for l in range(n_g):
                    g_indices[l] = np.arange(l, n_fk*n_fj*n_g, n_g)

                fj_indices = fj_indices.astype(int)
                fk_indices = fk_indices.astype(int)
                g_indices = g_indices.astype(int)

                tmp_list.append({'fj_indices':fj_indices,
                                 'fk_indices':fk_indices,
                                 'g_indices':g_indices})

            ffg_indices.append(tmp_list)

        return ffg_indices

    # @profile
    def energy_gradient_wrt_pvec(self, pvec, u_ranges):
        parameters = np.atleast_2d(pvec)
        gradient = np.zeros(parameters.shape)

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(np.atleast_2d(parameters))

        self.n_pots = parameters.shape[0]

        grad_index = 0
        # gradients of phi are just their structure vectors
        for y, phi in zip(phi_pvecs, self.phis):
            gradient[:, grad_index:grad_index + y.shape[1]] += \
                phi.structure_vectors['energy']

            grad_index += y.shape[1]

        # chain rule on U functions means dU/dn values are needed
        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(ni, u_pvecs, u_ranges)

        for y, rho in zip(rho_pvecs, self.rhos):

            partial_ni = rho.structure_vectors['energy']

            gradient[:, grad_index:grad_index + y.shape[1]] += \
                (uprimes @ partial_ni)

            grad_index += y.shape[1]

        # add in first term of chain rule
        for i,(y,u) in enumerate(zip(u_pvecs, self.us)):
            new_range = u_ranges[i]

            ni_sublist = ni[:, self.type_of_each_atom - 1 == i]

            u.update_knot_positions(
                new_range[0], new_range[1], y.shape[0]
                )

            num_embedded = ni_sublist.shape[1]

            if num_embedded > 0:
                u.add_to_energy_struct_vec(
                    ni_sublist, new_range[0], new_range[1]
                    )

                gradient[:, grad_index:grad_index + y.shape[1]] += \
                    u.structure_vectors['energy']

            grad_index += y.shape[1]

            u.reset()

        ffg_indices = self.build_ffg_grad_index_list(grad_index, f_pvecs,
                                                     g_pvecs)

        # add in second term of chain rule
        for j, (y_fj, ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            n_fj = y_fj.shape[1]

            for k, (y_fk, ffg) in enumerate(zip(f_pvecs, ffg_list)):
                g_idx = src.meam.ij_to_potl(j+1, k+1, self.ntypes)

                y_g = g_pvecs[g_idx]

                n_fk = y_fk.shape[1]
                n_g = y_g.shape[1]

                scaled_sv = ffg.structure_vectors['energy']
                scaled_sv = np.einsum('pz,zk->pk', uprimes, scaled_sv)

                coeffs_for_fj = np.einsum("pi,pk->pik", y_fk, y_g)
                coeffs_for_fk = np.einsum("pi,pk->pik", y_fj, y_g)
                coeffs_for_g = np.einsum("pi,pk->pik", y_fj, y_fk)

                coeffs_for_fj = coeffs_for_fj.reshape(
                    (self.n_pots, y_fk.shape[1] * y_g.shape[1])
                )

                coeffs_for_fk = coeffs_for_fk.reshape(
                    (self.n_pots, y_fj.shape[1] * y_g.shape[1])
                )

                coeffs_for_g = coeffs_for_g.reshape(
                    (self.n_pots, y_fj.shape[1] * y_fk.shape[1])
                )

                # every ffgSpline affects grad(f_j), grad(f_k), and grad(g)

                # pre-computed indices for outer product indexing
                indices_tuple = self.ffg_grad_indices[j][k]

                stack = np.zeros((self.n_pots, n_fj, n_fk*n_g))

                # grad(f_j) contribution
                for l in range(n_fj):
                    sample_indices = indices_tuple['fj_indices'][l]

                    stack[:, l, :] = scaled_sv[:, sample_indices]

                # stack = stack @ coeffs_for_fj
                stack = np.einsum('pzk,pk->pz', stack, coeffs_for_fj)
                gradient[:, ffg_indices[j]:ffg_indices[j] + n_fj] += stack

                stack = np.zeros((self.n_pots, n_fk, n_fj*n_g))

                # grad(f_k) contribution
                for l in range(n_fk):
                    sample_indices = indices_tuple['fk_indices'][l]

                    stack[:, l, :] = scaled_sv[:, sample_indices]

                stack = np.einsum('pzk,pk->pz', stack, coeffs_for_fk)

                gradient[:, ffg_indices[k]:ffg_indices[k] + n_fk] += stack

                stack = np.zeros((self.n_pots, n_g, n_fj*n_fk))

                # grad(g) contribution
                for l in range(n_g):
                    sample_indices = indices_tuple['g_indices'][l]

                    stack[:, l, :] = scaled_sv[:, sample_indices]

                stack = np.einsum('pzk,pk->pz', stack, coeffs_for_g)

                tmp_idx = ffg_indices[self.ntypes + g_idx]
                gradient[:, tmp_idx:tmp_idx + n_g] += stack

        return gradient

    # @profile
    def forces_gradient_wrt_pvec(self, pvec, u_ranges, sparse=False):
        parameters = np.atleast_2d(pvec)
        self.n_pots = parameters.shape[0]

        if sparse:
            raise NotImplementedError("Sparsity for gradients needs to be done")
        else:
            gradient = np.zeros((self.n_pots, self.natoms, 3, self.len_param_vec))

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(np.atleast_2d(parameters))

        grad_index = 0

        # gradients of phi are just their structure vectors
        for y, phi in zip(phi_pvecs, self.phis):
            gradient[:, :, :, grad_index:grad_index + y.shape[1]] += \
                phi.structure_vectors['forces'].reshape((self.natoms, 3, y.shape[1]))

            grad_index += y.shape[1]

        # chain rule on U functions means dU/dn values are needed
        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)

        N = self.natoms

        uprimes, uprimes_2 = self.evaluate_uprimes(
            ni, u_pvecs, u_ranges, second=True
            )

        embedding_forces = np.zeros((self.n_pots, 3, self.natoms, self.natoms))

        # pre-compute all rho forces
        for rho_idx, (y_rho,rho) in enumerate(zip(rho_pvecs, self.rhos)):
            rho_forces = rho.calc_forces(y_rho)
            rho_forces = rho_forces.reshape((self.n_pots, 3, self.natoms,
                                             self.natoms))

            embedding_forces += rho_forces

        # pre-compute all ffg forces
        for j, (y_fj,ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            for k, (y_fk,ffg) in enumerate(zip(f_pvecs, ffg_list)):
                y_g = g_pvecs[src.meam.ij_to_potl(j+1, k+1, self.ntypes)]

                ffg_forces = ffg.calc_forces(y_fj, y_fk, y_g)
                ffg_forces = ffg_forces.reshape((self.n_pots, 3, self.natoms,
                                                 self.natoms))

                embedding_forces += ffg_forces

        # rho gradient term; there is a U'' and a U' term for each rho
        for rho_index, (y, rho) in enumerate(zip(rho_pvecs, self.rhos)):

            rho_e_sv = rho.structure_vectors['energy']

            rho_sv = rho.structure_vectors['forces']
            rho_sv = rho_sv.reshape((3, self.natoms, self.natoms,
                                     y.shape[1]))

            # U'' term
            uprimes_scaled = np.einsum('pi,ij->pij', uprimes_2, rho_e_sv)

            stacking_results = np.zeros((self.n_pots, N, 3, y.shape[1]))

            stacking_results += np.einsum('pij,pkli->plkj', uprimes_scaled,
                                          embedding_forces)

            gradient[:, :, :, grad_index:grad_index + y.shape[1]] += \
                stacking_results

            # U' term
            up_contracted_sv = np.einsum('ijkl,pk->pjil', rho_sv, uprimes)

            gradient[:, :,:,grad_index:grad_index + y.shape[1]] += \
                up_contracted_sv

            grad_index += y.shape[1]

        # save indices so that embedding_forces can be added later
        tmp_U_indices = []

        # prep for U gradient term
        for i,(y,u) in enumerate(zip(u_pvecs, self.us)):
            tmp_U_indices.append((grad_index, grad_index + y.shape[1]))
            grad_index += y.shape[1]

        # TODO: this should occur in __init__
        ffg_indices = self.build_ffg_grad_index_list(grad_index, f_pvecs,
                                                     g_pvecs)

        # ffg gradient terms
        for j, (y_fj, ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            n_fj = y_fj.shape[1]

            for k, (y_fk, ffg) in enumerate(zip(f_pvecs, ffg_list)):
                g_idx = src.meam.ij_to_potl(j+1, k+1, self.ntypes)

                y_g = g_pvecs[g_idx]

                n_fk = y_fk.shape[1]
                n_g = y_g.shape[1]

                full_len = n_fj*n_fk*n_g

                ffg_sv = ffg.structure_vectors['forces']
                ffg_sv = ffg_sv.reshape((3, N, N, full_len))

                ffg_e_sv = ffg.structure_vectors['energy']

                # TODO: this gets huge if evaluating lots of pots on many atoms
                upp_contrib = np.zeros((self.n_pots, N, 3, full_len))

                scaled_by_upp = np.einsum('pz,zk->pzk', uprimes_2, ffg_e_sv)

                # U'' term
                upp_contrib += np.einsum('pzk,paiz->piak', scaled_by_upp,
                                         embedding_forces)

                # U' term
                scaled_sv = ffg_sv
                up_contrib = np.einsum('pz,aizk->paik', uprimes, scaled_sv)

                up_contrib = np.transpose(up_contrib, axes=(0, 2, 1, 3))

                # Group terms and add to gradient

                coeffs_for_fj = np.einsum("pi,pk->pik", y_fk, y_g)
                coeffs_for_fk = np.einsum("pi,pk->pik", y_fj, y_g)
                coeffs_for_g = np.einsum("pi,pk->pik", y_fj, y_fk)

                coeffs_for_fj = coeffs_for_fj.reshape(
                    (self.n_pots, y_fk.shape[1] * y_g.shape[1])
                )

                coeffs_for_fk = coeffs_for_fk.reshape(
                    (self.n_pots, y_fj.shape[1] * y_g.shape[1])
                )

                coeffs_for_g = coeffs_for_g.reshape(
                    (self.n_pots, y_fj.shape[1] * y_fk.shape[1])
                )

                # pre-computed indices for outer product indexing
                indices_tuple = self.ffg_grad_indices[j][k]

                stack_up = np.zeros(
                    (self.n_pots, self.natoms, 3, n_fj, n_fk*n_g)
                )

                stack_upp = np.zeros(
                    (self.n_pots, self.natoms, 3, n_fj, n_fk*n_g)
                )

                for l in range(n_fj):
                    sample_indices = indices_tuple['fj_indices'][l]

                    stack_up[:, :, :, l, :] = up_contrib[:, :, :,sample_indices]
                    stack_upp[:, :, :, l, :] = upp_contrib[:,:,:,sample_indices]

                stack_up = np.einsum('pzakt,pt->pzak', stack_up, coeffs_for_fj)
                stack_upp = np.einsum('pzakt,pt->pzak', stack_upp,coeffs_for_fj)

                tmp_ind = ffg_indices[j]
                gradient[:, :, :, tmp_ind:tmp_ind + n_fj] += stack_up
                gradient[:, :, :, tmp_ind:tmp_ind + n_fj] += stack_upp

                stack_up = np.zeros(
                    (self.n_pots, self.natoms, 3, n_fk, n_fj*n_g)
                )

                stack_upp = np.zeros(
                    (self.n_pots, self.natoms, 3, n_fk, n_fj*n_g)
                )

                for l in range(n_fk):
                    sample_indices = indices_tuple['fk_indices'][l]

                    stack_up[:, :, :, l, :] = up_contrib[:, :, :,
                                              sample_indices]
                    stack_upp[:, :, :, l, :] = upp_contrib[:, :, :,
                                               sample_indices]

                stack_up = np.einsum('pzakt,pt->pzak', stack_up, coeffs_for_fk)
                stack_upp = np.einsum('pzakt,pt->pzak', stack_upp,coeffs_for_fk)

                tmp_ind = ffg_indices[k]
                gradient[:, :, :, tmp_ind:tmp_ind + n_fk] += stack_up
                gradient[:, :, :, tmp_ind:tmp_ind + n_fk] += stack_upp

                stack_up = np.zeros(
                    (self.n_pots, self.natoms, 3, n_g, n_fj*n_fk)
                )

                stack_upp = np.zeros(
                    (self.n_pots, self.natoms, 3, n_g, n_fj*n_fk)
                )

                for l in range(n_g):
                    sample_indices = indices_tuple['g_indices'][l]

                    stack_up[:, :, :, l, :] = up_contrib[:,:,:,sample_indices]
                    stack_upp[:, :, :, l, :] = upp_contrib[:,:,:,sample_indices]

                stack_up = np.einsum('pzakt,pt->pzak', stack_up, coeffs_for_g)
                stack_upp = np.einsum('pzakt,pt->pzak', stack_upp,coeffs_for_g)

                tmp_ind = ffg_indices[self.ntypes + g_idx]
                gradient[:, :, :, tmp_ind:tmp_ind + n_g] += stack_up
                gradient[:, :, :, tmp_ind:tmp_ind + n_g] += stack_upp

        # U gradient terms

        for i, (indices,u) in enumerate(zip(tmp_U_indices, self.us)):
            tmp_U_indices.append((grad_index, grad_index + y.shape[1]))
            start, stop = indices

            u_term = np.einsum('zk,paiz->piak', u.structure_vectors['deriv'][0],
                        embedding_forces)

            gradient[:, :, :, start:stop] += u_term

        return gradient

    def build_ffg_grad_index_list(self, grad_index, f_pvecs, g_pvecs):
        """A helper function to simplify indexing the ffg parts of the gradient"""

        tmp_index = grad_index
        ffg_indices = [grad_index]

        for y_fj in f_pvecs:
            ffg_indices.append(tmp_index + y_fj.shape[1])
            tmp_index += y_fj.shape[1]

        for y_g in g_pvecs:
            ffg_indices.append(tmp_index + y_g.shape[1])
            tmp_index += y_g.shape[1]

        return ffg_indices

# TODO: replace with JIT outer_prod_simple
def my_outer_prod(y_fj, y_fk, y_g):
    cart1 = np.einsum("ij,ik->ijk", y_fj, y_fk)
    cart1 = cart1.reshape((cart1.shape[0], cart1.shape[1]*cart1.shape[2]))

    cart2 = np.einsum("ij,ik->ijk", cart1, y_g)

    cart_y = cart2.reshape((cart2.shape[0], cart2.shape[1]*cart2.shape[2]))

    return cart_y.ravel()

# @profile
def main():
    np.random.seed(42)
    import src.meam
    from tests.testPotentials import get_random_pots
    from tests.testStructs import allstructs

    pot = get_random_pots(1)['meams'][0]

    x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    def fd_gradient_eval(y_pvec, worker, type, h=1e-8):
        N = y_pvec.ravel().shape[0]

        cd_points = np.array([y_pvec] * N*2)

        for l in range(N):
            cd_points[2*l, l] += h
            cd_points[2*l+1, l] -= h

        if type == 'energy':
            cd_evaluated = worker.compute_energy(np.array(cd_points))
            fd_gradient = np.zeros(N)
        elif type == 'forces':
            cd_evaluated = worker.compute_forces(np.array(cd_points))
            fd_gradient = np.zeros((worker.natoms, 3, worker.len_param_vec))

        for l in range(N):
            if type == 'energy': fd_gradient[l] = \
                (cd_evaluated[2*l] - cd_evaluated[2*l+1]) / h / 2

            elif type == 'forces': fd_gradient[:, :, l] = \
                (cd_evaluated[2*l] - cd_evaluated[2*l+1]) / h / 2

        return fd_gradient

    import pickle
    import os

    # test_name = 'bulk_vac_rhombo_type1'
    # test_name = '8_atoms'
    # allstructs = {test_name:allstructs[test_name]}

    with open("grad_accuracy_normed_final.dat", 'w') as accuracy_outfile:
        with open("grad_time_normed_final.dat", 'w') as time_outfile:

            accuracy_outfile.write("name e_diff f_diff\n")
            time_outfile.write("name e_speedup e_w_time e_fd_time f_speedup f_w_time f_fd_time\n")
            for name,atoms in allstructs.items():
                print(name, end="")
                print()

                # if os.path.isfile('data/workers/' + name + '.pkl'):
                if False:
                    print(" -- Loading file", end="")
                    worker = pickle.load(open('data/workers/' + name + '.pkl', 'rb'))
                    print()
                else:
                    worker = Worker(atoms, x_pvec, indices, pot.types)
                    # pickle.dump(worker, open('data/workers/' + name + '.pkl', 'wb'))

                final_e_max = 0
                final_f_max = 0

                for phi in worker.phis:
                    e_max = np.max(phi.structure_vectors['energy'])
                    f_max = np.max(phi.structure_vectors['forces'])

                    if e_max > final_e_max: final_e_max = e_max
                    if f_max > final_f_max: final_f_max = f_max

                for rho in worker.rhos:
                    e_max = np.max(rho.structure_vectors['energy'])
                    f_max = np.max(rho.structure_vectors['forces'])

                    if e_max > final_e_max: final_e_max = e_max
                    if f_max > final_f_max: final_f_max = f_max

                for f in worker.fs:
                    e_max = np.max(f.structure_vectors['energy'])
                    f_max = np.max(f.structure_vectors['forces'])

                    if e_max > final_e_max: final_e_max = e_max
                    if f_max > final_f_max: final_f_max = f_max

                for g in worker.gs:
                    e_max = np.max(g.structure_vectors['energy'])
                    f_max = np.max(g.structure_vectors['forces'])

                    if e_max > final_e_max: final_e_max = e_max
                    if f_max > final_f_max: final_f_max = f_max

                for i in range(500):
                    start_fd_e = time.time()
                    fd_grad_e = fd_gradient_eval(y_pvec, worker, 'energy')
                    fd_e_time = time.time() - start_fd_e

                    start_fd_f = time.time()
                    fd_grad_f = fd_gradient_eval(y_pvec, worker, 'forces')
                    fd_f_time = time.time() - start_fd_f

                    start_w_e = time.time()
                    w_grad_e = worker.energy_gradient_wrt_pvec(y_pvec)
                    w_e_time = time.time() - start_w_e

                    worker.compute_energy(y_pvec)

                    start_w_f = time.time()
                    w_grad_f = worker.forces_gradient_wrt_pvec(y_pvec)
                    w_f_time = time.time() - start_w_f

                    worker.compute_forces(y_pvec)

                    diff_e = np.max(np.abs(fd_grad_e - w_grad_e))
                    diff_f = np.max(np.abs(fd_grad_f - w_grad_f))

                    for u in worker.us:
                        e_max = np.max(u.structure_vectors['energy'])

                        if e_max > final_e_max: final_e_max = e_max

                    e_speedup = fd_e_time / w_e_time
                    f_speedup = fd_f_time / w_f_time

                    accuracy_outfile.write(name+ ' ' +str(diff_e/final_e_max) +
                                           ' ' + str(diff_f/final_f_max) + "\n")

                    time_outfile.write(name + ' ' + str(e_speedup) + ' ' +
                                       str(w_e_time) + ' ' + str(fd_e_time) + ' ' +
                                       str(f_speedup) + ' ' + str(w_f_time) + ' ' +
                                       str(fd_f_time) + "\n")

                if (diff_e > 1e-5) or (diff_f > 1e-5):
                    print('\tDANGER, THIS IS LARGE ERROR')

                print(diff_e / final_e_max)
                print(diff_f / final_f_max)

if __name__ == "__main__":
    main()
