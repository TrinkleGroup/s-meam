import numpy as np
import logging

from ase.neighborlist import NeighborList

import src.lammpsTools
import src.meam
from src.workerSplines import WorkerSpline, RhoSpline, ffgSpline
# from src.numba_functions import jit_add_at_1D, jit_add_at_2D

# import pyximport; pyximport.install()
# from src.cython_functions import cython_add_at_2D, cython_add_at_1D

# import workerfunctions

logger = logging.getLogger(__name__)


class Worker:
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials

    Assumptions:
        - knot x-coordinates never change
        - the atomic structure never changes
    """

    # TODO: an assumption is made that all potentials have the same cutoffs

    # @profile
    def __init__(self, atoms, knot_xcoords, x_indices, types):
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

                    where phi_Ti-Ti has 5 knots, phi_Ti-O has 9, rho_O has 13,
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
        """

        # Basic variable initialization
        self.atoms      = atoms
        self.types      = types

        ntypes          = len(self.types)
        self.ntypes     = ntypes
        self.natoms     = len(atoms)

        self.type_of_each_atom = []
        for atom in atoms:
            self.type_of_each_atom.append(src.lammpsTools.symbol_to_type(
                atom.symbol, self.types))

        # there are nphi phi functions and nphi g fxns
        nphi            = int((self.ntypes+1)*self.ntypes/2)
        self.nphi       = nphi

        self.uprimes    = np.zeros(len(atoms))

        self.phis, self.rhos, self.us, self.fs, self.gs = \
            self.build_spline_lists(knot_xcoords, x_indices)

        self.phis = list(self.phis)
        self.rhos = list(self.rhos)
        self.us = list(self.us)
        self.fs = list(self.fs)
        self.gs = list(self.gs)

        self.ffgs = self.build_ffg_list(self.fs, self.gs)

        # Compute full potential cutoff distance (based only on radial fxns)
        radial_fxns = self.phis + self.rhos + self.fs
        self.cutoff = np.max([max(s.x) for s in radial_fxns])

        # Directional information for force calculations; grouped by spline
        # e.g. phi_directions = [phi_0 directions, phi_1 directions, ...]
        phi_directions = [[[] for j in range(self.natoms)]
                               for i in range(len(self.phis))]
        rho_directions = [[[] for j in range(self.natoms)]
                               for i in range(len(self.rhos))]
        ffg_directions = [[[[] for k in range(self.natoms)]
                           for j in range(len(self.fs))]
                               for i in range(len(self.fs))]
        # self.ffg_directions = [[[[] for k in range(6)] for j in range(len(
        #     self.fs))] for i in range(len(self.fs))]

        # Temporary collectors for spline values to optimize spline get_abcd()
        # 3D lists; indexed by (spline number, atom number, value)
        energy_phi_rij = [[[] for j in range(self.natoms)]
                       for i in range(len(self.phis))]
        forces_phi_rij = [[[] for j in range(self.natoms)]
                          for i in range(len(self.phis))]

        energy_rho_rij = [[[] for j in range(self.natoms)]
                       for i in range(len(self.rhos))]
        forces_rho_rij = [[[] for j in range(self.natoms)]
                          for i in range(len(self.rhos))]

        num_f = len(self.fs)
        # 4D list; (fj index, fk index, atom number, value)
        energy_ffg_rij = [[[[] for k in range(self.natoms)]
                           for i in range(num_f)] for j in range(num_f)]
        energy_ffg_rik = [[[[] for k in range(self.natoms)]
                           for i in range(num_f)] for j in range(num_f)]
        energy_ffg_cos = [[[[] for k in range(self.natoms)]
                           for i in range(num_f)] for j in range(num_f)]

        forces_ffg_rij = [[[[] for k in range(self.natoms)]
                           for i in range(num_f)] for j in range(num_f)]
        forces_ffg_rik = [[[[] for k in range(self.natoms)]
                           for i in range(num_f)] for j in range(num_f)]
        forces_ffg_cos = [[[[] for k in range(self.natoms)]
                           for i in range(num_f)] for j in range(num_f)]

        # Tags to track which atom each spline eval corresponds to
        phi_rij_indices = [[[] for j in range(self.natoms)]
                           for i in range(len(self.phis))]
        self.rho_rij_indices = [[[] for j in range(self.natoms)]
                           for i in range(len(self.rhos))]
        self.ffg_indices= [[[[] for k in range(self.natoms)]
                            for i in range(num_f)] for j in range(num_f)]

        # Build neighbor lists

        # No double counting of bonds; needed for pair interactions
        nl_noboth = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2.),
                                 self_interaction=False, bothways=False,
                                 skin=0.0)
        nl_noboth.build(atoms)

        # Allows double counting bonds; needed for embedding energy calculations
        nl = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2.),
                          self_interaction=False, bothways=True, skin=0.0)
        nl.build(atoms)

        # for i in range(self.natoms):
        for i, atom in enumerate(self.atoms):
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

                energy_phi_rij[phi_idx][i].append(rij)

                forces_phi_rij[phi_idx][i].append(rij)
                forces_phi_rij[phi_idx][j].append(rij)

                phi_rij_indices[phi_idx][i].append(j)

                phi_directions[phi_idx][i].append(jvec)
                phi_directions[phi_idx][j].append(-jvec)

            # Store distances, angle, and index info for embedding terms
            j_counter = 0  # for tracking neighbor
            for j, offsetj in zip(neighbors, offsets):

                jtype = self.type_of_each_atom[j]

                jpos = atoms[j].position + np.dot(offsetj, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0]**2 + jvec[1]**2 + jvec[2]**2)
                jvec /= rij

                rho_idx = src.meam.i_to_potl(jtype)

                energy_rho_rij[rho_idx][i].append(rij)

                forces_rho_rij[rho_idx][i].append(rij)
                forces_rho_rij[rho_idx][j].append(rij)

                self.rho_rij_indices[rho_idx][j].append(i)
                self.rho_rij_indices[rho_idx][i].append(i)

                rho_directions[rho_idx][i].append(jvec)
                rho_directions[rho_idx][j].append(-jvec)

                # prepare for angular calculations
                a = jpos - ipos
                na = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

                fj_idx = src.meam.i_to_potl(jtype)

                j_counter += 1
                for k, offsetk in zip(neighbors[j_counter:],
                                      offsets[j_counter:]):
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

                        # fk information
                        fk_idx = src.meam.i_to_potl(ktype)

                        energy_ffg_rij[fj_idx][fk_idx][i].append(rij)
                        energy_ffg_rik[fj_idx][fk_idx][i].append(rik)
                        energy_ffg_cos[fj_idx][fk_idx][i].append(cos_theta)

                        forces_ffg_rij[fj_idx][fk_idx][i].append(rij)
                        forces_ffg_rik[fj_idx][fk_idx][i].append(rik)
                        forces_ffg_cos[fj_idx][fk_idx][i].append(cos_theta)

                        self.ffg_indices[fj_idx][fk_idx][i] += [i]*6
                        # self.ffg_indices[fj_idx][fk_idx][j] += [i]*6

                        # Directions added to match ordering of terms in
                        # first derivative of fj*fk*g
                        d0 = jvec
                        d1 = -cos_theta * jvec# / rij
                        d2 = kvec# / rij
                        d3 = kvec
                        d4 = -cos_theta * kvec# / rik
                        d5 = jvec# / rik

                        ffg_directions[fj_idx][fk_idx][i] += [d0, d1, d2, d3,
                                                              d4, d5]

        # Add full groups of spline evaluations to respective splines
        for phi_idx, (phi, phi_dirs) in enumerate(zip(self.phis,
                                                      phi_directions)):

            max_num_evals_eng = max([len(el) for el in energy_phi_rij[phi_idx]])
            max_num_evals_fcs = max([len(el) for el in forces_phi_rij[phi_idx]])

            if max_num_evals_eng > 0: phi.max_num_evals_eng = max_num_evals_eng
            if max_num_evals_fcs > 0: phi.max_num_evals_fcs = max_num_evals_fcs

            phi.energy_struct_vec = np.zeros((self.natoms, 2*len(phi.x)+4))
            phi.forces_struct_vec = np.zeros((self.natoms, 2*len(phi.x)+4,
                                              max_num_evals_fcs, 3))

            for i, (energy_vals, dirs, force_vals) in enumerate(zip(
                    energy_phi_rij[phi_idx], phi_dirs,forces_phi_rij[phi_idx])):

                phi.add_to_energy_struct_vec(energy_vals, i)

                if dirs:
                    phi.add_to_forces_struct_vec(force_vals, dirs, i)

        for rho_idx, rho in enumerate(self.rhos):

            max_num_evals_eng = max([len(el) for el in energy_rho_rij[rho_idx]])
            max_num_evals_fcs = max([len(el) for el in forces_rho_rij[rho_idx]])

            indices = np.zeros((self.natoms, max_num_evals_fcs)) - 1

            for i, row in enumerate(self.rho_rij_indices[rho_idx]):
                indices[i, :len(row)] = row

            self.rho_rij_indices[rho_idx] = indices

            if max_num_evals_eng > 0: rho.max_num_evals_eng = max_num_evals_eng
            if max_num_evals_fcs > 0: rho.max_num_evals_fcs = max_num_evals_fcs

            rho.energy_struct_vec = np.zeros((self.natoms, 2*len(rho.x)+4))
            rho.forces_struct_vec = np.zeros((self.natoms, 2*len(rho.x)+4,
                                              max_num_evals_fcs, 3))

            for i, (energy_vals, dirs, force_vals) in enumerate(zip(
                    energy_rho_rij[rho_idx], rho_directions[rho_idx],
                                                    forces_rho_rij[rho_idx])):

                rho.add_to_energy_struct_vec(energy_vals, i)

                if dirs:
                    rho.add_to_forces_struct_vec(force_vals, dirs, i)

        for fj_idx in range(len(self.ffgs)):
            for fk_idx in range(len(self.ffgs)):
                ffg = self.ffgs[fj_idx][fk_idx]

                max_num_evals_eng = max([len(el) for el in energy_ffg_rij[
                    fj_idx][fk_idx]])
                max_num_evals_fcs = max([len(el) for el in forces_ffg_rij[
                    fj_idx][fk_idx]])

                if max_num_evals_eng > 0: ffg.max_num_evals_eng = max_num_evals_eng
                if max_num_evals_fcs > 0: ffg.max_num_evals_fcs = max_num_evals_fcs

                indices = np.zeros((self.natoms, 6*max_num_evals_fcs)) - 1

                for i, row in enumerate(self.ffg_indices[fj_idx][fk_idx]):
                    indices[i, :len(row)] = row

                self.ffg_indices[fj_idx][fk_idx] = indices

                energy_struct_vec = np.zeros((self.natoms, 2*len(rho.x)+4,
                                                  max_num_evals_eng))

                ffg.fj_energy_struct_vec = energy_struct_vec.copy()
                ffg.fk_energy_struct_vec = energy_struct_vec.copy()
                ffg.g_energy_struct_vec = energy_struct_vec.copy()

                energy_rij = energy_ffg_rij[fj_idx][fk_idx]
                energy_rik = energy_ffg_rik[fj_idx][fk_idx]
                energy_cos = energy_ffg_cos[fj_idx][fk_idx]

                # for i in range(len(energy_rij)):
                for i, (rij, rik, cos) in enumerate(zip(energy_rij, energy_rik,
                                                    energy_cos)):

                    ffg.add_to_energy_struct_vec(rij, rik, cos, i)

                forces_struct_vec = np.zeros((self.natoms, 2*len(rho.x)+4,
                                                  6*max_num_evals_fcs, 3))

                ffg.fj_forces_struct_vec = forces_struct_vec.copy()
                ffg.fk_forces_struct_vec = forces_struct_vec.copy()
                ffg.g_forces_struct_vec = forces_struct_vec.copy()

                forces_rij = forces_ffg_rij[fj_idx][fk_idx]
                forces_rik = forces_ffg_rik[fj_idx][fk_idx]
                forces_cos = forces_ffg_cos[fj_idx][fk_idx]

                # TODO: self.*_indices -> *.indices; have spline store

                for i, (rij, rik, cos, dirs) in\
                    enumerate(zip(forces_rij, forces_rik, forces_cos,
                                  ffg_directions[fj_idx][fk_idx])):

                    if dirs:
                        ffg.add_to_forces_struct_vec(rij, rik, cos, dirs, i)

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
        bc_type = ('fixed', 'fixed')

        splines = []

        # for i in range(self.ntypes * (self.ntypes + 4)):
        for i, knots in enumerate(knots_split):
            if (i < self.nphi)\
                or (self.nphi + self.ntypes <= i < self.nphi + 2 *self.ntypes):

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

        ffg_list = [[] for i in range(len(self.fs))]
        for j, fj in enumerate(fs):
            for k, fk in enumerate(fs):
                g = gs[src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                ffg_list[j].append(ffgSpline(fj, fk, g, self.natoms))

        return ffg_list

    def compute_energies(self, parameters):
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

        # TODO: should __call__() take in just a single potential?
        # TODO: Worker has list of Pots; each Pot is a list of WorkerSplines

        # Uprimes must be reset to avoid reusing old results
        self.uprimes[:] = 0.

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        energy = 0.

        # Pair interactions
        for y, phi in zip(phi_pvecs, self.phis, ):
            # if phi.struct_vecs[0] != []:
            if len(phi.energy_struct_vec) > 0:
                energy += np.sum(phi.calc_energy(y))

        # Embedding terms
        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        energy += self.embedding_energy(ni, u_pvecs)

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
        ni = np.zeros(self.natoms)

        # Rho contribution
        for y, rho in zip(rho_pvecs, self.rhos):
            ni += rho.calc_energy(y)

        # Three-body contribution
        for j, (y_fj,ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            for k, (y_fk,ffg) in enumerate(zip(f_pvecs, ffg_list)):

                y_g = g_pvecs[src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                ni += ffg.calc_energy(y_fj, y_fk, y_g)

        return ni

    def embedding_energy(self, ni, u_pvecs):
        """
        Computes embedding energy

        Args:
            ni: per-atom ni values
            u_pvecs: parameter vectors for U splines

        Returns:
            u_energy: total embedding energy
        """

        sorted_ni = [[] for i in range(len(self.us))]
        ni_indices = [[] for i in range(len(self.us))]

        # Add ni values to respective u splines
        # for i, atom in enumerate(self.atoms):
        for i in range(self.natoms):
            # itype = src.lammpsTools.symbol_to_type(atom.symbol, self.types)
            itype = self.type_of_each_atom[i]
            u_idx = itype - 1

            sorted_ni[u_idx].append(ni[i])
            ni_indices[u_idx].append([i,i])

        for u, ni, indices in zip(self.us, sorted_ni, ni_indices):
            u.add_to_struct_vec(ni, indices)

        # Evaluate U, U', and compute zero-point energies
        u_energy = 0
        for y, u in zip(u_pvecs, self.us):

            # zero-point has to be calculated separately bc has to be subtracted
            if len(u.struct_vecs[0]) > 0:
                u_energy -= np.sum(u.compute_zero_potential(y))
                u_energy += np.sum(u(y))

        return u_energy

    def evaluate_uprimes(self, ni, u_pvecs):
        """
        Computes U' values for every atom

        Args:
            ni: per-atom ni values
            u_pvecs: parameter vectors for U splines

        Returns:
            uprimes: per-atom U' values
        """

        sorted_ni = [[] for i in range(len(self.us))]
        ni_indices = [[] for i in range(len(self.us))]

        # Add ni values to respective u splines
        for i, itype in enumerate(self.type_of_each_atom):
            u_idx = src.meam.i_to_potl(itype)

            sorted_ni[u_idx].append(ni[i])
            ni_indices[u_idx].append([i,i])

        for u, ni, indices in zip(self.us, sorted_ni, ni_indices):
            u.struct_vecs = [[], []]
            u.indices_f = []
            u.indices_b = []
            u.add_to_struct_vec(ni, indices)

        # Evaluate U, U', and compute zero-point energies
        uprimes = np.zeros(self.natoms)
        for y, u in zip(u_pvecs, self.us):

            if len(u.struct_vecs[0]) > 0:
                np.add.at(uprimes, u.indices_f, u(y, 1))
                # uprimes += np.bincount(u.indices_f, weights=u(y, 1),
                #                        minlength=len(uprimes))
                # cython_add_at_1D(uprimes, np.array(u.indices_f,
                #                                    dtype=np.int32), u(y, 1))
                # uprimes += jit_add_at_1D(u.indices_f, u(y, 1), self.natoms)

        return uprimes

    # @profile
    def compute_forces(self, parameters):
        """Calculates the force vectors on each atom using the given spline
        parameters.

        Args:
            parameters (np.arr): the 1D array of concatenated parameter
                vectors for all splines in the system
        """

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        forces = np.zeros((len(self.atoms), 3))

        # Pair forces (phi)
        for phi_idx, (phi, y) in enumerate(zip(self.phis, phi_pvecs)):

            if len(phi.forces_struct_vec) > 0:
                # Evaluate derivatives and multiply by direction vectors
                forces += np.sum(phi.calc_forces(y), axis=1)

        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(ni, u_pvecs)
        uprimes = np.append(uprimes, 0.)

        # Electron density embedding (rho)
        for rho_idx, (rho, y) in enumerate(zip(self.rhos, rho_pvecs)):

            if len(rho.forces_struct_vec) > 0:
                rho_forces = rho.calc_forces(y)

                uprime_scales = np.take(uprimes,\
                                    self.rho_rij_indices[rho_idx].astype(int))

                forces += np.einsum('ijk,ij->ik', rho_forces, uprime_scales)

        # Angular terms (ffg)
        for j in range(len(self.ffgs)):
            ffg_list = self.ffgs[j]

            for k in range(len(ffg_list)):
                ffg = ffg_list[k]

                # if len(ffg.indices_f[0]) > 0:
                if len(ffg.fj_forces_struct_vec) > 0:

                    y_fj = f_pvecs[j]
                    y_fk = f_pvecs[k]
                    y_g = g_pvecs[src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                    uprime_scales = np.take(uprimes,
                                            self.ffg_indices[j][k].astype(int))

                    # logging.info("WORKER: u' = {0}".format(uprime_scales))

                    ffg_forces = ffg.calc_forces(y_fj, y_fk, y_g)
                    forces += np.einsum('ijk,ij->ik', ffg_forces, uprime_scales)

                    # logging.info("WORKER: {0}".format(np.einsum('ijk,'\
                    #                     'ij->ijk',ffg_forces, uprime_scales)))
                    # logging.info("WORKER: {0}".format(ffg_forces[:,:,1]))

                    # ffg_forces = np.einsum('ij,i->ij', ffg_dirs,
                    #                        ffg(y_fj, y_fk, y_g, 1))

                    # ffg_forces = np.einsum('ij,i->ij', ffg_forces,
                    #                        uprimes[ffg.indices_f[1]])

                    # f0 = lambda a: np.bincount(ffg.indices_f[1], weights=a,
                    #                            minlength=self.natoms)
                    # f1 = lambda a: np.bincount(ffg.indices_b[1], weights=a,
                    #                            minlength=self.natoms)

                    # forces += np.apply_along_axis(f0, 0, ffg_forces)
                    # forces -= np.apply_along_axis(f1, 0, ffg_forces)

                    # forces += jit_add_at_2D(ffg.indices_f[1], ffg_forces,
                    #                         self.natoms)
                    # forces -= jit_add_at_2D(ffg.indices_b[1], ffg_forces,
                    #                         self.natoms)

        return forces

    def parse_parameters(self, parameters):
        """Separates the pre-ordered 1D vector of all spline parameters into
        groups.

        Args:
            parameters (np.arr):
                1D array of knot points and boundary conditions for ALL
                splines for ALL intervals

        Returns:
            *_pvecs (np.arr):
                each return is a list of arrays of parameters. e.g.
                phi_pvecs[0] is the parameters for the first phi spline
        """

        splines = self.phis + self.rhos + self.us + self.fs + self.gs

        # Parse parameter vector
        x_indices = [s.index for s in splines]
        y_indices = [x_indices[i] + 2 * i for i in range(len(x_indices))]

        params_split = np.split(parameters, y_indices[1:])

        nphi = self.nphi
        ntypes = self.ntypes

        split_indices = [nphi, nphi + ntypes, nphi + 2 * ntypes,
                         nphi + 3 * ntypes]
        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = np.split(
            params_split, split_indices)

        return phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs

if __name__ == "__main__":
    pass
