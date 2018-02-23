import numpy as np
import lammpsTools
import meam
import logging

from ase.neighborlist import NeighborList
from workerSplines import WorkerSpline, RhoSpline, ffgSpline

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
        self.phi_directions = [[] for i in range(len(self.phis))]
        self.rho_directions = [[] for i in range(len(self.rhos))]
        self.ffg_directions = [[[[] for k in range(6)] for j in range(len(
            self.fs))] for i in range(len(self.fs))]

        # Temporary collectors for spline values to optimize spline get_abcd()
        all_phi_rij = [[] for i in range(len(self.phis))]
        all_rho_rij = [[] for i in range(len(self.rhos))]

        num_f = len(self.fs)
        all_ffg_rij = [[[] for i in range(num_f)] for j in range(num_f)]
        all_ffg_rik = [[[] for i in range(num_f)] for j in range(num_f)]
        all_ffg_cos = [[[] for i in range(num_f)] for j in range(num_f)]

        # Tags to track which atom each spline eval corresponds to
        phi_rij_indices = [[] for i in range(len(self.phis))]
        rho_rij_indices = [[] for i in range(len(self.rhos))]
        ffg_indices= [[[] for i in range(num_f)] for j in range(num_f)]

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
            itype = lammpsTools.symbol_to_type(atom.symbol, self.types)
            ipos = atom.position

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # Stores pair information for phi
            for j, offset in zip(neighbors_noboth, offsets_noboth):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)

                # Find displacement vector (with periodic boundary conditions)
                jpos = atoms[j].position + np.dot(offset, atoms.get_cell())
                jvec = jpos - ipos
                rij = np.linalg.norm(jvec)

                # Add distance/index/direction information to necessary lists
                phi_idx = meam.ij_to_potl(itype, jtype, self.ntypes)

                all_phi_rij[phi_idx].append(rij)
                phi_rij_indices[phi_idx].append([i,j])

                self.phi_directions[phi_idx].append(jvec / rij)

            # Store distances, angle, and index info for embedding terms
            j_counter = 0  # for tracking neighbor
            for j, offsetj in zip(neighbors, offsets):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offsetj, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.linalg.norm(jvec)
                jvec /= rij

                rho_idx = meam.i_to_potl(jtype)

                all_rho_rij[rho_idx].append(rij)
                rho_rij_indices[rho_idx].append([i,j])

                self.rho_directions[rho_idx].append(jvec)

                # prepare for angular calculations
                a = jpos - ipos
                na = np.linalg.norm(a)

                fj_idx = meam.i_to_potl(jtype)

                j_counter += 1
                for k, offsetk in zip(neighbors[j_counter:],
                                      offsets[j_counter:]):
                    if k != j:
                        ktype = lammpsTools.symbol_to_type(atoms[k].symbol,
                                                           self.types)
                        kpos = atoms[k].position + np.dot(offsetk,
                                                          atoms.get_cell())

                        kvec = kpos - ipos
                        rik = np.linalg.norm(kvec)
                        kvec /= rik

                        b = kpos - ipos
                        nb = np.linalg.norm(b)

                        cos_theta = np.dot(a, b) / na / nb

                        # fk information
                        fk_idx = meam.i_to_potl(ktype)

                        all_ffg_rij[fj_idx][fk_idx].append(rij)
                        all_ffg_rik[fj_idx][fk_idx].append(rik)
                        all_ffg_cos[fj_idx][fk_idx].append(cos_theta)

                        ffg_indices[fj_idx][fk_idx].append([i,j,k])

                        # Directions added to match ordering of terms in
                        # first derivative of fj*fk*g
                        d0 = jvec
                        d1 = -cos_theta * jvec / rij
                        d2 = kvec / rij
                        d3 = kvec
                        d4 = -cos_theta * kvec / rik
                        d5 = jvec / rik

                        self.ffg_directions[fj_idx][fk_idx][0].append(d0)
                        self.ffg_directions[fj_idx][fk_idx][1].append(d1)
                        self.ffg_directions[fj_idx][fk_idx][2].append(d2)
                        self.ffg_directions[fj_idx][fk_idx][3].append(d3)
                        self.ffg_directions[fj_idx][fk_idx][4].append(d4)
                        self.ffg_directions[fj_idx][fk_idx][5].append(d5)

        # Add full groups of spline evaluations to respective splines
        for phi_idx in range(len(self.phis)):
            self.phis[phi_idx].add_to_struct_vec(all_phi_rij[phi_idx],
                                                 phi_rij_indices[phi_idx])

        for rho_idx in range(len(self.rhos)):
            self.rhos[rho_idx].add_to_struct_vec(all_rho_rij[rho_idx],
                                                 rho_rij_indices[rho_idx])

        for fj_idx in range(len(self.fs)):
            for fk_idx in range(len(self.fs)):
                rij = all_ffg_rij[fj_idx][fk_idx]
                rik = all_ffg_rik[fj_idx][fk_idx]
                cos_theta = all_ffg_cos[fj_idx][fk_idx]
                indices = ffg_indices[fj_idx][fk_idx]

                self.ffgs[fj_idx][fk_idx].add_to_struct_vec(rij, rik,
                                                            cos_theta,
                                                            indices)

        # Convert directional information to arrays for future computations
        self.phi_directions = [np.array(el) for el in self.phi_directions]
        self.rho_directions = [np.array(el) for el in self.rho_directions]

        for j,fj_dirs in enumerate(self.ffg_directions):
            for k,fk_dirs in enumerate(fj_dirs):
                for p,dir in enumerate(fk_dirs):
                    self.ffg_directions[j][k][p] = np.array(dir)

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

                s = WorkerSpline(knots, bc_type)
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
                g = gs[meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

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
        self.uprimes = np.zeros(self.uprimes.shape)

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        energy = 0.

        # Pair interactions
        for y, phi in zip(phi_pvecs, self.phis):
            energy += np.sum(phi(y))

        # Embedding terms
        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        energy += self.embedding_energy(ni, u_pvecs)

        return energy

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
            ni += rho.compute_for_all(y)

        # Three-body contribution
        for j, (y_fj,ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            for k, (y_fk,ffg) in enumerate(zip(f_pvecs, ffg_list)):

                y_g = g_pvecs[meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                ni += ffg.compute_for_all(y_fj, y_fk, y_g)

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
        for i, atom in enumerate(self.atoms):
            itype = lammpsTools.symbol_to_type(atom.symbol, self.types)
            u_idx = meam.i_to_potl(itype)

            sorted_ni[u_idx].append(ni[i])
            ni_indices[u_idx].append([i,i])

        for u, ni, indices in zip(self.us, sorted_ni, ni_indices):
            u.struct_vecs = [[], []]
            u.indices = []
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
        for i, atom in enumerate(self.atoms):
            itype = lammpsTools.symbol_to_type(atom.symbol, self.types)
            u_idx = meam.i_to_potl(itype)

            sorted_ni[u_idx].append(ni[i])
            ni_indices[u_idx].append([i,i])

        for u, ni, indices in zip(self.us, sorted_ni, ni_indices):
            u.struct_vecs = [[], []]
            u.indices = []
            u.add_to_struct_vec(ni, indices)

        # Evaluate U, U', and compute zero-point energies
        uprimes = np.zeros(self.natoms)
        for y, u in zip(u_pvecs, self.us):

            if len(u.struct_vecs[0]) > 0:
                np.add.at(uprimes, [el[0] for el in u.indices], u(y, 1))

        return uprimes

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
        # forces = [np.zeros(3) for i in range(len(self.atoms))]

        # Pair forces (phi)
        for phi_idx, (phi, phi_dirs, y) in\
            enumerate(zip(self.phis, self.phi_directions, phi_pvecs)):

            if len(phi_dirs) > 0:
                # Evaluate derivatives and multiply by direction vectors
                phi_forces = np.einsum('ij,i->ij', phi_dirs, phi(y, 1))

                # Pull atom ID info and update forces in both directions
                indices_0, indices_1 = zip(*phi.indices)
                indices_0 = list(indices_0)
                indices_1 = list(indices_1)

                f0 = lambda a: np.bincount(indices_0, weights=a,
                                           minlength=self.natoms)
                f1 = lambda a: np.bincount(indices_1, weights=a,
                                           minlength=self.natoms)

                forces += np.apply_along_axis(f0, 0, phi_forces)
                forces -= np.apply_along_axis(f1, 0, phi_forces)

        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(ni, u_pvecs)

        # Electron density embedding (rho)
        for rho_idx, (rho, rho_dirs, y) in\
            enumerate(zip(self.rhos, self.rho_directions, rho_pvecs)):

            if len(rho_dirs) > 0:
                rho_forces = np.einsum('ij,i->ij', rho_dirs, rho(y, 1))

                indices_0, indices_1 = zip(*rho.indices)
                indices_0 = list(indices_0)
                indices_1 = list(indices_1)

                rho_forces = np.einsum('ij,i->ij', rho_forces,
                                       uprimes[indices_0])

                f0 = lambda a: np.bincount(indices_0, weights=a,
                                           minlength=self.natoms)
                f1 = lambda a: np.bincount(indices_1, weights=a,
                                           minlength=self.natoms)

                forces += np.apply_along_axis(f0, 0, rho_forces)
                forces -= np.apply_along_axis(f1, 0, rho_forces)

        # Angular terms (ffg)
        for j in range(len(self.ffgs)):
            ffg_list = self.ffgs[j]

            for k in range(len(ffg_list)):
                ffg = ffg_list[k]

                ffg_dirs = self.ffg_directions[j][k]
                ffg_dirs = np.vstack(ffg_dirs)

                if len(ffg.indices[1]) > 0:
                    indices_0, indices_1 = zip(*ffg.indices[1])
                    indices_0 = list(indices_0)
                    indices_1 = list(indices_1)

                    y_fj = f_pvecs[j]
                    y_fk = f_pvecs[k]
                    y_g = g_pvecs[meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                    ffg_forces = np.einsum('ij,i->ij', ffg_dirs,
                                           ffg(y_fj, y_fk, y_g, 1))

                    ffg_forces = np.einsum('ij,i->ij', ffg_forces,
                                           uprimes[indices_0])

                    f0 = lambda a: np.bincount(indices_0, weights=a,
                                               minlength=self.natoms)
                    f1 = lambda a: np.bincount(indices_1, weights=a,
                                               minlength=self.natoms)

                    forces += np.apply_along_axis(f0, 0, ffg_forces)
                    forces -= np.apply_along_axis(f1, 0, ffg_forces)

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
