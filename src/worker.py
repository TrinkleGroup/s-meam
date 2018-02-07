import numpy as np
import lammpsTools
import meam
import logging

from ase.neighborlist import NeighborList
from workerSplines import WorkerSpline, RhoSpline, ffgSpline

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

        nphi            = int((self.ntypes+1)*self.ntypes/2)
        self.nphi       = nphi # there are nphi phi functions and nphi g fxns

        self.uprimes    = np.zeros(len(atoms))

        # Initialize splines; group by type and calculate potential cutoff range

        # TODO: separate out __init__ into functions for readability

        knots_split = np.split(knot_xcoords, x_indices[1:])

        splines = []

        for i in range(ntypes * (ntypes + 4)):
            idx = x_indices[i]

            # TODO: could specify bc outside of Worker & pass in
            # # check if phi/rho or f
            # if (i < nphi+ntypes) or ((i >= nphi+2*ntypes) and
            #                          (i < nphi+3*ntypes)):
            #     bc_type = ('natural','fixed')
            # else:
            #     bc_type = ('natural','natural')

            # for comparing against TiO.meam.spline; all are 'fixed'
            bc_type = ('fixed', 'fixed')

            if (i < nphi) or ((i >= nphi + ntypes) and (i < nphi + 2 * ntypes)):
                # phi or U
                s = WorkerSpline(knots_split[i], bc_type)
            else:
                s = RhoSpline(knots_split[i], bc_type, len(self.atoms))

            s.index = idx

            splines.append(s)

        split_indices = np.array([nphi, nphi + ntypes, nphi + 2 * ntypes,
                                  nphi + 3 * ntypes])
        self.phis, self.rhos, self.us, self.fs, self.gs =\
            np.split(np.array(splines), np.array(split_indices))

        self.phis = list(self.phis)
        self.rhos = list(self.rhos)
        self.us = list(self.us)
        self.fs = list(self.fs)
        self.gs = list(self.gs)

        self.ffgs = []

        # Build all combinations of ffg splines
        for j in range(len(self.fs)):
            inner_list = []

            for k in range(len(self.fs)):
                fj = self.fs[j]
                fk = self.fs[k]
                g = self.gs[meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                inner_list.append(ffgSpline(fj, fk, g, len(self.atoms)))
            self.ffgs.append(inner_list)

        # Compute full potential cutoff distance (based only on radial fxns)
        radial_fxns = self.phis + self.rhos + self.fs
        self.cutoff = np.max([max(s.x) for s in radial_fxns])

        # Building neighbor lists
        natoms = len(atoms)

        # No double counting; needed for pair interactions
        nl_noboth = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2.),
                                 self_interaction=False, bothways=False,
                                 skin=0.0)
        nl_noboth.build(atoms)

        # Directions grouped by spline group AND by spline type
        # e.g. phi_directions = [phi_0 directions, phi_1 directions, ...]
        # Will be sorted into per-atom groups in evaluation

        # TODO: huge redundancy in directional information
        self.phi_directions = [[] for i in range(len(self.phis))]
        self.rho_directions = [[] for i in range(len(self.rhos))]
        self.ffg_directions = [[[] for j in range(len(self.fs))] for i in range(
            len(self.fs))]

        # Allows double counting; needed for embedding energy calculations
        nl = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2.),
                          self_interaction=False, bothways=True, skin=0.0)
        nl.build(atoms)

        for i in range(natoms):
            # Record atom type info
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # Stores pair information for phi
            for j, offset in zip(neighbors_noboth, offsets_noboth):
                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset, atoms.get_cell())

                jvec = jpos - ipos

                rij = np.linalg.norm(jvec)
                jvec /= rij

                phi_idx = meam.ij_to_potl(itype, jtype, self.ntypes)

                self.phis[phi_idx].add_to_struct_vec(rij, [i,j])

                self.phi_directions[phi_idx].append(jvec)

            # Store distances, angle, and index info for embedding terms
            j_counter = 0  # for tracking neighbor
            for j, offsetj in zip(neighbors, offsets):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offsetj, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.linalg.norm(jvec)
                jvec /= rij

                rho_idx = meam.i_to_potl(jtype)

                self.rhos[rho_idx].add_to_struct_vec(rij, [i, j])

                self.rho_directions[rho_idx].append(jvec)

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

                        self.ffgs[fj_idx][fk_idx].add_to_struct_vec(rij, rik,
                                                                    cos_theta,
                                                                    [i, j, k])

                        # Directions added to match ordering of terms in
                        # first derivative of fj*fk*g
                        d0 = jvec
                        d1 = -cos_theta * jvec / rij
                        d2 = kvec / rij
                        d3 = kvec
                        d4 = -cos_theta * kvec / rik
                        d5 = jvec / rik

                        self.ffg_directions[fj_idx][fk_idx] += [d0, d1, d2,
                                                                d3, d4, d5]

        # self.phi_directions = np.array(self.phi_directions)
        # self.rho_directions = np.array(self.rho_directions)
        # self.ffg_directions = np.array(self.ffg_directions)
        self.phi_directions = [np.array(el) for el in self.phi_directions]
        self.rho_directions = [np.array(el) for el in self.rho_directions]
        self.ffg_directions = [[np.array(el) for el in l] for l in
                               self.ffg_directions]

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

        # TODO: ***** Ensure that everything in this function MUST be here *****

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        energy = 0.

        # Pair interactions
        for i in range(len(self.phis)):
            y = phi_pvecs[i]
            s = self.phis[i]

            energy += np.sum(s(y))

        # Calculate rho contributions to ni
        ni = np.zeros(len(self.atoms))

        for i in range(len(self.rhos)):
            rho = self.rhos[i]
            y = rho_pvecs[i]

            ni += rho.compute_for_all(y)

        # Calculate three-body contributions to ni
        for j in range(len(self.ffgs)):
            ffg_list = self.ffgs[j]

            y_fj = f_pvecs[j]

            for k in range(len(ffg_list)):
                ffg = ffg_list[k]

                y_fk = f_pvecs[k]
                y_g = g_pvecs[meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                val = ffg.compute_for_all(y_fj, y_fk, y_g)
                ni += val

        # TODO: vectorize this
        # TODO: build a zero_struct here to avoid iterating over each atom twice

        # Add ni values to respective u splines
        for i in range(len(self.atoms)):
            itype = lammpsTools.symbol_to_type(self.atoms[i].symbol, self.types)
            u_idx = meam.i_to_potl(itype)

            u = self.us[u_idx]

            # TODO: it's dumb that you have to pass in [i,i] for U b/c inherits
            u.add_to_struct_vec(ni[i], [i, i])

        # Evaluate u splines and zero-point energies
        zero_point_energy = 0
        for i in range(len(self.us)):
            u = self.us[i]
            y = u_pvecs[i]

            # zero-point has to be calculated separately bc has to be SUBTRACTED
            # off of the energy
            if u.struct_vecs != [[], []]:
                tmp_struct = u.struct_vecs
                tmp_indices = u.indices

                u.struct_vecs = [[], []]
                u.indices = []
                u.add_to_struct_vec(np.zeros(len(tmp_struct[0])), [0, 0])

                zero_point_energy += np.sum(u(y))

                u.struct_vecs = tmp_struct
                u.indices = tmp_indices

                energy += np.sum(u(y))

                np.add.at(self.uprimes, np.array(u.indices)[:, 0], u(y, 1))

        return energy - zero_point_energy

    def compute_forces(self, parameters):
        """Calculates the force vectors on each atom using the given spline
        parameters.

        Args:
            parameters (np.arr):
                the 1D array of concatenated parameter vectors for all
                splines in the system
        """

        # Compute system energy; needed for U' values
        self.compute_energies(parameters)

        # Parse vectors into groups of splines; init variables
        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        self.forces = np.zeros((len(self.atoms), 3))

        # Pair forces (phi)
        for phi_idx in range(len(self.phis)):
            # Extract spline and corresponding parameter vector
            s = self.phis[phi_idx]

            phi_dirs = self.phi_directions[phi_idx]

            if len(phi_dirs) > 0:
                y = phi_pvecs[phi_idx]

                # Evaluate derivatives and multiply by direction vectors
                phi_primes = s(y, 1)
                phi_forces = np.einsum('ij,i->ij', phi_dirs, phi_primes)

                # Pull atom ID info and update forces in both directions
                phi_indices = np.array(s.indices)

                self.update_forces(phi_forces, phi_indices)

        # Electron density embedding (rho)
        for rho_idx in range(len(self.rhos)):
            s = self.rhos[rho_idx]

            rho_dirs = self.rho_directions[rho_idx]

            if len(rho_dirs) > 0:
                y = rho_pvecs[rho_idx]
                rho_primes = s(y, 1)
                rho_forces = np.einsum('ij,i->ij', rho_dirs, rho_primes)

                rho_indices = np.array(s.indices)
                rho_forces = np.einsum('ij,i->ij', rho_forces, self.uprimes[
                    rho_indices[:, 0]])

                self.update_forces(rho_forces, rho_indices)

        # Angular terms (ffg)
        for j in range(len(self.ffgs)):
            ffg_list = self.ffgs[j]

            for k in range(len(ffg_list)):
                ffg = ffg_list[k]

                ffg_dirs = self.ffg_directions[j][k]

                if len(ffg_dirs) > 0:
                    y_fj = f_pvecs[j]
                    y_fk = f_pvecs[k]
                    y_g = g_pvecs[meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                    ffg_primes = ffg(y_fj, y_fk, y_g, 1)
                    ffg_forces = np.einsum('ij,i->ij', ffg_dirs, ffg_primes)

                    ffg_indices = np.array(ffg.indices[1])
                    ffg_forces = np.einsum('ij,i->ij', ffg_forces,
                                           self.uprimes[ffg_indices[:, 0]])

                    self.update_forces(ffg_forces, ffg_indices)

                    # rzm: forces are incorrect after vstack -> append?

        return self.forces

    def update_forces(self, new_forces, indices):
        """Updates the system's per atom forces.

        Args:
            new_forces (np.arr):
                Nx3 array of force vectors to add to the system ordered to
                match indices
            indices (np.arr):
                Nx2 array of indices where the first column is the atom ID
                of the first atom, and the second column is the neighbor's ID

        Returns:
            None; manually updates self.forces
        """

        # TODO: get rid of this function

        np.add.at(self.forces, indices[:, 0], new_forces)
        np.add.at(self.forces, indices[:, 1], -new_forces)

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
