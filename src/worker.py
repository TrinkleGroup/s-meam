import numpy as np
import lammpsTools
import meam
import logging
import matplotlib.pyplot as plt

from ase.neighborlist import NeighborList
from scipy.sparse import diags

from spline import Spline
from workerSplines import WorkerSpline, RhoSpline, ffgSpline

logger = logging.getLogger(__name__)
# logging.basicConfig(filename='worker.log')

class Worker:
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials

    Assumptions:
        - knot x-coordinates never change
        - the atomic structure never changes
        - boundary conditions always passed in as pairs of floats (typically
          d0,dN), then 'natural' or 'fixed' decided later"""

    # TODO: an assumption is made that all potentials have the same cutoffs
    # TODO: check speeds of list append() vs. numpy vstack/concatenate/append

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
                have every atom type in it."""

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

        for i in range(ntypes*(ntypes+4)):
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

            if (i < nphi) or ((i >= nphi+ntypes) and (i < nphi+2*ntypes)):
                # phi or U
                s = WorkerSpline(knots_split[i], bc_type)
            else:
                s = RhoSpline(knots_split[i], bc_type, len(self.atoms))

            s.index = idx

            splines.append(s)

        split_indices = np.array([nphi, nphi+ntypes, nphi+2*ntypes,
                                 nphi+3*ntypes])
        self.phis, self.rhos, self.us, self.fs, self.gs = np.split(splines,
                                                               split_indices)

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
                g = self.gs[meam.ij_to_potl(j+1, k+1, self.ntypes)]

                inner_list.append(ffgSpline(fj, fk, g, len(self.atoms)))
            self.ffgs.append(inner_list)

        # Compute full potential cutoff distance (based only on radial fxns)
        radial_fxns = self.phis + self.rhos + self.fs
        self.cutoff = np.max([max(s.x) for s in radial_fxns])

        # Building neighbor lists
        natoms = len(atoms)

        # No double counting; needed for pair interactions
        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=False, skin=0.0)
        nl_noboth.build(atoms)

        # Directions grouped by spling group AND by spline type
        # e.g. phi_directions = [phi_0 directions, phi_1 directions, ...]
        # Will be sorted into per-atom groups in evaluation
        # Splines track their own indices
        self.phi_directions = [np.empty((0,3)) for i in range(len(self.phis))]
        self.rho_directions = [np.empty((0,3)) for i in range(len(self.rhos))]
        self.ffg_directions = [[np.empty((0,3)) for i in range(len(self.fs))]
                               for j in range(len(self.fs))]

        # Allows double counting; needed for embedding energy calculations
        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=True, skin=0.0)
        nl.build(atoms)

        for i in range(natoms):
            # Record atom type info
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            idx_counter = 0
            # Stores pair information for phi
            for j,offset in zip(neighbors_noboth, offsets_noboth):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset, atoms.get_cell())

                jvec = jpos - ipos

                rij = np.linalg.norm(jvec)
                jvec /= rij

                phi_idx = meam.ij_to_potl(itype, jtype, self.ntypes)

                self.phis[phi_idx].add_to_struct_vec(rij, [i,j])

                # TODO: don't use vstack to add directions
                self.phi_directions[phi_idx] = np.vstack((self.phi_directions[
                                                             phi_idx], jvec))
                # self.phis[phi_idx].indices.append([i,j])
                # jvec *= rij
                # logging.info("WORKER: {0}, {1}, {2}, {3}, {4}".format(i,j,
                #                                       jvec[0], jvec[1], jvec[2]))

            # Store distances, angle, and index info for embedding terms
            j_counter = 0 # for tracking neighbor
            for j,offset in zip(neighbors,offsets):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                jvec = jpos - ipos
                rij = np.linalg.norm(jvec)
                jvec /= rij

                rho_idx = meam.i_to_potl(jtype)

                # self.rhos[rho_idx].update_struct_vec_dict(rij, i)
                self.rhos[rho_idx].add_to_struct_vec(rij, [i,j])

                self.rho_directions[rho_idx] = np.vstack((self.rho_directions[
                                                             rho_idx], jvec))
                # self.rhos[rho_idx].indices.append([i,j])

                a = jpos - ipos
                na = np.linalg.norm(a)

                fj_idx = meam.i_to_potl(jtype)

                j_counter += 1
                for k,offset in zip(neighbors[j_counter:], offsets[j_counter:]):
                    if k != j:
                        ktype = lammpsTools.symbol_to_type(atoms[k].symbol,
                                                           self.types)
                        kpos = atoms[k].position + np.dot(offset,
                                                          atoms.get_cell())

                        kvec = kpos - ipos
                        rik = np.linalg.norm(kvec)
                        kvec /= rik

                        b = kpos - ipos
                        nb = np.linalg.norm(b)

                        cos_theta = np.dot(a,b)/na/nb

                        # fk information
                        fk_idx = meam.i_to_potl(ktype)

                        self.ffgs[fj_idx][fk_idx].add_to_struct_vec(rij, rik,
                                                     cos_theta, [i,j,k])

                        # Directions added to match ordering of terms in
                        # first derivative of fj*fk*g
                        d0 = jvec; d1 = -cos_theta*jvec/rij; d2 = kvec/rij
                        d3 = kvec; d4 = -cos_theta*kvec/rik; d5 = jvec/rik

                        # logging.info("WORKER: jdel = {0}".format(jvec))
                        # logging.info("WORKER: jdel*cos = {0}".format(d1))
                        # logging.info("WORKER: kdel = {0}".format(kvec))
                        # logging.info("WORKER: kdel*cos = {0}".format(d4))

                        self.ffg_directions[fj_idx][fk_idx] = np.vstack((
                            self.ffg_directions[fj_idx][fk_idx],
                            d0, d1, d2, d3, d4, d5))


    def compute_energies(self, parameters):
        """Calculates energies for all potentials using information
        pre-computed during initialization.

        Args:
            parameters (np.arr):
                2D list of all parameters for all splines; each row
                corresponds to a unique potential. Each group in a
                single row should have K+2 elements where K is the number
                of knot points, and the 2 additional are boundary conditions.
                The first K in each group are the knot y-values"""

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

            # val = np.sum(rho.compute_for_all(y), axis=0)
            ni += rho.compute_for_all(y)


        # Calculate three-body contributions to ni
        for j in range(len(self.ffgs)):
            ffg_list = self.ffgs[j]

            y_fj = f_pvecs[j]

            for k in range(len(ffg_list)):
                ffg = ffg_list[k]

                y_fk = f_pvecs[k]
                y_g = g_pvecs[meam.ij_to_potl(j+1, k+1, self.ntypes)]

                val = ffg.compute_for_all(y_fj, y_fk, y_g)
                ni += val

        # logging.info("WORKER: ni = {0}".format(ni))

        # TODO: vectorize this
        # TODO: build a zero_struct here to avoid iterating over each atom twice
        # Add ni values to respective u splines
        for i in range(len(self.atoms)):
            itype = lammpsTools.symbol_to_type(self.atoms[i].symbol, self.types)
            u_idx = meam.i_to_potl(itype)

            u = self.us[u_idx]

            # self.us[u_idx].indices.append(i)
            # self.us[u_idx].indices = np.append(self.us[u_idx].indices, i)

            # TODO: clear_struct_vec(); sets struct_vec = []

            # TODO: it's dumb that you have to pass in [i,i] for U b/c inherits
            u.add_to_struct_vec(ni[i], [i,i])

        # Evaluate u splines and zero-point energies
        zero_point_energy = 0
        for i in range(len(self.us)):
            u = self.us[i]
            y = u_pvecs[i]

            # zero-point has to be calculated separately bc has to be SUBTRACTED
            # off of the energy
            if u.struct_vecs != [[],[]]:
                tmp_struct  = u.struct_vecs
                tmp_indices = u.indices

                u.struct_vecs = [[],[]]
                u.indices = []
                u.add_to_struct_vec(np.zeros(len(tmp_struct[0])), [0,0])

                zero_point_energy += np.sum(u(y))

                u.struct_vecs   = tmp_struct
                u.indices       = tmp_indices

                energy += np.sum(u(y))
                # logging.info("WORKER: U'({0}) = {1}".format(ni, u(y,1)))
                # self.uprimes += u(y, 1)
                np.add.at(self.uprimes, np.array(u.indices)[:,0], u(y, 1))

        # logging.info("WORKER: ni = {0}".format(ni))
        # logging.info("WORKER: U' = {0}".format(self.uprimes))

        return energy - zero_point_energy

    def compute_forces(self, parameters):
        """Calculates the force vectors on each atom using the given spline
        parameters.

        Args:
            parameters (np.arr):
                the 1D array of concatenated parameter vectors for all
                splines in the system"""

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

                joined = np.concatenate((phi_indices, phi_dirs), axis=1)

                # logging.info("WORKER: {0}".format(joined))
                #
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
                                       rho_indices[:,0]])

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
                    y_g = g_pvecs[meam.ij_to_potl(j+1, k+1, self.ntypes)]

                    ffg_primes = ffg(y_fj, y_fk, y_g, 1)
                    ffg_forces = np.einsum('ij,i->ij', ffg_dirs, ffg_primes)

                    ffg_indices = np.array(ffg.indices[1])
                    ffg_forces = np.einsum('ij,i->ij', ffg_forces,
                                           self.uprimes[ffg_indices[:,0]])
                    # logging.info("WORKER: ffg_forces = {0}".format(ffg_forces))

                    self.update_forces(ffg_forces, ffg_indices)

                    # logging.info("WORKER: ffg_dirs = {0}".format(ffg_dirs))
                    # logging.info("WORKER: ffg = {0}".format(ffg_primes))
                    # logging.info("WORKER: indices = {0}".format(ffg_indices))

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
            None; manually updates self.forces"""

        # # Parse unsorted atom ID's
        # forwards_indices = indices[:,0]
        # backwards_indices = indices[:,1]
        #
        # # Get ordered set of atom tags
        # indices_to_sort_tags_f = forwards_indices.argsort()
        # sorted_indices_f = set(forwards_indices[indices_to_sort_tags_f])
        #
        # indices_to_sort_tags_b = backwards_indices.argsort()
        # sorted_indices_b = set(backwards_indices[indices_to_sort_tags_b])
        #
        # # Sort forces for forwards/backwards
        # forwards_forces = new_forces[indices_to_sort_tags_f]
        # backwards_forces = new_forces[indices_to_sort_tags_b]
        #
        # # Split into groups per atom tag
        # forwards_forces = np.split(forwards_forces, np.where(np.diff(
        #     forwards_indices[indices_to_sort_tags_f]))[0]+1)
        #
        # backwards_forces = np.split(backwards_forces, np.where(np.diff(
        #     backwards_indices[indices_to_sort_tags_b]))[0]+1)
        #
        # # Sum per atom forces and add to total
        # forwards_forces = np.array([np.sum(el, axis=0) for el in
        #                             forwards_forces])
        # backwards_forces = np.array([np.sum(el, axis=0) for el in
        #                              backwards_forces])
        #
        # np.add.at(self.forces, np.array(list(sorted_indices_f)),
        #           forwards_forces)
        # np.add.at(self.forces, np.array(list(sorted_indices_b)),
        #           -backwards_forces)

        # joined = np.concatenate((indices, new_forces), axis=1)

        np.add.at(self.forces, indices[:,0], new_forces)
        np.add.at(self.forces, indices[:,1], -new_forces)

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
                phi_pvecs[0] is the parameters for the first phi spline"""

        splines = self.phis + self.rhos + self.us + self.fs + self.gs

        # Parse parameter vector
        x_indices = [s.index for s in splines]
        y_indices = [x_indices[i]+2*i for i in range(len(x_indices))]

        params_split = np.split(parameters, y_indices[1:])

        nphi = self.nphi
        ntypes = self.ntypes

        split_indices = [nphi, nphi+ntypes, nphi+2*ntypes, nphi+3*ntypes]
        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = np.split(
            params_split, split_indices)

        return phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs

    #     """Calculates energies for all potentials using information
    #     pre-computed during initialization."""
    #     #TODO: only thing changing should be y-coords?
    #     # changing y-coords changes splines, changes coeffs
    #     # maybe could use cubic spline eval equation, not polyval()
    #
    #     # TODO: should __call__() take in just a single potential?
    #
    #     # TODO: is it worth creating a huge matrix of ordered coeffs so that
    #     # it's just a single matrix multiplication? probably yes because
    #     # Python for-loops are terrible! problem: need to zero pad to make
    #     # same size
    #
    #     # TODO: turn coeffs into array of knot y-values
    #
    #     # TODO: eventually move away from scipy Spline, towards Recipes equation
    #
    #     # TODO: might as well just merge with compute_energies()?
    #     self.compute_energies(potentials)
    #
    #     natoms = len(self.atoms)
    #     npots = len(potentials)
    #
    #     # Derivative matrix for cubic coefficients
    #     D = np.array([[0,1,0,0], [0,0,2,0], [0,0,0,3]])
    #
    #     self.potentials = potentials
    #     forces = np.zeros((npots, natoms, 3))
    #
    #     # Compute phi contribution to total energy
    #     phi_coeffs = coeffs_from_indices(self.phis, self.phi_index_info)
    #
    #     i_indices = self.pair_indices_oneway[:,0]
    #     j_indices = self.pair_indices_oneway[:,1]
    #
    #     j_directions = self.pair_directions_oneway
    #
    #     phi_prime_coeffs = np.einsum('ij,jdk->idk', D, phi_coeffs)
    #     phi_prime_values = eval_all_polynomials(self.pair_distances_oneway,
    #                             phi_prime_coeffs)
    #     phi_forces = np.einsum('ij,jk->ijk',phi_prime_values,j_directions)
    #
    #     # See np.ufunc.at documentation for why this is necessary
    #     np.add.at(forces, (slice(None), i_indices, slice(None)), phi_forces)
    #     np.add.at(forces, (slice(None), j_indices, slice(None)), -phi_forces)

    #     # Calculate ni values; ni intervals cannot be pre-computed in init()
    #     for i in range(natoms):
    #         itype = lammpsTools.symbol_to_type(self.atoms[i].symbol, self.types)
    #         # logging.info("Atom {0} of type {1}---------------".format(i, itype))
    #         Uprime_i = self.uprimes[:,i]
    #         Uprime_i = Uprime_i.reshape( (len(Uprime_i),1) )
    #
    #         # Compute phi contributions
    #
    #         # Pull per-atom neighbor distances and compute rho contribution
    #         pair_distances_bothways = self.pair_distances_bothways[i]
    #         pair_directions_bothways = self.pair_directions_bothways[i]
    #         pair_indices_bothways = self.pair_indices_bothways[i]
    #
    #         i_indices = pair_indices_bothways[:,0]
    #         j_indices = pair_indices_bothways[:,1]
    #
    #         rho_types_j = self.rho_index_info['pot_type_idx'][i]
    #         rho_indices_j = self.rho_index_info['interval_idx'][i]
    #
    #         # Forces from i to neighbors
    #         rho_coeffs_j = coeffs_from_indices(self.rhos,
    #                 {'pot_type_idx':rho_types_j, 'interval_idx':rho_indices_j})
    #
    #         # Rho prime values for neighbors
    #         rho_prime_coeffs_j = np.einsum('ij,jdk->idk', D, rho_coeffs_j)
    #         rho_prime_values_j = eval_all_polynomials(pair_distances_bothways,
    #                                           rho_prime_coeffs_j)
    #
    #         # Forces from neighbors to i
    #         itype = lammpsTools.symbol_to_type(self.atoms[i].symbol, self.types)
    #         rho_idx_i = meam.i_to_potl(itype, 'rho', self.ntypes)
    #
    #         rho_indices_i = self.rho_index_info['interval_idx_backwards'][i]
    #
    #         rho_coeffs_i = coeffs_from_indices(self.rhos,
    #                {'pot_type_idx':np.ones(len(
    #                 pair_distances_bothways))*rho_idx_i,
    #                 'interval_idx':rho_indices_i})
    #
    #         rho_prime_coeffs_i = np.einsum('ij,jdk->idk', D, rho_coeffs_i)
    #         rho_prime_values_i = eval_all_polynomials(pair_distances_bothways,
    #                                           rho_prime_coeffs_i)
    #
    #         # Combine forwards/backwards forces
    #         fpair = rho_prime_values_j*Uprime_i
    #         rho_prime_values_i = np.multiply(rho_prime_values_i, self.uprimes[:,
    #                                                    j_indices])
    #         fpair += rho_prime_values_i
    #         #fpair += phi_prime_values
    #
    #         # TODO: seem to be double-counting rho terms. hence the /= 2.0
    #         rho_forces = np.einsum('ij,jk->ijk',fpair, pair_directions_bothways)
    #         rho_forces /= 2.0
    #
    #         np.add.at(forces, (slice(None),i_indices, slice(None)), rho_forces)
    #         np.add.at(forces, (slice(None),j_indices, slice(None)), -rho_forces)
    #
    #         if len(self.triplet_values[i]) > 0: # check needed in case of dimer
    #
    #             # Unpack ordered sets of values and indexing information
    #             values_tuple    = self.triplet_values[i]
    #             rij_values      = values_tuple[:,0]
    #             rik_values      = values_tuple[:,1]
    #             cos_values        = values_tuple[:,2]
    #
    #             values_tuple    = self.triplet_unshifted[i]
    #             rij_values_unshifted      = values_tuple[:,0]
    #             rik_values_unshifted      = values_tuple[:,1]
    #             cos_values_unshifted        = values_tuple[:,2]
    #
    #             types_tuple = self.triplet_info['pot_type_idx'][i]
    #             fj_types    = types_tuple[:,0]
    #             fk_types    = types_tuple[:,1]
    #             g_types     = types_tuple[:,2]
    #
    #             # type dependent.
    #             intervals_tuple = self.triplet_info['interval_idx'][i]
    #             fj_intervals    = intervals_tuple[:,0]
    #             fk_intervals    = intervals_tuple[:,1]
    #             g_intervals     = intervals_tuple[:,2]
    #
    #             # Calculate three body contributions
    #             fj_coeffs = coeffs_from_indices(self.fs,
    #                         {'pot_type_idx':fj_types, 'interval_idx':fj_intervals})
    #             fk_coeffs = coeffs_from_indices(self.fs,
    #                         {'pot_type_idx':fk_types, 'interval_idx':fk_intervals})
    #             g_coeffs = coeffs_from_indices(self.gs,
    #                         {'pot_type_idx':g_types, 'interval_idx':g_intervals})
    #
    #             fj_results = eval_all_polynomials(rij_values, fj_coeffs)
    #             fk_results = eval_all_polynomials(rik_values, fk_coeffs)
    #             g_results = eval_all_polynomials(cos_values, g_coeffs)
    #
    #             # TODO: consider putting derivative evals in eval_all() 4 speed?
    #             # Take derivatives and compute forces
    #             fj_prime_coeffs = np.einsum('ij,jdk->idk', D, fj_coeffs)
    #             fk_prime_coeffs = np.einsum('ij,jdk->idk', D, fk_coeffs)
    #             g_prime_coeffs = np.einsum('ij,jdk->idk', D, g_coeffs)
    #
    #             fj_primes = eval_all_polynomials(rij_values, fj_prime_coeffs)
    #             fk_primes = eval_all_polynomials(rik_values, fk_prime_coeffs)
    #             g_primes = eval_all_polynomials(cos_values, g_prime_coeffs)
    #
    #             #logging.info("{0}, {1}, {2}".format(g_results.shape,
    #             #                        fk_results.shape, fj_primes.shape))
    #
    #             fij = -Uprime_i*np.multiply(g_results, np.multiply(
    #                 fk_results, fj_primes))
    #
    #             fik = -Uprime_i*np.multiply(g_results, np.multiply(
    #                 fj_results, fk_primes))
    #
    #             # logging.info("fij = {0}".format(fij))
    #             # logging.info("fik = {0}".format(fik))
    #             # logging.info("cos_values = {0}".format(cos_values_unshifted))
    #             # logging.info("rij_values = {0}".format(rij_values_unshifted))
    #             # logging.info("rik_values = {0}".format(rik_values_unshifted))
    #
    #             prefactor = Uprime_i*np.multiply(fj_results, np.multiply(
    #                 fk_results, g_primes))
    #             prefactor_ij = np.divide(prefactor, rij_values_unshifted)
    #             prefactor_ik = np.divide(prefactor, rik_values_unshifted)
    #
    #             fij += np.multiply(prefactor_ij, cos_values_unshifted)
    #             fik += np.multiply(prefactor_ik, cos_values_unshifted)
    #
    #             # logging.info("prefactor = {0}".format(prefactor))
    #             # logging.info("prefactor_ij = {0}".format(prefactor_ij))
    #             # logging.info("prefactor_ik = {0}".format(prefactor_ik))
    #
    #             jvec = self.triplet_directions[i][:,0]
    #             kvec = self.triplet_directions[i][:,1]
    #
    #             forces_j = np.einsum('ij,jk->ijk', fij, jvec)
    #             forces_j -= np.einsum('ij,jk->ijk', prefactor_ij, kvec)
    #
    #             forces_k = np.einsum('ij,jk->ijk', fik, kvec)
    #             forces_k -= np.einsum('ij,jk->ijk', prefactor_ik, jvec)
    #
    #             j_indices = self.triplet_indices[i][:,0]
    #             k_indices = self.triplet_indices[i][:,1]
    #
    #             np.add.at(forces, (slice(None), j_indices, slice(None)),
    #                     forces_j)
    #             np.add.at(forces, (slice(None), k_indices, slice(None)),
    #                     forces_k)
    #
    #             # logging.info("{0}, {1}: fij, prefactor_ij".format(fij,
    #             #                                                prefactor_ij))
    #             # logging.info("{0}, {1}: fik, prefactor_ik".format(fik,
    #             #                                                   prefactor_ik))
    #             # logging.info("{0}, {1}, {4}, fj_results, fk_results, "
    #             #              "g_results, {2},{3} j,k".format(
    #             #     fj_results, fk_results, fj_types+1, fk_types+1, g_results))
    #
    #             # logging.info("fj_results = {0}".format(fj_results))
    #             # logging.info("fj_primes = {0}".format(fj_primes))
    #             # logging.info("fk_results = {0}".format(fk_results))
    #             # logging.info("fk_primes = {0}".format(fk_primes))
    #             # logging.info("g_results = {0}".format(g_results))
    #             # logging.info("g_primes = {0}".format(g_primes))
    #             # logging.info("forces_j = {0}".format(forces_j))
    #             # logging.info("forces_k = {0}".format(forces_k))
    #
    #             forces[:,i,:] -= np.sum(forces_k, axis=1)
    #             forces[:,i,:] -= np.sum(forces_j, axis=1)
    #
    #             # TODO: pass disp vectors, not rij, then vectorize reduction
    #             # to rij_values?
    #
    #     #return forces.reshape((npots, natoms, 3))
    #     return forces

    # @property
    # def potentials(self):
    #     """Set of MEAM objects of various potentials"""
    #     return self._potentials
    #
    # @potentials.setter
    # def potentials(self, potentials):
    #     self._potentials = potentials
    #     self._types = potentials[0].types
    #     self._ntypes = len(self.types)
    #
    #     slices = np.split(potentials, self.indices, axis=1)
    #
    #     phis    = np.split(slices[0], self.nphi, axis=1)
    #     u       = np.split(slices[1], self.ntypes, axis=1)
    #     rho     = np.split(slices[2], self.ntypes, axis=1)
    #     f       = np.split(slices[3], self.ntypes, axis=1)
    #     g       = np.split(slices[4], self.nphi, axis=1)
    #
    #     self._cutoff = potentials[0].cutoff

if __name__ == "__main__":
    pass
