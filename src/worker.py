import numpy as np
import lammpsTools
import meam
import logging
import matplotlib.pyplot as plt

from ase.neighborlist import NeighborList
from scipy.sparse import diags

from spline import Spline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logging.disable(logging.CRITICAL)

class Worker:
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials

    Assumptions:
        - knot x-coordinates never change
        - the atomic structure never changes
        - boundary conditions always passed in as pairs of floats (typically
          d0,dN), then 'natural' or 'fixed' decided later
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
                have every atom type in it."""

        # Basic variable initialization
        self.atoms      = atoms
        self.types      = types

        ntypes          = len(self.types)
        self.ntypes     = ntypes

        nphi            = int((self.ntypes+1)*self.ntypes/2)
        self.nphi       = nphi

        # Initialize splines; group by type and calculate potential cutoff range

        knots_split = np.split(knot_xcoords, x_indices[1:])

        splines = []

        for i in range(ntypes*(ntypes+4)):
            idx = x_indices[i]

            # TODO: could specify bc outside of Worker & pass in
            # check if phi/rho or f
            if (i < nphi+ntypes) or ((i >= nphi+2*ntypes) and
                                     (i < nphi+3*ntypes)):
                bc_type = ('natural','fixed')
            else:
                bc_type = ('natural','natural')

            # for comparing against TiO.meam.spline; all are 'fixed'
            bc_type = ('fixed', 'fixed')

            if (i >= nphi+ntypes) and (i < nphi+2*ntypes): # U splines
                s = InnerSpline(knots_split[i], bc_type)
            else:
                s = WorkerSpline(knots_split[i], bc_type)

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

        self.max_phi_knots  = max([len(s.x) for s in self.phis])
        self.max_rho_knots  = max([len(s.x) for s in self.rhos])
        self.max_u_knots    = max([len(s.x) for s in self.us])
        self.max_f_knots    = max([len(s.x) for s in self.fs])
        self.max_g_knots    = max([len(s.x) for s in self.gs])

        radial_fxns = self.phis + self.rhos + self.fs

        self.cutoff = np.max([max(s.x) for s in radial_fxns])

        # Building neighbor lists
        natoms = len(atoms)

        # No double counting; needed for pair interactions
        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=False, skin=0.0)
        nl_noboth.build(atoms)

        total_num_neighbors = sum(len(nl_noboth.get_neighbors(i)) for i in
                                  range(natoms))

        self.phi_structure_array = []

        # Allows double counting; needed for embedding energy calculations
        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=True, skin=0.0)
        nl.build(atoms)

        for i in range(natoms):
            # Record atom type info
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            self.us[meam.i_to_potl(itype)].natoms += 1 # for computing
            # zero-point-energy

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            phis_struct_vec = None
            # Stores pair information for phi
            for j,offset in zip(neighbors_noboth, offsets_noboth):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                jvec = jpos - ipos
                rij = np.linalg.norm(jvec)
                jvec /= rij

                # TODO: how do you handle directionality for forces?
                # maybe each el in structure vec could be 3D?

                phi_idx = meam.ij_to_potl(itype, jtype, self.ntypes)

                self.phis[phi_idx].add_to_struct_vec(rij)

            # Store distances, angle, and index info for embedding terms
            j_counter = 1 # for tracking neighbor
            for j,offset in zip(neighbors,offsets):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                jvec = jpos - ipos
                rij = np.linalg.norm(jvec)
                jvec /= rij

                a = jpos - ipos
                na = np.linalg.norm(a)

                # Rho information; need forward AND backwards info for forces
                rho_idx = meam.i_to_potl(jtype)

                self.us[meam.i_to_potl(itype)].update_struct_vec_dict(rij,
                                                               rho_idx, 'rho')

            #     # fj information
            #     fj_idx = meam.i_to_potl(jtype, self.ntypes)
            #
            #     rij -= self.rhos[0][rho_idx].knotsx[rho_knot_num]
            #
            #     rij += self.rhos[0][rho_idx].knotsx[rho_knot_num]
            #     rij -= self.fs[0][fj_idx].knotsx[fj_knot_num]
            #
            #     # Three-body contributions
            #     for k,offset in zip(neighbors[j_counter:], offsets[j_counter:]):
            #         if k != j:
            #             ktype = lammpsTools.symbol_to_type(atoms[k].symbol,
            #                                                self.types)
            #             kpos = atoms[k].position + np.dot(offset,
            #                                               atoms.get_cell())
            #
            #             kvec = kpos - ipos
            #             rik = np.linalg.norm(kvec)
            #             kvec /= rik
            #
            #             b = kpos - ipos
            #             nb = np.linalg.norm(b)
            #
            #             cos_theta = np.dot(a,b)/na/nb
            #
            #             # fk information
            #             fk_idx = meam.i_to_potl(ktype, 'f', self.ntypes)
            #
            #             # g information
            #             g_idx = meam.ij_to_potl(jtype, ktype, 'g', self.ntypes)
            #
            #             rik -= self.fs[0][fk_idx].knotsx[fk_knot_num]
            #             cos_theta -= self.gs[0][g_idx].knotsx[g_knot_num]
            #
            #             rik += self.fs[0][fk_idx].knotsx[fk_knot_num]
            #             cos_theta += self.gs[0][g_idx].knotsx[g_knot_num]
            #             tmp_rij = rij + self.fs[0][fj_idx].knotsx[fj_knot_num]
            #
            #     j_counter += 1

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

            indices (list):
                since each spline does not necessarily have the same number
                of knots, a list of indices must be provided to deliminate
                each spline in the 1D vector. 'indices' will be exactly
                N*(N+4)-1 elements, since there are N*(N+4) total splines
                indies"""

        # TODO: should __call__() take in just a single potential?

        # TODO: Worker has list of Pots; each Pot is a list of WorkerSplines
        # TODO: phis[i] needs to be phis[i][phi_idx]; do in init()?

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

        energy = 0.

        # Pair interactions
        for i in range(len(self.phis)):
            y = phi_pvecs[i]
            s = self.phis[i]

            energy += s(y)

        # Electron density embedding cost
        for i in range(len(self.us)):
            u = self.us[i]
            y_u = u_pvecs[i]


            for j in range(len(self.rhos)):
                # TODO: each ni should only have the neighbors of that i
                rho = self.rhos[j]
                rho.struct_vec = u.get_struct_vec(j, 'rho')

                y_rho = rho_pvecs[j]

                ni = rho(y_rho)
                u.add_to_struct_vec(ni)

            val2 = u(y_u)
            energy += val2

        return energy

        # Calculate ni values; ni intervals cannot be pre-computed in init()
        # total_ni = np.zeros( (len(potentials),natoms) )

        # for i in range(natoms):
        #
        #     # Pull per-atom neighbor distances and compute rho contribution
        #     pair_distances_bothways = self.pair_distances_bothways[i]
        #     rho_types = self.rho_index_info['pot_type_idx'][i]
        #     rho_indices = self.rho_index_info['interval_idx'][i]
        #
        #     rho_coeffs = coeffs_from_indices(self.rhos,
        #                 {'pot_type_idx':rho_types, 'interval_idx':rho_indices})
        #     results = eval_all_polynomials(pair_distances_bothways, rho_coeffs)
        #     total_ni[:,i] += np.sum(results, axis=1)
        #
        #     if len(self.triplet_values[i]) > 0: # check needed in case of dimer
        #
        #         # Unpack ordered sets of values and indexing information
        #         values_tuple    = self.triplet_values[i]
        #         rij_values      = values_tuple[:,0]
        #         rik_values      = values_tuple[:,1]
        #         cos_values        = values_tuple[:,2]
        #
        #         types_tuple = self.triplet_info['pot_type_idx'][i]
        #         fj_types    = types_tuple[:,0]
        #         fk_types    = types_tuple[:,1]
        #         g_types     = types_tuple[:,2]
        #
        #         intervals_tuple = self.triplet_info['interval_idx'][i]
        #         fj_intervals    = intervals_tuple[:,0]
        #         fk_intervals    = intervals_tuple[:,1]
        #         g_intervals     = intervals_tuple[:,2]
        #
        #         # Calculate three body contributions
        #         fj_coeffs = coeffs_from_indices(self.fs,
        #                     {'pot_type_idx':fj_types, 'interval_idx':fj_intervals})
        #         fk_coeffs = coeffs_from_indices(self.fs,
        #                     {'pot_type_idx':fk_types, 'interval_idx':fk_intervals})
        #         g_coeffs = coeffs_from_indices(self.gs,
        #                     {'pot_type_idx':g_types, 'interval_idx':g_intervals})
        #
        #         fj_results = eval_all_polynomials(rij_values, fj_coeffs)
        #         fk_results = eval_all_polynomials(rik_values, fk_coeffs)
        #         g_results = eval_all_polynomials(cos_values, g_coeffs)
        #
        #         results = np.multiply(fj_results,np.multiply(fk_results, g_results))
        #         total_ni[:,i] += np.sum(results, axis=1)
        #
        # # Perform binning and shifting for each ni (per atom) in each potential
        # knots = self.us[0][0].knotsx
        # for p in range(total_ni.shape[0]):
        #     ni_indices_for_one_pot = np.zeros(natoms)
        #
        #     for q in range(total_ni.shape[1]):
        #         # Shift according to pre-computed U index info
        #         u_idx = self.u_index_info['pot_type_idx'][q]
        #
        #         ni = total_ni[p][q]
        #
        #         u_interval_num, u_knot_num = intervals_from_splines(
        #             self.us,ni,u_idx)
        #
        #         ni_indices_for_one_pot[q] = u_interval_num
        #         total_ni[p][q] -= knots[u_knot_num]
        #
        #     self.u_index_info['interval_idx'].append(ni_indices_for_one_pot)
        #
        # self.u_index_info['interval_idx'] = np.array(self.u_index_info[
        #                                                  'interval_idx'])
        #
        # # Compute U values and derivatives; add to energy and forces
        # u_types = self.u_index_info['pot_type_idx']
        #
        # self.uprimes = np.zeros( (len(potentials),natoms) )
        # for i,row  in enumerate(total_ni):
        #     u_indices = self.u_index_info['interval_idx'][i]
        #
        #     u_coeffs = coeffs_from_indices([self.us[i]],
        #                 {'pot_type_idx':u_types, 'interval_idx':u_indices})
        #
        #     results = eval_all_polynomials(row, u_coeffs)
        #     energies[i] += np.sum(results, axis=1)
        #
        #     # dots each layer of u_coeffs with the derivative matrix
        #     uprime_coeffs = np.einsum('ij,jdk->idk', D, u_coeffs)
        #
        #     results = eval_all_polynomials(row, uprime_coeffs)
        #     self.uprimes[i,:] = eval_all_polynomials(row, uprime_coeffs)
        #
        # # Subtract off zero-point energies
        # zero_atom_energy = np.array([self.zero_atom_energies[:,z] for z in
        #                     u_types]).transpose()
        # zero_atom_energy = np.sum(zero_atom_energy, axis=1)
        #
        # energies -= zero_atom_energy
        #
        # return energies

    # def compute_forces(self, potentials):
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
    #
    #     # TODO: switch to full vectorization; tags shouldn't be needed
    #     # TODO: unless tags actually improve speed?
    #
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
    #         # Potential index and atom tags for neighbors
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

    @property
    def potentials(self):
        """Set of MEAM objects of various potentials"""
        return self._potentials

    @potentials.setter
    def potentials(self, potentials):
        self._potentials = potentials
        self._types = potentials[0].types
        self._ntypes = len(self.types)

        slices = np.split(potentials, self.indices, axis=1)

        phis    = np.split(slices[0], self.nphi, axis=1)
        u       = np.split(slices[1], self.ntypes, axis=1)
        rho     = np.split(slices[2], self.ntypes, axis=1)
        f       = np.split(slices[3], self.ntypes, axis=1)
        g       = np.split(slices[4], self.nphi, axis=1)

        # TODO: read_from_file() to test working properly

        # N = len(potentials)
        # self.phis   = [potentials[i].phis for i in range(N)]
        # self.rhos   = [potentials[i].rhos for i in range(N)]
        # self.us     = [potentials[i].us for i in range(N)]
        # self.fs     = [potentials[i].fs for i in range(N)]
        # self.gs     = [potentials[i].gs for i in range(N)]

        # Initialize zero-point embedding energies
        # u_shift = nphi + 2
        # self.zero_atom_energies = np.zeros((len(potentials), self.ntypes))
        # for i,p in enumerate(potentials):
        #     for j in range(self.ntypes):
        #         self.zero_atom_energies[i][j] = potentials[j+u_shift](0.0)

        # TODO: zero_atom_energies need to be computed in __call__()

        self._cutoff = potentials[0].cutoff
    # TODO: can we condense oneway/bothways? derive oneway from bothways?

def get_ymat(y, y2, intervals):
    """Extracts the matrix of [y_j, y_j+1, y''_j, y''_j+1] from a set of
    splines for the given intervals.

    Args:
        y (np.arr):
            the knot y-coordinates for all of the splines of one type for each
            potential (e.g. the phi_Ti splines for every potential). Each row is
            the y-coordinates of the knots for a single potential.
        y2 (np.arr):
            the second derivatives at the knots.
        intervals (list):
            an ordered list of integers specifying which intervals in the
            splines to extract (e.g. [1,2,3] meaning the first three pair
            distances fall into the intervals starting at knots 1, 2,
            and 3 respectively).

    Returns:
        ymat (np.arr):
            the ordered matrix of [y_j, y_j+1, y''_j, y''_j+1] with
            dimensions (N x 4 x P) where N is the number of values being
            computed and P is the number of potentials"""

    intervals = np.array(intervals)

    N = len(intervals)
    P = y.shape[0]

    ymat = np.zeros((N, 4, P))

    s0 = y[:,intervals]     # y_j
    s1 = y[:,intervals+1]   # y_j+1
    s2 = y2[:, intervals]   # y''_j
    s3 = y2[:, intervals+1] # y''_j+1

    ymat[:,:,0] = s0
    ymat[:,:,1] = s1
    ymat[:,:,2] = s2
    ymat[:,:,3] = s3

    return ymat

def read_from_file(atom_file_name, potential_file_name):
    """Builds a Worker from files specifying the atom structure and the
    potential.

    Args:
        atoms_file_name (str):
            LAMMPS style data file name; types are assumed from potential_file

        potential_file_name (str):
            LAMMPS style spline meam potential file"""

    # read in potential
    try:
        f = open(potential_file_name, 'r')

        f.readline()                    # Remove header
        temp = f.readline().split()     # 'meam/spline ...' line
        types = temp[2:]
        ntypes = len(types)

        nsplines = ntypes*(ntypes+4)    # Based on how fxns in MEAM are defined

        # Calculate the number of splines for phi/g each
        nphi = (ntypes + 1) * ntypes / 2

        # Build all splines; separate into different types later
        knot_x_points = []
        knot_y_points = []
        indices     = [] # tracks ends of phi, rho, u, ,f, g groups
        idx         = 0

        for i in range(nsplines):

            # if      i == nphi:          indices.append(idx)
            # elif    i == nphi+ntypes:   indices.append(idx)
            # elif    i == nphi+2*ntypes: indices.append(idx)
            # elif    i == nphi+3*ntypes: indices.append(idx)

            f.readline()                # throw away 'spline3eq' line
            nknots  = int(f.readline())
            idx     += nknots

            d0, dN = [float(el) for el in f.readline().split()]
            # TODO: do we need to keep d0, dN? also, move this all to worker.py

            for j in range(nknots):
                x,y,y2 = [np.float(el) for el in f.readline().split()]
                knot_x_points.append(x)
                knot_y_points.append(y)

            indices.append(idx)

        knot_x_points   = np.array(knot_x_points)
        indices         = np.array(indices[:-1])
        knot_y_points   = np.array(knot_y_points)


    except IOError as error:
        raise IOError("Could not open potential file: {0}".format(
            potential_file_name))

    try:
        atoms = lammpsTools.atoms_from_file(atom_file_name, types)

    except IOError as error:
        raise IOError("Could not open structure file: {0}".format(
            potential_file_name))

    worker = Worker(atoms, knot_x_points, indices, types)

    return worker, knot_y_points

class WorkerSpline:
    """A representation of a cubic spline specifically tailored to meet
    the needs of a Worker object.

    Attributes:
        x (np.arr):
            x-coordinates of knot points

        h (float):
            knot spacing (assumes equally-spaced)

        y (np.arr):
            y-coordinates of knot points; only set by solve_for_derivs()

        y1 (np.arr):
            first derivatives at knot points; only set by solve_for_derivs()

        M (np.arr):
            M = AB where A and B are the matrices from the system of
            equations that comes from matching second derivatives at knot
            points (see build_M() for details)

        cutoff (tuple-like):
            upper and lower bounds of knots (x[0], x[-1])

        index (int):
            the index of the first knot of this spline when the knots of all
            splines in a potential are grouped into a single 1D vector

        end_derivs (tuple-like):
            first derivatives at endpoints

        bc_type (tuple):
            2-element tuple corresponding to boundary conditions at LHS/RHS
            respectively. Options are 'natural' (zero second derivative) and
            'fixed' (fixed first derivative). If 'fixed' on one/both ends,
            self.end_derivatives cannot be None for that end

        struct_vec (np.arr):
            2D array for evaluating the spline on the structure; each row
            corresponds to a single pair/triplet evaluation

    Notes:
        This object is distinct from a spline.Spline since it requires some
        attributes and functionality that a spline.Spline doesn't have."""


    def __init__(self, x, bc_type):

        if not np.all(x[1:] > x[:-1], axis=0):
            raise ValueError("x must be strictly increasing")

        if not ((bc_type[0]=='natural') or (bc_type[0] == 'fixed')):
            raise ValueError("boundary conditions must be one of 'natural' or"
                             "'fixed'")
        if not ((bc_type[1]=='natural') or (bc_type[1] == 'fixed')):
            raise ValueError("boundary conditions must be one of 'natural' or"
                             "'fixed'")

        # Set at beginning
        self.x = x;
        self.h = x[1]-x[0]

        self.bc_type = bc_type

        self.cutoff = (x[0], x[-1])
        self.M = build_M(len(x), self.h, self.bc_type)

        # Set at some point
        self.index = None
        self.struct_vec = None

        # Set on evaluation
        self.y = None
        self.y1 = None
        self.end_derivs = None

    def __call__(self, y):

        if self.struct_vec is None:
            return 0
            # raise ValueError("Structure vector has not been built yet")

        self.y, self.end_derivs = np.split(y, [-2])
        self.y1 = self.M @ y.transpose()

        z = np.concatenate((self.y, self.y1))

        return np.sum(self.struct_vec @ z.transpose())

    def get_abcd(self, x):
        """Calculates the coefficients needed for spline interpolation.

            x (float):
                point at which to evaluate spline

        Returns:
            vec (np.arr):
                vector of length len(knots)*2; this formatting is used since for
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

            with t = (x-x_k)/(x_k+1 - x_k)"""

        knots = self.x.copy()

        h_00 = lambda t: (1+2*t)*(1-t)**2
        h_10 = lambda t: t*(1-t)**2
        h_01 = lambda t: (t**2)*(3-2*t)
        h_11 = lambda t: (t**2)*(t-1)

        h = knots[1] - knots[0]

        # Find spline interval; -1 to account for zero indexing
        k = int(np.floor((x-knots[0])/h))

        nknots = len(knots)
        vec = np.zeros(2*nknots)

        if k < 0: # LHS extrapolation
            k = 0

            A = 1
            B = x - knots[0] # negative for correct sign with derivative

            vec[k] = A; vec[k+nknots] = B

        elif k >= nknots-1: # RHS extrapolation
            k = nknots-1
            C = 1
            D = x - knots[-1]

            vec[k] = C; vec[k+nknots] = D
        else:
            prefactor  = (knots[k+1] - knots[k])

            t = (x - knots[k])/prefactor

            A = h_00(t)
            B = h_10(t)*prefactor
            C = h_01(t)
            D = h_11(t)*prefactor

            vec[k] = A
            vec[k+nknots] = B
            vec[k+1] = C
            vec[k+1+nknots] = D

        return vec

    def add_to_struct_vec(self, val):

        abcd = self.get_abcd(val)

        if self.struct_vec is None:
            self.struct_vec = abcd
        else:
            self.struct_vec = np.vstack((self.struct_vec, abcd))

    def to_normal_spline(self):
        if self.y is None:
            raise ValueError("Must specify y before converting to Spline")

        bc_type = []
        for i in range(len(self.bc_type)):
            if self.bc_type[i] == 'fixed':
                bc_type.append((1,self.y[-i-1]))
            else:
                bc_type.append(self.bc_type[-i-1])

        return Spline(self.x, self.y[:-2], bc_type=bc_type,
                      end_derivs=self.y[-2:])

    def plot(self, fname=''):

        s = self.to_normal_spline()

        if self.y is None:
            raise ValueError("Must specify y before plotting")

        plt.figure()
        plt.plot(self.x, self.y[:-2], 'ro', label='knots')

        plot_x = np.linspace(self.x[0]-2, self.x[-1]+2, 1000)
        plot_y = np.zeros(len(plot_x))

        for i in range(len(plot_x)):
            plot_y[i] = s(plot_x[i])

        plt.plot(plot_x, plot_y)
        plt.legend()
        plt.show()

class InnerSpline(WorkerSpline):
    """Special case of a WorkerSpline that is used for rho, f, and g
    since they are 'inside' of the U function and have to keep their per-atom
    contributions separate until they pass the results to U

    Attributes:
        struct_vec_dict (dict):
            key = 'rho', 'f', or 'g'
            val = list [np.arr] where each element in the list is a structure
                vector corresponding to the i-th spline of the given type

        ni (float):
            value that the embedding function U will be evaluated at on
            __call__()

        natoms (float):
            the number of atoms that are being embedded; used to calculate
            the zero-point-energy"""

    def __init__(self, x, bc_type):
        super(InnerSpline, self).__init__(x, bc_type)

        self.struct_vec_dict = {'rho':[], 'f':[], 'g':[]}
        self.ni = None
        self.natoms = 0

    # TODO: ni should be updated for each atom, added to struct_vec
    def __call__(self, y):

        val = super(InnerSpline, self).__call__(y) + \
              self.compute_zero_point_energy(y)

        return np.sum(val)

    def get_struct_vec(self, i, type):
        """Safely returns either None or the structure vector of the desired
        spline.

        Args:
            i (int):
                spline number
            type (str):
                type of spline; one of 'rho', 'f', or 'g'"""

        if len(self.struct_vec_dict[type]) < (i+1):
            return None
        else:
            return self.struct_vec_dict[type][i]

    def compute_zero_point_energy(self, y):
        """Computes the zero-point energy using the given y

        Args:
            y (np.arr):
                parameter vector to evaluate with"""

        # safe return if there are no atoms for this spline
        if self.natoms == 0:
            return 0

        row = self.get_abcd(0.)

        temp = np.tile(row, self.natoms).reshape((self.natoms, len(row)))

        self.y, self.end_derivs = np.split(y, [-2])
        self.y1 = self.M @ y.transpose()

        z = np.concatenate((self.y, self.y1))

        return self.struct_vec @ z.transpose()

    # rzm: initializing rhos/fs/gs/
    def update_struct_vec_dict(self, val, i, type):
        """Overrides WorkerSpline.add_to_struct_vec()

        Updates the structure vector of the <i>th spline of type <type> for a
        value of r

        Args:
            val (float):
                value to evaluate the spline at
            i (int):
                index of spline
            type (str):
                type of spline; can be one of 'rho', 'f', or 'g'"""

        type = type.lower()

        abcd = self.get_abcd(val)

        if len(self.struct_vec_dict[type]) < (i+1):
            struct_vec = abcd
            self.struct_vec_dict[type].append(struct_vec)
        else:
            struct_vec = np.vstack((self.struct_vec_dict[type][i], abcd))
            self.struct_vec_dict[type][i] = struct_vec

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
        h_10 = t^2 ( 3-2t)
        h_01 = t (1-t)^2
        h_11 = t^2 (t-1)

        with t = (x-x_k)/dx

    which means that the h''_ij functions are:

        h''_00 = 12t - 6
        h''_01 = -12t + 6
        h''_10 = 6t - 4
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
            A^(-1)B"""

    n = num_x - 2

    if n <= 0:
        raise ValueError("the number of knots must be greater than 2")

    # note that values for h''_ij(0) and h''_ij(1) are substituted in
    A = diags(np.array([2,8,2]), [0,1,2], (n,n+2))
    A = A.toarray()

    B = diags([-6, 0, 6], [0,1,2], (n,n+2))
    B = B.toarray()

    bc_lhs, bc_rhs = bc_type
    bc_lhs = bc_lhs.lower()
    bc_rhs = bc_rhs.lower()

    # Determine 1st equation based on LHS boundary condition
    if bc_lhs == 'natural':
        topA = np.zeros(n+2).reshape((1,n+2)); topA[0,0] = -4; topA[0,1] = -2
        topB = np.zeros(n+2).reshape((1,n+2)); topB[0,0] = 6; topB[0,1] = -6
    elif bc_lhs == 'fixed':
        topA = np.zeros(n+2).reshape((1,n+2)); topA[0,0] = 1/dx;
        topB = np.zeros(n+2).reshape((1,n+2));
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' "
                         "or 'fixed'")

    # Determine last equation based on RHS boundary condition
    if bc_rhs == 'natural':
        botA = np.zeros(n+2).reshape((1,n+2)); botA[0,-2] = 2; botA[0,-1] = 4
        botB = np.zeros(n+2).reshape((1,n+2)); botB[0,-2] = -6; botB[0,-1] = 6
    elif bc_rhs == 'fixed':
        botA = np.zeros(n+2).reshape((1,n+2)); botA[0,-1] = 1/dx;
        botB = np.zeros(n+2).reshape((1,n+2));# botB[0,-2] = 6; botB[0,-1] = -6
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' "
                         "or 'fixed'")

    rightB = np.zeros((n+2,2)); rightB[0,0] = rightB[-1,-1] = 1

    # Build matrices
    A = np.concatenate((topA, A), axis=0)
    A = np.concatenate((A, botA), axis=0)

    A *= dx

    B = np.concatenate((topB, B), axis=0)
    B = np.concatenate((B, botB), axis=0)
    B = np.concatenate((B, rightB), axis=1)

    # M = A^(-1)B
    return np.dot(np.linalg.inv(A), B)

if __name__ == "__main__":

    worker, y_vals = read_from_file('Ti_only_crowd.Ti', 'TiO.meam.spline')

    logging.info("{0}".format(worker.compute_energies(y_vals)))
