import numpy as np
import lammpsTools
import meam
import logging

from numpy.polynomial.polynomial import polyval
from ase.neighborlist import NeighborList

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.disable(logging.CRITICAL)

class Worker(object):

    def __init__(self):
        self.atoms = []
        self.potentials = []

    @property
    def atoms(self):
        """List of structures; each structure is an array of coordinates"""
        return self._structs

    @atoms.setter
    def atoms(self, structs):
        self._structs = structs

    @property
    def potentials(self):
        return self._potentials

    @potentials.setter
    def potentials(self,c):
        raise NotImplementedError

    def compute_energies(self,structs):
        raise NotImplementedError

    def compute_forces(self,structs):
        raise NotImplementedError

class WorkerManyPotentialsOneStruct(Worker):
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials"""

    # TODO: instead of multiple worker types, could just do special evals based
    # on if A, X are matrices or not? Downside: can't manually decide method

    # TODO: __init__() should only take in knot x-coordinates for each spline
    # TODO: an assumption is made that all potentials have the same cutoffs

    # TODO: assumption that len(potentials) is constant across init() & call()

    def __init__(self, atoms, potentials):

        # Basic variable initialization
        super(Worker,self).__init__()
        self.atoms = atoms
        self.potentials = potentials    # also sets self.cutoff

        # Condensed structure information (ordered distances and angles)
        #TODO: is it faster to build np.arr of desired length?
        self.pair_distances_oneway      = [] # 1D, (iindex, jindex)
        self.pair_directions_oneway     = [] # 2D, 1 direction per neighbor
        self.pair_indices_oneway        = [] # 1D, list of atom id's

        self.pair_distances_bothways    = [] # 1D, list of phi pair distances
        self.pair_directions_bothways   = [] # 3D, 1 dir per neighbor per atom
        self.pair_indices_bothways      = [] # 1D, (iindex, jindex) atom id's

        self.triplet_unshifted          = [] # 1D, (fj, fk, cos) unshifted
        self.triplet_values             = [] # 1D, (fj, fk, cos)
        self.triplet_directions         = [] # 1D, (jpos-ipos, kpos-ipos)
        self.triplet_indices            = [] # 1D, (jindex, kindex) atom id's
        # TODO: redundancy; triplet info has overlap with pair_distances

        # Indexing information specifying atom type and interval number
        self.phi_index_info = {'pot_type_idx':[], 'interval_idx':[]}
        self.rho_index_info = {'pot_type_idx':[], 'interval_idx':[],
                               'interval_idx_backwards':[]}
        self.u_index_info   = {'pot_type_idx':[], 'interval_idx':[]}
        self.triplet_info   = {'pot_type_idx':[], 'interval_idx':[]} # (fj,fk,g)

        # Building neighbor lists
        natoms = len(atoms)

        # No double counting; needed for pair interactions
        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=False, skin=0.0)
        nl_noboth.build(atoms)

        # Allows double counting; needed for embedding energy calculations
        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=True, skin=0.0)
        nl.build(atoms)

        # Calculate E_i for each atom
        for i in range(natoms):
            # Record atom type info
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            u_idx = meam.i_to_potl(itype)
            self.u_index_info['pot_type_idx'].append(u_idx)

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # Stores pair information for phi
            for j,offset in zip(neighbors_noboth, offsets_noboth):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                jvec = jpos - ipos
                rij = np.linalg.norm(jvec)
                jvec /= rij

                self.pair_directions_oneway.append(jvec)
                self.pair_indices_oneway.append((i,j))

                phi_idx = meam.ij_to_potl(itype,jtype,self.ntypes)

                phi_spline_num, phi_knot_num = intervals_from_splines(
                    self.phis,rij,phi_idx)

                self.phi_index_info['pot_type_idx'].append(phi_idx)
                self.phi_index_info['interval_idx'].append(phi_spline_num)

                # knot index != cmat index since cmat has extrapolation splines
                rij -= self.phis[0][phi_idx].knotsx[phi_knot_num]
                self.pair_distances_oneway.append(rij)

            # Initialize per-atom value/index lists
            pair_distances_for_one_atom     = []
            pair_directions_for_one_atom    = []
            pair_indices_for_one_atom       = []

            triplet_values_for_one_atom     = []

            rho_type_indices_for_one_atom   = []
            rho_spline_indices_for_one_atom_forwards = []
            rho_spline_indices_for_one_atom_backwards = []

            triplet_unshifted_one_atom      = []
            triplet_types_for_one_atom      = []
            triplet_intervals_for_one_atom  = []
            triplet_directions_for_one_atom = []
            triplet_indices_for_one_atom    = []

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

                rho_spline_num, rho_knot_num = intervals_from_splines(
                    self.rhos,rij,rho_idx)
                rho_type_indices_for_one_atom.append(rho_idx)
                rho_spline_indices_for_one_atom_forwards.append(rho_spline_num)

                rho_idx_i = meam.i_to_potl(itype)
                rho_spline_num_i,_ = intervals_from_splines(
                    self.rhos,rij,rho_idx_i)
                rho_spline_indices_for_one_atom_backwards.append(
                    rho_spline_num_i)

                # fj information
                fj_idx = meam.i_to_potl(jtype)

                fj_spline_num, fj_knot_num = intervals_from_splines(
                    self.fs,rij,fj_idx)

                rij -= self.rhos[0][rho_idx].knotsx[rho_knot_num]

                pair_distances_for_one_atom.append(rij)
                pair_directions_for_one_atom.append(jvec)
                pair_indices_for_one_atom.append((i,j))

                rij += self.rhos[0][rho_idx].knotsx[rho_knot_num]
                rij -= self.fs[0][fj_idx].knotsx[fj_knot_num]
                # Three-body contributions
                for k,offset in zip(neighbors[j_counter:], offsets[j_counter:]):
                    if k != j:
                        ktype = lammpsTools.symbol_to_type(atoms[k].symbol,
                                                           self.types)
                        kpos = atoms[k].position + np.dot(offset,
                                                          atoms.get_cell())

                        kvec = kpos - ipos
                        rik = np.linalg.norm(kvec)
                        kvec /= rik

                        triplet_directions_for_one_atom.append( (jvec, kvec) )

                        b = kpos - ipos
                        nb = np.linalg.norm(b)

                        cos_theta = np.dot(a,b)/na/nb

                        # fk information
                        fk_idx = meam.i_to_potl(ktype)

                        fk_spline_num, fk_knot_num = \
                            intervals_from_splines(self.fs,rik,fk_idx)

                        # g information
                        g_idx = meam.ij_to_potl(jtype, ktype, self.ntypes)

                        g_spline_num, g_knot_num = \
                            intervals_from_splines(self.gs,cos_theta, g_idx)

                        rik -= self.fs[0][fk_idx].knotsx[fk_knot_num]
                        cos_theta -= self.gs[0][g_idx].knotsx[g_knot_num]

                        # Organize triplet information
                        tup1 = np.array([rij, rik, cos_theta])
                        tup2 = np.array([fj_idx, fk_idx, g_idx])
                        tup3 = np.array([fj_spline_num, fk_spline_num,
                                         g_spline_num])

                        rik += self.fs[0][fk_idx].knotsx[fk_knot_num]
                        cos_theta += self.gs[0][g_idx].knotsx[g_knot_num]
                        tmp_rij = rij + self.fs[0][fj_idx].knotsx[fj_knot_num]

                        tup0 = np.array([tmp_rij, rik, cos_theta])
                        triplet_unshifted_one_atom.append(tup0)

                        triplet_values_for_one_atom.append(tup1)
                        triplet_types_for_one_atom.append(tup2)
                        triplet_intervals_for_one_atom.append(tup3)
                        triplet_indices_for_one_atom.append((j,k))

                j_counter += 1

            # Add lists and convert everything to arrays for easy indexing
            self.triplet_unshifted.append(
                np.array(triplet_unshifted_one_atom))

            self.triplet_directions.append(
                np.array(triplet_directions_for_one_atom))

            self.triplet_indices.append(
                np.array(triplet_indices_for_one_atom))

            self.triplet_info['pot_type_idx'].append(
                np.array(triplet_types_for_one_atom))

            self.triplet_info['interval_idx'].append(
                np.array(triplet_intervals_for_one_atom))

            self.triplet_values.append(
                np.array(triplet_values_for_one_atom))

            self.pair_distances_bothways.append(
                np.array(pair_distances_for_one_atom))

            self.pair_directions_bothways.append(
                np.array(pair_directions_for_one_atom))

            self.pair_indices_bothways.append(
                np.array(pair_indices_for_one_atom))

            self.rho_index_info['pot_type_idx'].append(
                np.array(rho_type_indices_for_one_atom))

            self.rho_index_info['interval_idx'].append(
                np.array(rho_spline_indices_for_one_atom_forwards))

            self.rho_index_info['interval_idx_backwards'].append(
                np.array(rho_spline_indices_for_one_atom_backwards))

        self.pair_directions_oneway = np.array(self.pair_directions_oneway)
        self.pair_indices_oneway = np.array(self.pair_indices_oneway)

        self.triplet_unshifted = np.array(self.triplet_unshifted)

        self.triplet_info['pot_type_idx'] = np.array(self.triplet_info['pot_type_idx'])
        self.triplet_info['interval_idx'] = np.array(self.triplet_info['interval_idx'])
        self.triplet_values = np.array(self.triplet_values)

        self.pair_distances_oneway = np.array(self.pair_distances_oneway)
        self.pair_directions_oneway = np.array(self.pair_directions_oneway)

        self.pair_distances_bothways = np.array(self.pair_distances_bothways)

    def compute_energies(self, potentials):
        """Calculates energies for all potentials using information
        pre-computed during initialization."""
        #TODO: only thing changing should be y-coords?
        # changing y-coords changes splines, changes coeffs
        # maybe could use cubic spline eval equation, not polyval()

        # TODO: should __call__() take in just a single potential?

        # TODO: standardize ordering of matrix indices

        # TODO: is it worth creating a huge matrix of ordered coeffs so that
        # it's just a single matrix multiplication? probably yes because
        # Python for-loops are terrible! problem: need to zero pad to make
        # same size

        # TODO: turn coeffs into array of knot y-values

        # TODO: eventually move away from scipy Spline, towards Recipes equation

        D = np.array([[0,1,0,0], [0,0,2,0], [0,0,0,3]])

        # TODO: refactor len(self.atoms) -> natoms, len(potentials) -> npots
        self.potentials = potentials
        energies = np.zeros(len(potentials))

        # Compute phi contribution to total energy
        phi_coeffs = coeffs_from_indices(self.phis, self.phi_index_info)
        results = eval_all_polynomials(self.pair_distances_oneway, phi_coeffs)
        energies += np.sum(results, axis=1)

        # Calculate ni values; ni intervals cannot be pre-computed in init()
        total_ni = np.zeros( (len(potentials),len(self.atoms)) )

        for i in range(len(self.atoms)):
            forces_i = np.zeros((3,))

            # Pull per-atom neighbor distances and compute rho contribution
            pair_distances_bothways = self.pair_distances_bothways[i]
            rho_types = self.rho_index_info['pot_type_idx'][i]
            rho_indices = self.rho_index_info['interval_idx'][i]

            rho_coeffs = coeffs_from_indices(self.rhos,
                        {'pot_type_idx':rho_types, 'interval_idx':rho_indices})
            results = eval_all_polynomials(pair_distances_bothways, rho_coeffs)
            total_ni[:,i] += np.sum(results, axis=1)

            if len(self.triplet_values[i]) > 0: # check needed in case of dimer

                # Unpack ordered sets of values and indexing information
                values_tuple    = self.triplet_values[i]
                rij_values      = values_tuple[:,0]
                rik_values      = values_tuple[:,1]
                cos_values        = values_tuple[:,2]

                types_tuple = self.triplet_info['pot_type_idx'][i]
                fj_types    = types_tuple[:,0]
                fk_types    = types_tuple[:,1]
                g_types     = types_tuple[:,2]

                intervals_tuple = self.triplet_info['interval_idx'][i]
                fj_intervals    = intervals_tuple[:,0]
                fk_intervals    = intervals_tuple[:,1]
                g_intervals     = intervals_tuple[:,2]

                # Calculate three body contributions
                fj_coeffs = coeffs_from_indices(self.fs,
                            {'pot_type_idx':fj_types, 'interval_idx':fj_intervals})
                fk_coeffs = coeffs_from_indices(self.fs,
                            {'pot_type_idx':fk_types, 'interval_idx':fk_intervals})
                g_coeffs = coeffs_from_indices(self.gs,
                            {'pot_type_idx':g_types, 'interval_idx':g_intervals})

                fj_results = eval_all_polynomials(rij_values, fj_coeffs)
                fk_results = eval_all_polynomials(rik_values, fk_coeffs)
                g_results = eval_all_polynomials(cos_values, g_coeffs)

                results = np.multiply(fj_results,np.multiply(fk_results, g_results))
                total_ni[:,i] += np.sum(results, axis=1)

        # Perform binning and shifting for each ni (per atom) in each potential
        knots = self.us[0][0].knotsx
        for p in range(total_ni.shape[0]):
            ni_indices_for_one_pot = np.zeros(len(self.atoms))

            for q in range(total_ni.shape[1]):
                # Shift according to pre-computed U index info
                u_idx = self.u_index_info['pot_type_idx'][q]

                ni = total_ni[p][q]

                u_spline_num, u_knot_num = intervals_from_splines(
                    self.us,ni,u_idx)

                ni_indices_for_one_pot[q] = u_spline_num
                total_ni[p][q] -= knots[u_knot_num]

            self.u_index_info['interval_idx'].append(ni_indices_for_one_pot)

        self.u_index_info['interval_idx'] = np.array(self.u_index_info[
                                                         'interval_idx'])

        # Compute U values and derivatives; add to energy and forces
        u_types = self.u_index_info['pot_type_idx']

        self.uprimes = np.zeros( (len(potentials),len(self.atoms)) )
        for i,row  in enumerate(total_ni):
            u_indices = self.u_index_info['interval_idx'][i]

            u_coeffs = coeffs_from_indices([self.us[i]],
                        {'pot_type_idx':u_types, 'interval_idx':u_indices})

            results = eval_all_polynomials(row, u_coeffs)
            energies[i] += np.sum(results, axis=1)

            # dots each layer of u_coeffs with the derivative matrix
            uprime_coeffs = np.einsum('ij,jdk->idk', D, u_coeffs)

            results = eval_all_polynomials(row, uprime_coeffs)
            self.uprimes[i,:] = eval_all_polynomials(row, uprime_coeffs)

        # Subtract off zero-point energies
        zero_atom_energy = np.array([self.zero_atom_energies[:,z] for z in
                            u_types]).transpose()
        zero_atom_energy = np.sum(zero_atom_energy, axis=1)

        energies -= zero_atom_energy

        return energies

    def compute_forces(self, potentials):
        """Calculates energies for all potentials using information
        pre-computed during initialization."""
        #TODO: only thing changing should be y-coords?
        # changing y-coords changes splines, changes coeffs
        # maybe could use cubic spline eval equation, not polyval()

        # TODO: should __call__() take in just a single potential?

        # TODO: is it worth creating a huge matrix of ordered coeffs so that
        # it's just a single matrix multiplication? probably yes because
        # Python for-loops are terrible! problem: need to zero pad to make
        # same size

        # TODO: turn coeffs into array of knot y-values

        # TODO: eventually move away from scipy Spline, towards Recipes equation

        # TODO: might as well just merge with compute_energies()?
        self.compute_energies(potentials)

        natoms = len(self.atoms)
        npots = len(potentials)

        # Derivative matrix for cubic coefficients
        D = np.array([[0,1,0,0], [0,0,2,0], [0,0,0,3]])

        self.potentials = potentials
        forces = np.zeros((npots, natoms, 3))

        # Compute phi contribution to total energy
        phi_coeffs = coeffs_from_indices(self.phis, self.phi_index_info)

        i_indices = self.pair_indices_oneway[:,0]
        j_indices = self.pair_indices_oneway[:,1]

        j_directions = self.pair_directions_oneway

        phi_prime_coeffs = np.einsum('ij,jdk->idk', D, phi_coeffs)
        phi_prime_values = eval_all_polynomials(self.pair_distances_oneway,
                                phi_prime_coeffs)
        phi_forces = np.einsum('ij,jk->ijk',phi_prime_values,j_directions)

        # See np.ufunc.at documentation for why this is necessary
        np.add.at(forces, (slice(None), i_indices, slice(None)), phi_forces)
        np.add.at(forces, (slice(None), j_indices, slice(None)), -phi_forces)

        # TODO: switch to full vectorization; tags shouldn't be needed
        # TODO: unless tags actually improve speed?

        # Calculate ni values; ni intervals cannot be pre-computed in init()
        for i in range(natoms):
            itype = lammpsTools.symbol_to_type(self.atoms[i].symbol, self.types)
            logging.info("Atom {0} of type {1}---------------".format(i, itype))
            Uprime_i = self.uprimes[:,i]
            Uprime_i = Uprime_i.reshape( (len(Uprime_i),1) )

            # Compute phi contributions

            # Pull per-atom neighbor distances and compute rho contribution
            pair_distances_bothways = self.pair_distances_bothways[i]
            pair_directions_bothways = self.pair_directions_bothways[i]
            pair_indices_bothways = self.pair_indices_bothways[i]

            i_indices = pair_indices_bothways[:,0]
            j_indices = pair_indices_bothways[:,1]

            # Potential index and atom tags for neighbors
            rho_types_j = self.rho_index_info['pot_type_idx'][i]
            rho_indices_j = self.rho_index_info['interval_idx'][i]

            # Forces from i to neighbors
            rho_coeffs_j = coeffs_from_indices(self.rhos,
                    {'pot_type_idx':rho_types_j, 'interval_idx':rho_indices_j})

            # Rho prime values for neighbors
            rho_prime_coeffs_j = np.einsum('ij,jdk->idk', D, rho_coeffs_j)
            rho_prime_values_j = eval_all_polynomials(pair_distances_bothways,
                                              rho_prime_coeffs_j)

            # Forces from neighbors to i
            itype = lammpsTools.symbol_to_type(self.atoms[i].symbol, self.types)
            rho_idx_i = meam.i_to_potl(itype)

            rho_indices_i = self.rho_index_info['interval_idx_backwards'][i]

            rho_coeffs_i = coeffs_from_indices(self.rhos,
                   {'pot_type_idx':np.ones(len(
                    pair_distances_bothways))*rho_idx_i,
                    'interval_idx':rho_indices_i})

            rho_prime_coeffs_i = np.einsum('ij,jdk->idk', D, rho_coeffs_i)
            rho_prime_values_i = eval_all_polynomials(pair_distances_bothways,
                                              rho_prime_coeffs_i)

            # Combine forwards/backwards forces
            fpair = rho_prime_values_j*Uprime_i
            rho_prime_values_i = np.multiply(rho_prime_values_i, self.uprimes[:,
                                                       j_indices])
            fpair += rho_prime_values_i
            #fpair += phi_prime_values

            # TODO: seem to be double-counting rho terms. hence the /= 2.0
            rho_forces = np.einsum('ij,jk->ijk',fpair, pair_directions_bothways)
            rho_forces /= 2.0

            np.add.at(forces, (slice(None),i_indices, slice(None)), rho_forces)
            np.add.at(forces, (slice(None),j_indices, slice(None)), -rho_forces)

            if len(self.triplet_values[i]) > 0: # check needed in case of dimer

                # Unpack ordered sets of values and indexing information
                values_tuple    = self.triplet_values[i]
                rij_values      = values_tuple[:,0]
                rik_values      = values_tuple[:,1]
                cos_values        = values_tuple[:,2]

                values_tuple    = self.triplet_unshifted[i]
                rij_values_unshifted      = values_tuple[:,0]
                rik_values_unshifted      = values_tuple[:,1]
                cos_values_unshifted        = values_tuple[:,2]

                types_tuple = self.triplet_info['pot_type_idx'][i]
                fj_types    = types_tuple[:,0]
                fk_types    = types_tuple[:,1]
                g_types     = types_tuple[:,2]

                # rzm: norhophi fails for triplets only if not aaa or bbb.
                # type dependent.
                intervals_tuple = self.triplet_info['interval_idx'][i]
                fj_intervals    = intervals_tuple[:,0]
                fk_intervals    = intervals_tuple[:,1]
                g_intervals     = intervals_tuple[:,2]

                # Calculate three body contributions
                fj_coeffs = coeffs_from_indices(self.fs,
                            {'pot_type_idx':fj_types, 'interval_idx':fj_intervals})
                fk_coeffs = coeffs_from_indices(self.fs,
                            {'pot_type_idx':fk_types, 'interval_idx':fk_intervals})
                g_coeffs = coeffs_from_indices(self.gs,
                            {'pot_type_idx':g_types, 'interval_idx':g_intervals})

                fj_results = eval_all_polynomials(rij_values, fj_coeffs)
                fk_results = eval_all_polynomials(rik_values, fk_coeffs)
                g_results = eval_all_polynomials(cos_values, g_coeffs)

                # TODO: consider putting derivative evals in eval_all() 4 speed?
                # Take derivatives and compute forces
                fj_prime_coeffs = np.einsum('ij,jdk->idk', D, fj_coeffs)
                fk_prime_coeffs = np.einsum('ij,jdk->idk', D, fk_coeffs)
                g_prime_coeffs = np.einsum('ij,jdk->idk', D, g_coeffs)

                fj_primes = eval_all_polynomials(rij_values, fj_prime_coeffs)
                fk_primes = eval_all_polynomials(rik_values, fk_prime_coeffs)
                g_primes = eval_all_polynomials(cos_values, g_prime_coeffs)

                #logging.info("{0}, {1}, {2}".format(g_results.shape,
                #                        fk_results.shape, fj_primes.shape))

                fij = -Uprime_i*np.multiply(g_results, np.multiply(
                    fk_results, fj_primes))

                fik = -Uprime_i*np.multiply(g_results, np.multiply(
                    fj_results, fk_primes))

                logging.info("fij = {0}".format(fij))
                logging.info("fik = {0}".format(fik))
                logging.info("cos_values = {0}".format(cos_values_unshifted))
                logging.info("rij_values = {0}".format(rij_values_unshifted))
                logging.info("rik_values = {0}".format(rik_values_unshifted))

                # rzm: rij_values have been shifted by knot positions already?
                prefactor = Uprime_i*np.multiply(fj_results, np.multiply(
                    fk_results, g_primes))
                prefactor_ij = np.divide(prefactor, rij_values_unshifted)
                prefactor_ik = np.divide(prefactor, rik_values_unshifted)

                fij += np.multiply(prefactor_ij, cos_values_unshifted)
                fik += np.multiply(prefactor_ik, cos_values_unshifted)

                logging.info("prefactor = {0}".format(prefactor))
                logging.info("prefactor_ij = {0}".format(prefactor_ij))
                logging.info("prefactor_ik = {0}".format(prefactor_ik))

                jvec = self.triplet_directions[i][:,0]
                kvec = self.triplet_directions[i][:,1]

                forces_j = np.einsum('ij,jk->ijk', fij, jvec)
                forces_j -= np.einsum('ij,jk->ijk', prefactor_ij, kvec)

                forces_k = np.einsum('ij,jk->ijk', fik, kvec)
                forces_k -= np.einsum('ij,jk->ijk', prefactor_ik, jvec)

                j_indices = self.triplet_indices[i][:,0]
                k_indices = self.triplet_indices[i][:,1]

                np.add.at(forces, (slice(None), j_indices, slice(None)),
                        forces_j)
                np.add.at(forces, (slice(None), k_indices, slice(None)),
                        forces_k)

                # logging.info("{0}, {1}: fij, prefactor_ij".format(fij,
                #                                                prefactor_ij))
                # logging.info("{0}, {1}: fik, prefactor_ik".format(fik,
                #                                                   prefactor_ik))
                # logging.info("{0}, {1}, {4}, fj_results, fk_results, "
                #              "g_results, {2},{3} j,k".format(
                #     fj_results, fk_results, fj_types+1, fk_types+1, g_results))

                logging.info("fj_results = {0}".format(fj_results))
                logging.info("fj_primes = {0}".format(fj_primes))
                logging.info("fk_results = {0}".format(fk_results))
                logging.info("fk_primes = {0}".format(fk_primes))
                logging.info("g_results = {0}".format(g_results))
                logging.info("g_primes = {0}".format(g_primes))
                logging.info("forces_j = {0}".format(forces_j))
                logging.info("forces_k = {0}".format(forces_k))

                forces[:,i,:] -= np.sum(forces_k, axis=1)
                forces[:,i,:] -= np.sum(forces_j, axis=1)

                # TODO: forces worker? forces here needs directional info
                # TODO: pass disp vectors, not rij, then vectorize reduction
                # to rij_values?

        #return forces.reshape((npots, natoms, 3))
        return forces

    @property
    def potentials(self):
        """Set of MEAM objects of various potentials"""
        return self._potentials

    @potentials.setter
    def potentials(self, potentials):
        self._potentials = potentials
        self._cutoff = potentials[0].cutoff
        self._types = potentials[0].types
        self._ntypes = len(self.types)

        N = len(potentials)
        self.phis   = [potentials[i].phis for i in range(N)]
        self.rhos   = [potentials[i].rhos for i in range(N)]
        self.us     = [potentials[i].us for i in range(N)]
        self.fs     = [potentials[i].fs for i in range(N)]
        self.gs     = [potentials[i].gs for i in range(N)]

        # Initialize zero-point embedding energies
        self.zero_atom_energies = np.zeros((len(potentials), self.ntypes))
        for i,p in enumerate(potentials):
            for j in range(self.ntypes):
                self.zero_atom_energies[i][j] = p.us[j](0.0)

    # TODO: can we condense oneway/bothways? derive oneway from bothways?
    @property
    def pair_distances_oneway(self):
        return self._pair_distances_oneway

    @pair_distances_oneway.setter
    def pair_distances_oneway(self, R):
        self._pair_distances_oneway = R

    @property
    def pair_distances_bothways(self):
        """2D list where each row is the set of neighbor distances for a
        single atom"""
        return self._pair_distances_bothways

    @pair_distances_bothways.setter
    def pair_distances_bothways(self, R):
        self._pair_distances_bothways = R

    @property
    def phi_index_info(self):
        return self._phi_index_info

    @phi_index_info.setter
    def phi_index_info(self, info):
        self._phi_index_info = info

    @property
    def rho_index_info(self):
        return self._rho_index_info

    @rho_index_info.setter
    def rho_index_info(self, info):
        self._rho_index_info = info

    @property
    def phis(self):
        """NxM list where N = #potentials, M = #spline intervals"""
        return self._phis

    # Individual splines will need to be tunable during optimization
    @phis.setter
    def phis(self, p):
        self._phis = p

    @property
    def rhos(self):
        """NxM list where N = #potentials, M = #spline intervals"""
        return self._rhos

    @rhos.setter
    def rhos(self, p):
        self._rhos = p

    @property
    def us(self):
        """NxM list where N = #potentials, M = #spline intervals"""
        return self._us

    @us.setter
    def us(self, p):
        self._us = p

    @property
    def fs(self):
        """NxM list where N = #potentials, M = #spline intervals"""
        return self._fs

    @fs.setter
    def fs(self, p):
        self._fs = p

    @property
    def gs(self):
        """NxM list where N = #potentials, M = #spline intervals"""
        return self._gs

    @gs.setter
    def gs(self, p):
        self._gs = p

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, c):
        raise AttributeError("cutoff can only be set by setting the 'potentials' property")

    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, c):
        raise AttributeError("types can only be set by setting the 'potentials' property")

    @property
    def ntypes(self):
        return self._ntypes

    @ntypes.setter
    def ntypes(self, n):
        raise AttributeError("ntypes can only be set by setting the 'potentials' property")

    # def compute_energies(self):
    #     atoms = self.atoms
    #     natoms = len(atoms)

        # nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
        #         self_interaction=False, bothways=False, skin=0.0)
        # nl_noboth.build(atoms)

        # nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
        #         self_interaction=False, bothways=True, skin=0.0)
        # nl.build(atoms)

        # # energies[z] corresponds to the energy as calculated by the k-th
        # # potential
        # energies = np.zeros(len(self.potentials))

        # for i in range(natoms):
        #     total_ni = np.zeros(len(self.potentials))

            # itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            # ipos = atoms[i].position

            # u_idx = meam.i_to_potl(itype)

            # neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            # neighbors, offsets = nl.get_neighbors(i)

#             # TODO: requires knowledge of ni to get interval
#             #u_coeffs = np.array([us[potnum][idx].cmat[:,]])

            # # Calculate pair interactions (phi)
            # for j,offset in zip(neighbors_noboth,offsets_noboth):
            #     jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
            #     jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                # rij = np.linalg.norm(ipos -jpos)

                # # Finds correct type of phi fxn
                # phi_idx = meam.ij_to_potl(itype,jtype,self.ntypes)

                # phi_coeffs, spline_num, phi_knot_num = coeffs_from_splines(
                #     self.phis,rij,phi_idx)

                # # knot index != cmat index
                # rij -= self.phis[0][phi_idx].knotsx[phi_knot_num]

                # energies += polyval(rij, phi_coeffs)

#                 # TODO: is it actually faster to create matrix of coefficients,
#                 # then use np.polynomial.polynomial.polyval() (Horner's) than to just use
#                 # CubicSpline.__call__() (unknown method; from LAPACK)?

            # # Calculate three-body contributions
            # j_counter = 1 # for tracking neighbor
            # for j,offset in zip(neighbors,offsets):

                # jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                # jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                # a = jpos - ipos
                # na = np.linalg.norm(a)

                # rij = np.linalg.norm(ipos -jpos)

                # rho_idx = meam.i_to_potl(jtype)
                # fj_idx = meam.i_to_potl(jtype)

                # rho_coeffs, rho_spline_num, rho_knot_num = coeffs_from_splines(
                #     self.rhos,rij,rho_idx)
                # fj_coeffs, fj_spline_num, fj_knot_num = coeffs_from_splines(
                #     self.fs,rij,fj_idx)
                # # assumes rho and f knots are in same positions
                # rijo = rij
                # rij = rij - self.rhos[0][rho_idx].knotsx[rho_knot_num]

                # #logging.info("expected = {2},rho_val = {0}, type = {1}".format(
                # #    polyval(rij,rho_coeffs), rho_idx, self.rhos[0][rho_idx](
                # #        rijo)))
                # total_ni += polyval(rij, rho_coeffs)# + self.rhos[0][
                # # 0].knotsy[rho_spline_num-1]

                # rij += self.rhos[0][rho_idx].knotsx[rho_knot_num]

                # # Three-body contributions
                # partialsum = np.zeros(len(self.potentials))
                # for k,offset in zip(neighbors[j_counter:], offsets[j_counter:]):
                #     if k != j:
                #         ktype = lammpsTools.symbol_to_type(atoms[k].symbol,
                #                                            self.types)

#                         #logging.info("{0}, {1}, {2}".format(itype,jtype,ktype))

                        # kpos = atoms[k].position + np.dot(offset,
                        #                                   atoms.get_cell())
                        # rik = np.linalg.norm(ipos-kpos)

                        # b = kpos - ipos
                        # nb = np.linalg.norm(b)

                        # cos_theta = np.dot(a,b)/na/nb

                        # fk_idx = meam.i_to_potl(ktype)
                        # g_idx = meam.ij_to_potl(jtype, ktype, self.ntypes)

                        # fk_coeffs, fk_spline_num, fk_knot_num = \
                        #     coeffs_from_splines(self.fs,rik,fk_idx)
                        # g_coeffs, g_spline_num, g_knot_num = \
                        #     coeffs_from_splines(self.gs,cos_theta, g_idx)

                        # rik -= self.fs[0][fk_idx].knotsx[fk_knot_num]
                        # cos_theta -= self.gs[0][g_idx].knotsx[g_knot_num]


#                         #fk_val = polyval(rik, fk_coeffs)
#                         #g_val = polyval(cos_theta, g_coeffs)

                        # #logging.info('fk_val = {0}'.format(fk_val))
                        # #logging.info('g_val = {0}'.format(g_val))
                        # partialsum += polyval(rik, fk_coeffs)*polyval(
                        #     cos_theta, g_coeffs)

                # j_counter += 1
                # rij -= self.fs[0][fj_idx].knotsx[fj_knot_num]
                # total_ni += polyval(rij, fj_coeffs)*partialsum

            # # Build U coefficient matrix
            # u_coeffs = np.zeros((4,len(self.potentials)))
            # for l,p in enumerate(self.potentials):
            #     knots = p.us[u_idx].knotsx

                # h = p.us[u_idx].h

                # top = total_ni[l] - knots[0]
                # tmp1 = top/h
                # tmp2 = np.floor(tmp1)
                # tmp = int(tmp2)
                # u_spline_num = tmp + 1
                # #u_spline_num = int(np.floor((total_ni[l]-knots[0])/h)) + 1

                # if u_spline_num <= 0:
                #     #str = "EXTRAP: total_ni = {0}".format(total_ni)
                #     #logging.info(str)
                #     u_spline_num = 0
                #     u_knot_num = 0
                # elif u_spline_num > len(knots):
                #     #str = "EXTRAP: total_ni = {0}".format(total_ni)
                #     #logging.info(str)
                #     u_spline_num = len(knots)
                #     u_knot_num = u_spline_num - 1
                # else:
                #     u_knot_num = u_spline_num - 1

                # #logging.info("expected = {0}, total_ni = {1}, knot0 = {"
                # #             "2}".format(p.us[u_idx](total_ni), total_ni,
                # #                         knots[u_spline_num-1]))
                # u_coeffs[:,l] = p.us[u_idx].cmat[:,u_spline_num]
                # total_ni[l] -= knots[u_knot_num]

            # u_coeffs = u_coeffs[::-1]

            # #logging.info("real = {0}, total_ni = {1}".format(polyval(total_ni,
            # #    u_coeffs,tensor=False)-self.zero_atom_energies[:,u_idx],
            # #                                                 total_ni))
            # energies += polyval(total_ni, u_coeffs,tensor=False) -\
            #             self.zero_atom_energies[:,u_idx]

        # return energies

def intervals_from_splines(splines, x, pot_type):
    """Extracts the interval corresponding to a given value of x. Assumes
    linear extrapolation outside of knots and fixed knot positions.

    Args:
        splines (list):
            a 2D list of splines where each row corresponds to a unique MEAM
            potential, and each column is a spline interval
        x (float):
            the point used to find the spline interval
        pot_type (int):
            index identifying the potential type, ordered as seen in meam.py

    Returns:
        spline_num (int):
            index of spline interval
        knot_num (int):
            knot index used for value shifting; LHS knot for internal"""

    # TODO: intervals_from_splines() should only need knot x-positions
    knots = splines[0][pot_type].knotsx

    h = splines[0][pot_type].h

    # Find spline interval; +1 to account for extrapolation
    # TODO: '//' operator is floor
    spline_num = int(np.floor((x-knots[0])/h)) + 1 # shift x by leftmost knot

    if spline_num <= 0:
        spline_num = 0
        knot_num = 0
    elif spline_num > len(knots):
        spline_num = len(knots)
        knot_num = spline_num - 1
    else:
        knot_num = spline_num - 1

    return spline_num, knot_num

def coeffs_from_indices(splines, index_info):
    """Extracts the coefficient matrix for the given potential type and
    interval number.

    Args:
        splines (list):
            a 2D list of splines where each row corresponds to a unique MEAM
            potential, and each column is a spline interval
        index_info (dict):
            dictionary of index values. keys = 'pot_type_idx', 'interval_idx'
    Returns:
        coeffs (np.arr):
            (4 x Nelements x Npots) array of coefficients where coeffs[i][j] is
            the i-th coefficient of the N-th potential. Formatted to work with
            np.polynomial.polynomial.polyval(). Nelements can be pair
            distances, angles, ni values, etc."""

    pot_types = np.array(index_info['pot_type_idx'])
    intervals  = np.array(index_info['interval_idx'])
    n = len(pot_types)

    splines = np.array(splines)

    coeffs = np.zeros((4, n, len(splines)))

    for k in range(n):
        pot_type_idx = int(pot_types[k])
        interval_idx = int(intervals[k])

        part = np.array([splines[i][pot_type_idx].cmat[:,interval_idx] for i in
                       range(len(splines))])

        # Adjust to match polyval() ordering
        part = part.transpose()
        coeffs[:,k,:] = part[::-1]

    return coeffs

def coeffs_from_splines(splines, x, pot_type):
    """Extracts the coefficient matrix for an interval corresponding to a
    given value of x. Assumes linear extrapolation outside of knots and fixed
    knot positions.

    Args:
        splines (list):
            a 2D list of splines where each row corresponds to a unique MEAM
            potential, and each column is a spline interval
        x (float):
            the point used to find the spline interval
        pot_type (int):
            index identifying the potential type, ordered as seen in meam.py

    Returns:
        coeffs (np.arr):
            4xN array of coefficients where coeffs[i][j] is the i-th
            coefficient of the N-th potential. Formatted to work with
            np.polynomial.polynomial.polyval()
        spline_num (int):
            index of spline interval
        knot_num (int):
            knot index used for value shifting; LHS knot for internal
            intervals, endpoint for extrapolation intervals"""

    knots = splines[0][pot_type].knotsx

    h = splines[0][pot_type].h

    # Find spline interval; +1 to account for extrapolation
    # TODO: '//' operator is floor
    spline_num = int(np.floor((x-knots[0])/h)) + 1 # shift x by leftmost knot

    if spline_num <= 0:
        spline_num = 0
        knot_num = 0
    elif spline_num > len(knots):
        spline_num = len(knots)
        knot_num = spline_num - 1
    else:
        knot_num = spline_num - 1

   # for i in range(len(splines)):
   #     if spline_num >= splines[i][pot_type].cmat.shape[1]:
   #         logging.info('{0} {1}'.format(pot_type, spline_num))

    # Pull coefficients from Spline.cmat variable
    #logging.info('{0} {1}'.format(pot_type, spline_num))
    coeffs = np.array([splines[i][pot_type].cmat[:,spline_num] for i in
                       range(len(splines))])

    # Adjust to match polyval() ordering
    coeffs = coeffs.transpose()
    coeffs = coeffs[::-1]

    return coeffs, spline_num, knot_num

def eval_all_polynomials(x, coeffs):
    """Computes polynomial values for a 3D matrix of coefficients at all
    points in x. coeffs[:,k,:] contains the coefficients for N polynomials,
    all of which will be evaluated for p(x[k]).

    Args:
        x (np.arr):
            the points at which to evaluate the polynomials
        coeffs (np.arr):
            (l x m x n) matrix where l is the degree of the polynomials,
            m is the number of points being evaluated (len(x)), and n is the
            number of polynomials point

    Returns:
        results (np.arr):
            (m x n) matrix where results[i,j] is the value of the polynomial
            represented by coeffs[:,i,j] evaluated at point x[i]"""

    results = np.array([polyval(x, coeffs[:,:,k], tensor=False) for k in
                        range(coeffs.shape[2])])
    return results

def update_forces(original, update, i):
    """Updates the forces of the i-th atom."""

    original[:,i,:] += update
