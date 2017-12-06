import numpy as np
import lammpsTools
import meam
import logging

from numpy.polynomial.polynomial import polyval
from ase.neighborlist import NeighborList

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logging.disable(logging.CRITICAL)

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


    # TODO: if this turns out to have been a really good idea, just let the
    # record show that it was my first good idea in grad school, and it occured
    # at approximately 5pm on 12/1/17 (Fri)

    # - init() takes in structure and at least one potential
    # - computes all structure properties (rij, rik, costheta)
    # - precomputes spline stuff (ABCD)
    # - stores/returns the splines that are needed for evaluation
    # ---- ids: phi_Ti, U_O,... ordered to match each other
    # you'd think they'd all be updated at once, but recall the idea of only
    # calculating a dE.
    # ---- so could pass in the knoty values for a sinle relevant spline (e.g.
    # phi_Ti), then throw back into phi_eval([list of knotys])
    # ---- need a tag for which spline you're updating
    # ---- then you can update only the relevant parts; e.g. re-compute phi only
    # for the rij of that type
    # - store: bond types, bond lengths, triplet angles; ordered
    # - each bond/triplet has its own force/energy contrib.
    # - could just be four arrays (one for force/eng for each bond/triplet)
    # - the main benefit is that if you change a particular spline, you only
    # have to update one of the phi/u/rho/f/g of the bonds/triplets that are affected

    # rzm: refactoring to represent potentials as 4D numpy arrays (spline,
    # knot, x, y); 'potentials' is 5D now
    # TODO: refactor potentials.py to build these 5D arrays
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

        # Condensed structure information (not using Spline objects)
        self.phi_abcd               = [] # 3D, ABCD for each pot for each dist
        self.phi_intervals          = [] # 1D, spline interval for distances
        self.phi_types              = [] # 1D, atom types for each distance

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

            u_idx = meam.i_to_potl(itype, 'u', self.ntypes)
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

                # TODO: ij_to_potl, i_to_potl should take 'phi', 'rho', etc.
                # TODO: zero_atom_energies needs to be size of all pots
                phi_idx = meam.ij_to_potl(itype, jtype, 'phi', self.ntypes)

                phi_interval_num, phi_knot_num = intervals_from_splines(
                    self.phis,rij,phi_idx)

                self.phi_index_info['pot_type_idx'].append(phi_idx)
                self.phi_index_info['interval_idx'].append(phi_interval_num)

                # knot index != cmat index since cmat has extrapolation splines
                x = self.phis[0][phi_idx].knotsx

                rij -= x[phi_knot_num]
                self.pair_distances_oneway.append(rij)

                xj1 = x[phi_interval_num+1]
                xj  = x[phi_interval_num]

                A = (xj1-rij) / (xj1-xj)
                B = 1-A
                C = (1/6.)*(A**3 - A)*(xj1-xj)**2
                D = (1/6.)*(B**3 - B)*(xj1-xj)**2

                self.phi_abcd.append(np.array([A,B,C,D]))
                self.phi_intervals.append(phi_interval_num)
                self.phi_types.append(phi_idx)

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
                rho_idx = meam.i_to_potl(jtype, 'rho', self.ntypes)

                rho_interval_num, rho_knot_num = intervals_from_splines(
                    self.rhos,rij,rho_idx)
                rho_type_indices_for_one_atom.append(rho_idx)
                rho_spline_indices_for_one_atom_forwards.append(rho_interval_num)

                rho_idx_i = meam.i_to_potl(itype, 'rho', self.ntypes)
                rho_interval_num_i,_ = intervals_from_splines(
                    self.rhos,rij,rho_idx_i)
                rho_spline_indices_for_one_atom_backwards.append(
                    rho_interval_num_i)

                # fj information
                fj_idx = meam.i_to_potl(jtype, 'f', self.ntypes)

                fj_interval_num, fj_knot_num = intervals_from_splines(
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
                        fk_idx = meam.i_to_potl(ktype, 'f', self.ntypes)

                        fk_interval_num, fk_knot_num = \
                            intervals_from_splines(self.fs,rik,fk_idx)

                        # g information
                        g_idx = meam.ij_to_potl(jtype, ktype, 'g', self.ntypes)

                        g_interval_num, g_knot_num = \
                            intervals_from_splines(self.gs,cos_theta, g_idx)

                        rik -= self.fs[0][fk_idx].knotsx[fk_knot_num]
                        cos_theta -= self.gs[0][g_idx].knotsx[g_knot_num]

                        # Organize triplet information
                        tup1 = np.array([rij, rik, cos_theta])
                        tup2 = np.array([fj_idx, fk_idx, g_idx])
                        tup3 = np.array([fj_interval_num, fk_interval_num,
                                         g_interval_num])

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

        self.phi_abcd = np.array(self.phi_abcd)
        self.phi_interval = np.array(self.phi_intervals)

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

        self.potentials = potentials
        energies = np.zeros(len(potentials))

        natoms = len(self.atoms)
        npots = len(self.potentials)

        # Compute phi contribution to total energy
        # phi_coeffs = coeffs_from_indices(self.phis, self.phi_index_info)
        # results = eval_all_polynomials(self.pair_distances_oneway, phi_coeffs)
        # energies += np.sum(results, axis=1)

        # rzm: phis[i] needs to be phis[i][phi_idx]; do in init()?
        phi_types = list(self.phi_types)
        a = self.phis[0]
        b = a[phi_types]
        phi_ys =np.array([self.phis[i][phi_types].knotsy for i in range(npots)])
        phi_xs =np.array([self.phis[i][phi_types].knotsx for i in range(npots)])
        phi_d0 =np.array([self.phis[i][phi_types].d0 for i in range(npots)])
        phi_dN =np.array([self.phis[i][phi_types].dN for i in range(npots)])

        phi_y2s = second_derivatives(phi_xs, phi_ys, phi_d0, phi_dN)

        phi_abcd = self.phi_abcd
        phi_ymat = get_ymat(phi_ys, phi_y2s, self.phi_intervals)
        phi_results = np.einsum('ij,ijk->k', phi_abcd, phi_ymat)

        energies += phi_results

        # Calculate ni values; ni intervals cannot be pre-computed in init()
        total_ni = np.zeros( (len(potentials),natoms) )

        for i in range(natoms):

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
            ni_indices_for_one_pot = np.zeros(natoms)

            for q in range(total_ni.shape[1]):
                # Shift according to pre-computed U index info
                u_idx = self.u_index_info['pot_type_idx'][q]

                ni = total_ni[p][q]

                u_interval_num, u_knot_num = intervals_from_splines(
                    self.us,ni,u_idx)

                ni_indices_for_one_pot[q] = u_interval_num
                total_ni[p][q] -= knots[u_knot_num]

            self.u_index_info['interval_idx'].append(ni_indices_for_one_pot)

        self.u_index_info['interval_idx'] = np.array(self.u_index_info[
                                                         'interval_idx'])

        # Compute U values and derivatives; add to energy and forces
        u_types = self.u_index_info['pot_type_idx']

        self.uprimes = np.zeros( (len(potentials),natoms) )
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
            # logging.info("Atom {0} of type {1}---------------".format(i, itype))
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
            rho_idx_i = meam.i_to_potl(itype, 'rho', self.ntypes)

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

                # logging.info("fij = {0}".format(fij))
                # logging.info("fik = {0}".format(fik))
                # logging.info("cos_values = {0}".format(cos_values_unshifted))
                # logging.info("rij_values = {0}".format(rij_values_unshifted))
                # logging.info("rik_values = {0}".format(rik_values_unshifted))

                prefactor = Uprime_i*np.multiply(fj_results, np.multiply(
                    fk_results, g_primes))
                prefactor_ij = np.divide(prefactor, rij_values_unshifted)
                prefactor_ik = np.divide(prefactor, rik_values_unshifted)

                fij += np.multiply(prefactor_ij, cos_values_unshifted)
                fik += np.multiply(prefactor_ik, cos_values_unshifted)

                # logging.info("prefactor = {0}".format(prefactor))
                # logging.info("prefactor_ij = {0}".format(prefactor_ij))
                # logging.info("prefactor_ik = {0}".format(prefactor_ik))

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

                # logging.info("fj_results = {0}".format(fj_results))
                # logging.info("fj_primes = {0}".format(fj_primes))
                # logging.info("fk_results = {0}".format(fk_results))
                # logging.info("fk_primes = {0}".format(fk_primes))
                # logging.info("g_results = {0}".format(g_results))
                # logging.info("g_primes = {0}".format(g_primes))
                # logging.info("forces_j = {0}".format(forces_j))
                # logging.info("forces_k = {0}".format(forces_k))

                forces[:,i,:] -= np.sum(forces_k, axis=1)
                forces[:,i,:] -= np.sum(forces_j, axis=1)

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
        self._nphi = int((self.ntypes+1)*self.ntypes/2)

        # N = len(potentials)
        # self.phis   = [potentials[i].phis for i in range(N)]
        # self.rhos   = [potentials[i].rhos for i in range(N)]
        # self.us     = [potentials[i].us for i in range(N)]
        # self.fs     = [potentials[i].fs for i in range(N)]
        # self.gs     = [potentials[i].gs for i in range(N)]

        # Initialize zero-point embedding energies
        u_shift = nphi + 2
        self.zero_atom_energies = np.zeros((len(potentials), self.ntypes))
        for i,p in enumerate(potentials):
            for j in range(self.ntypes):
                self.zero_atom_energies[i][j] = potentials[j+u_shift](0.0)

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

    @property
    def nphi(self):
        """Used for indexing splines. Splines for a single potential are
        ordered [phis, rhos, us, fs, gs] where there are nphi phis,
        ntype rhos, ntype us, ntype fs, and nphi gs."""
        return self._nphi

    @nphi.setter
    def nphi(self, c):
        self._nphi = c

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
        interval_num (int):
            index of spline interval
        knot_num (int):
            knot index used for value shifting; LHS knot for internal"""

    # TODO: intervals_from_splines() should only need knot x-positions
    knots = splines[0][pot_type].knotsx

    h = splines[0][pot_type].h

    # Find spline interval; +1 to account for extrapolation
    interval_num = int(np.floor((x-knots[0])/h)) + 1 # shift x by leftmost knot

    if interval_num <= 0:
        interval_num = 0
        knot_num = 0
    elif interval_num > len(knots):
        interval_num = len(knots)
        knot_num = interval_num - 1
    else:
        knot_num = interval_num - 1

    return interval_num, knot_num

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
        interval_num (int):
            index of spline interval
        knot_num (int):
            knot index used for value shifting; LHS knot for internal
            intervals, endpoint for extrapolation intervals"""

    knots = splines[0][pot_type].knotsx

    h = splines[0][pot_type].h

    # Find spline interval; +1 to account for extrapolation
    interval_num = int(np.floor((x-knots[0])/h)) + 1 # shift x by leftmost knot

    if interval_num <= 0:
        interval_num = 0
        knot_num = 0
    elif interval_num > len(knots):
        interval_num = len(knots)
        knot_num = interval_num - 1
    else:
        knot_num = interval_num - 1

   # for i in range(len(splines)):
   #     if interval_num >= splines[i][pot_type].cmat.shape[1]:
   #         logging.info('{0} {1}'.format(pot_type, interval_num))

    # Pull coefficients from Spline.cmat variable
    #logging.info('{0} {1}'.format(pot_type, interval_num))
    coeffs = np.array([splines[i][pot_type].cmat[:,interval_num] for i in
                       range(len(splines))])

    # Adjust to match polyval() ordering
    coeffs = coeffs.transpose()
    coeffs = coeffs[::-1]

    return coeffs, interval_num, knot_num

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

def second_derivatives(x, y, d0, dN, boundary=None):
    """Computes the second derivatives of the interpolating function at the
    knot points. Follows the algorithm in the classic Numerical Recipes
    textbook.

    Args:
        x (np.arr):
            (P,N) array, knot x-positions
        y (np.arr):
            (P,N) array, knot y-positions
        d0 (np.arr):
            length P array of first derivatives at first knot
        dN (np.arr):
            length P array of first derivatives at last knot
        boundary (str):
            the type of boundary conditions to use. 'natural', 'fixed', ...

    Returns:
        y2 (np.arr):
            second derivative at knots"""

    P = x.shape[0] # number of potentials
    N = x.shape[1] # number of knots

    y2 = np.zeros((P,N))    # second derivative values
    u = np.zeros((P,N))     # placeholder for solving system of equations

    if boundary == 'natural':
        y2[:,0] = u[:,0] = 0
    else:
        y2[:,0] = -0.5
        u[:,0] = (3.0 / (x[:,1]-x[:,0])) * ((y[:,1]-y[:,0]) / (x[:,1]-x[:,0]) - d0[i])

    for i in range(1, N-1):
        sig = (x[:,i]-x[:,i-1]) / (x[:,i+1]-x[:,i-1])
        p = sig*y2[:,i-1] + 2.0

        y2[:,i] = (sig - 1.0) / p
        u[:,i] = (y[:,i+1]-y[:,i]) / (x[:,i+1]-x[:,i]) - (y[:,i]-y[:,i-1]) / (x[:,i]-x[:,i-1])
        u[:,i] = (6.0*u[:,i] / (x[:,i+1]-x[:,i-1]) - sig*u[:,i-1])/p

    qn = 0.5
    un = (3.0/(x[:,N-1]-x[:,N-2])) * (dN[i] - (y[:,N-1]-y[:,N-2])/(x[:,N-1]-x[:,N-2]))
    y2[:,N-1] = (un - qn*u[:,N-2]) / (qn*y2[:,N-2] + 1.0)

    for k in range(N-2, -1, -1): # loops over [:,N-2, 0] inclusive
        y2[:,k] = y2[:,k]*y2[:,k+1] + u[:,k]

    # TODO: use solve_banded to quickly do tridiag system
    # TODO: fix for y multi-dimensional (multiple potentials at once)
    return y2

def second_derivatives2(x, y, d0, dN, boundary=None):
    """Computes the second derivatives of the interpolating function at the
    knot points. Follows the algorithm in the classic Numerical Recipes
    textbook.

    Args:
        x (np.arr):
            knot x-positions
        y (np.arr):
            knot y-positions
        d0 (float):
            first derivative at first knot
        dN (float):
            first derivative at last knot
        boundary (str):
            the type of boundary conditions to use. 'natural', 'fixed', ...

    Returns:
        y2 (np.arr):
            second derivative at knots"""

    N = len(x)

    A = diags([1,4,1], [-1,0,1], shape=(N,N))
    A = A.tocsr()
    A[0,0] = A[N-1,N-1] = 2
    A = A.todia()

    tmp1 = np.append(y[1:], y[-1])
    tmp2 = np.insert(y[:-1], 0, y[0])
    b = 3*(tmp1 - tmp2)

    s = solve_banded((1,1), A.data[::-1], b)
    print(s.shape)

    y2 = s
    print(y2)
    #y2[1:N-1] = s

    y2[0] = -0.5

    # qn = 0.5
    # un = (3.0/(x[N-1]-x[N-2])) * (dN - (y[N-1]-y[N-2])/(x[N-1]-x[N-2]))
    # y2[N-1] = (un - qn*u[N-2]) / (qn*y2[N-2] + 1.0)
    # print(y2)

    # TODO: finish this for faster 2nd derivative calculations

    return y2

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
