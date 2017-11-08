import numpy as np
import lammpsTools
import meam
import logging

from numpy.polynomial.polynomial import polyval
from ase.neighborlist import NeighborList

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    # TODO: should __call__() take in just a single potential?
    # TODO: an assumption is made that all potentials have the same cutoffs

    def __init__(self, atoms, potentials):

        # Basic variable initialization
        super(Worker,self).__init__()
        self.atoms = atoms
        self.potentials = potentials    # also sets self.cutoff

        # Building pair distances for phi calculations
        natoms = len(atoms)

        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=False, skin=0.0)
        nl_noboth.build(atoms)

        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=True, skin=0.0)
        nl.build(atoms)

        self.pair_distances_oneway  = [] #TODO: is it faster to build np.arr of
        # desired len?
        self.pair_distances_bothways = []

        # 'index_info' vars are lists of tuples (pot_type_idx, interval_idx)
        self.phi_index_info = {'pot_type_idx':[], 'interval_idx':[]}
        self.rho_index_info = {'pot_type_idx':[], 'interval_idx':[]}
        self.u_index_info   = {'pot_type_idx':[], 'interval_idx':[]}
        #total_num_neighbors = sum([len(nl_noboth.get_neighbors(i) for i in
        #                               range(len(atoms)))])
        for i in range(natoms):
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            u_idx = meam.i_to_potl(itype)
            self.u_index_info['pot_type_idx'].append(u_idx)

            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            for j,offset in zip(neighbors_noboth,offsets_noboth):
                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                rij = np.linalg.norm(ipos -jpos)

                # Finds correct type of phi fxn
                phi_idx = meam.ij_to_potl(itype,jtype,self.ntypes)

                phi_spline_num, phi_knot_num = intervals_from_splines(
                    self.phis,rij,phi_idx)

                self.phi_index_info['pot_type_idx'].append(phi_idx)
                self.phi_index_info['interval_idx'].append(phi_spline_num)

                # knot index != cmat index
                rij -= self.phis[0][phi_idx].knotsx[phi_knot_num]
                self.pair_distances_oneway.append(rij)
                #self.pair_distances_bothways.append(rij)

            pair_distances_for_one_atom = []
            rho_type_indices_for_one_atom = []
            rho_spline_indices_for_one_atom = []

            # Calculate three-body contributions
            j_counter = 1 # for tracking neighbor
            for j,offset in zip(neighbors,offsets):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                a = jpos - ipos
                na = np.linalg.norm(a)

                rij = np.linalg.norm(ipos -jpos)

                rho_idx = meam.i_to_potl(jtype)
                fj_idx = meam.i_to_potl(jtype)

                rho_spline_num, rho_knot_num = intervals_from_splines(
                    self.rhos,rij,rho_idx)
                fj_coeffs, fj_spline_num, fj_knot_num = coeffs_from_splines(
                    self.fs,rij,fj_idx)

                rho_type_indices_for_one_atom.append(rho_idx)
                rho_spline_indices_for_one_atom.append(rho_spline_num)

                # assumes rho and f knots are in same positions
                rij = rij - self.rhos[0][rho_idx].knotsx[rho_knot_num]
                pair_distances_for_one_atom.append(rij)

                # TODO: add this back to pair_distances_bothways later?
                rij += self.rhos[0][rho_idx].knotsx[rho_knot_num]

                # Three-body contributions
                partialsum = np.zeros(len(self.potentials))
                for k,offset in zip(neighbors[j_counter:], offsets[j_counter:]):
                    pass

            self.pair_distances_bothways.append(
                np.array(pair_distances_for_one_atom))

            self.rho_index_info['pot_type_idx'].append(
                np.array(rho_type_indices_for_one_atom))

            self.rho_index_info['interval_idx'].append(
                rho_spline_indices_for_one_atom)

        self.pair_distances_bothways =np.array(self.pair_distances_bothways)

    def __call__(self, potentials):
        #TODO: only thing changing should be y-coords?
        # changing y-coords changes splines, changes coeffs
        # maybe could use cubic spline eval equation, not polyval()

        self.potentials = potentials

        energies = np.zeros(len(potentials))
        #total_ni = np.zeros((len(self.u_index_info['pot_type_idx']),
        #                   len(potentials)))

        # Compute phi contribution to total energy
        phi_coeffs = coeffs_from_indices(self.phis, self.phi_index_info)
        results = eval_all_potentials(self.pair_distances_oneway, phi_coeffs)
        energies += np.sum(results, axis=1)

        # Compute 3-body contributions
        total_ni = np.zeros( (len(potentials),len(self.atoms)) )

        for i in range(len(self.atoms)):

            # TODO: rho_indices are atom_i dependent
            pair_distances_bothways = self.pair_distances_bothways[i]
            rho_types = self.rho_index_info['pot_type_idx'][i]
            rho_indices = self.rho_index_info['interval_idx'][i]

            rho_coeffs = coeffs_from_indices(self.rhos,
                        {'pot_type_idx':rho_types, 'interval_idx':rho_indices})

            results = eval_all_potentials(pair_distances_bothways, rho_coeffs)

            total_ni[:,i] += np.sum(results, axis=1)

        for p in range(total_ni.shape[0]):
            ni_indices_for_one_pot = np.zeros(len(self.atoms))

            for q in range(total_ni.shape[1]):
                u_idx = self.u_index_info['pot_type_idx'][q]
                knots = self.us[0][u_idx].knotsx

                ni = total_ni[p][q]

                u_spline_num, u_knot_num = intervals_from_splines(
                    self.us,ni,u_idx)

                ni_indices_for_one_pot[q] = u_spline_num
                total_ni[p][q] -= knots[u_knot_num]

            self.u_index_info['interval_idx'].append(ni_indices_for_one_pot)

        self.u_index_info['interval_idx'] = np.array(self.u_index_info[
                                                         'interval_idx'])

        results = np.zeros(len(potentials))

        # rzm: need to properly subtract zero_atom_energy
        u_types = self.u_index_info['pot_type_idx']
        zero_atom_energy = np.array([self.zero_atom_energies[:,z] for z in
                            u_types])
        # len(self.atoms)))
        for j,row  in enumerate(total_ni):
            u_indices = self.u_index_info['interval_idx'][j]

            u_coeffs = coeffs_from_indices(self.us,
                        {'pot_type_idx':u_types, 'interval_idx':u_indices})

            results = eval_all_potentials(row, u_coeffs)
            #logging.info("{0}".format(results))
            #logging.info("{0}".format(zero_atom_energy))
            # TODO: need to subtract zero_atom_energies
            energies += np.sum(results, axis=1)
            energies -= np.sum(zero_atom_energy, axis=0)

        #logging.info("real = {0}, total_ni = {1}".format(polyval(total_ni,
        #    u_coeffs,tensor=False)-self.zero_atom_energies[:,u_idx],
        #                                                 total_ni))
        #energies += np.sum(results, axis=1)

        return energies

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

    def compute_energies(self):
        atoms = self.atoms
        natoms = len(atoms)

        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=False, skin=0.0)
        nl_noboth.build(atoms)

        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=True, skin=0.0)
        nl.build(atoms)

        # energies[z] corresponds to the energy as calculated by the k-th
        # potential
        energies = np.zeros(len(self.potentials))

        for i in range(natoms):
            total_ni = np.zeros(len(self.potentials))

            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            u_idx = meam.i_to_potl(itype)

            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # TODO: requires knowledge of ni to get interval
            #u_coeffs = np.array([us[potnum][idx].cmat[:,]])

            # Calculate pair interactions (phi)
            for j,offset in zip(neighbors_noboth,offsets_noboth):
                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                rij = np.linalg.norm(ipos -jpos)

                # Finds correct type of phi fxn
                phi_idx = meam.ij_to_potl(itype,jtype,self.ntypes)

                phi_coeffs, spline_num, phi_knot_num = coeffs_from_splines(
                    self.phis,rij,phi_idx)

                # knot index != cmat index
                rij -= self.phis[0][phi_idx].knotsx[phi_knot_num]

                energies += polyval(rij, phi_coeffs)

                # TODO: is it actually faster to create matrix of coefficients,
                # then use np.polynomial.polynomial.polyval() (Horner's) than to just use
                # CubicSpline.__call__() (unknown method; from LAPACK)?

            # Calculate three-body contributions
            j_counter = 1 # for tracking neighbor
            for j,offset in zip(neighbors,offsets):

                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                a = jpos - ipos
                na = np.linalg.norm(a)

                rij = np.linalg.norm(ipos -jpos)

                rho_idx = meam.i_to_potl(jtype)
                fj_idx = meam.i_to_potl(jtype)

                rho_coeffs, rho_spline_num, rho_knot_num = coeffs_from_splines(
                    self.rhos,rij,rho_idx)
                fj_coeffs, fj_spline_num, fj_knot_num = coeffs_from_splines(
                    self.fs,rij,fj_idx)
                # assumes rho and f knots are in same positions
                rijo = rij
                rij = rij - self.rhos[0][rho_idx].knotsx[rho_knot_num]

                #logging.info("expected = {2},rho_val = {0}, type = {1}".format(
                #    polyval(rij,rho_coeffs), rho_idx, self.rhos[0][rho_idx](
                #        rijo)))
                total_ni += polyval(rij, rho_coeffs)# + self.rhos[0][
                # 0].knotsy[rho_spline_num-1]

                rij += self.rhos[0][rho_idx].knotsx[rho_knot_num]

                # Three-body contributions
                partialsum = np.zeros(len(self.potentials))
                for k,offset in zip(neighbors[j_counter:], offsets[j_counter:]):
                    if k != j:
                        ktype = lammpsTools.symbol_to_type(atoms[k].symbol,
                                                           self.types)

                        #logging.info("{0}, {1}, {2}".format(itype,jtype,ktype))

                        kpos = atoms[k].position + np.dot(offset,
                                                          atoms.get_cell())
                        rik = np.linalg.norm(ipos-kpos)

                        b = kpos - ipos
                        nb = np.linalg.norm(b)

                        cos_theta = np.dot(a,b)/na/nb

                        fk_idx = meam.i_to_potl(ktype)
                        g_idx = meam.ij_to_potl(jtype, ktype, self.ntypes)

                        fk_coeffs, fk_spline_num, fk_knot_num = \
                            coeffs_from_splines(self.fs,rik,fk_idx)
                        g_coeffs, g_spline_num, g_knot_num = \
                            coeffs_from_splines(self.gs,cos_theta, g_idx)

                        rik -= self.fs[0][fk_idx].knotsx[fk_knot_num]
                        cos_theta -= self.gs[0][g_idx].knotsx[g_knot_num]


                        #fk_val = polyval(rik, fk_coeffs)
                        #g_val = polyval(cos_theta, g_coeffs)

                        #logging.info('fk_val = {0}'.format(fk_val))
                        #logging.info('g_val = {0}'.format(g_val))
                        partialsum += polyval(rik, fk_coeffs)*polyval(
                            cos_theta, g_coeffs)

                j_counter += 1
                rij -= self.fs[0][fj_idx].knotsx[fj_knot_num]
                total_ni += polyval(rij, fj_coeffs)*partialsum

            # Build U coefficient matrix
            u_coeffs = np.zeros((4,len(self.potentials)))
            for l,p in enumerate(self.potentials):
                knots = p.us[u_idx].knotsx

                h = p.us[u_idx].h

                top = total_ni[l] - knots[0]
                tmp1 = top/h
                tmp2 = np.floor(tmp1)
                tmp = int(tmp2)
                u_spline_num = tmp + 1
                #u_spline_num = int(np.floor((total_ni[l]-knots[0])/h)) + 1

                if u_spline_num <= 0:
                    #str = "EXTRAP: total_ni = {0}".format(total_ni)
                    #logging.info(str)
                    u_spline_num = 0
                    u_knot_num = 0
                elif u_spline_num > len(knots):
                    #str = "EXTRAP: total_ni = {0}".format(total_ni)
                    #logging.info(str)
                    u_spline_num = len(knots)
                    u_knot_num = u_spline_num - 1
                else:
                    u_knot_num = u_spline_num - 1

                #logging.info("expected = {0}, total_ni = {1}, knot0 = {"
                #             "2}".format(p.us[u_idx](total_ni), total_ni,
                #                         knots[u_spline_num-1]))
                u_coeffs[:,l] = p.us[u_idx].cmat[:,u_spline_num]
                total_ni[l] -= knots[u_knot_num]

            u_coeffs = u_coeffs[::-1]

            #logging.info("real = {0}, total_ni = {1}".format(polyval(total_ni,
            #    u_coeffs,tensor=False)-self.zero_atom_energies[:,u_idx],
            #                                                 total_ni))
            energies += polyval(total_ni, u_coeffs,tensor=False) -\
                        self.zero_atom_energies[:,u_idx]

        return energies

    def compute_forces(self):
        pass

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

def eval_all_potentials(x, coeffs):
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
