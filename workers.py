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
        self.structs = []
        self.potentials = []

    @property
    def structs(self):
        """List of structures; each structure is an array of coordinates"""
        return self._structs

    @structs.setter
    def structs(self,structs):
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

    # TODO: an assumption is made that all potentials have the same cutoffs

    def __init__(self,struct,potentials=[]):
        # TODO: c should instead be a list of MEAM potentials
        super(Worker,self).__init__()

        self.structs = [struct]
        self.potentials = potentials

        # Initialize zero-point embedding energies
        self.zero_atom_energies = np.zeros((len(potentials),self.ntypes))
        for i,p in enumerate(self.potentials):
            for j in range(self.ntypes):
                self.zero_atom_energies[i][j] = p.us[j](0.0)

    @property
    def potentials(self):
        """Set of MEAM objects of various potentials"""
        return self._potentials

    @potentials.setter
    def potentials(self, p):
        self._potentials = p
        self._cutoff = p[0].cutoff
        self._types = p[0].types
        self._ntypes = len(self.types)

        N = len(p)
        self.phis   = [p[i].phis for i in range(N)]
        self.rhos   = [p[i].rhos for i in range(N)]
        self.us     = [p[i].us for i in range(N)]
        self.fs     = [p[i].fs for i in range(N)]
        self.gs     = [p[i].gs for i in range(N)]

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
        atoms = self.structs[0]
        natoms = len(atoms)

        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=False, skin=0.0)
        nl_noboth.build(atoms)

        # TODO: only getting phi working first; bothways=True not needed yet
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
                #rzm think of a prettier way to get knot_num

            u_coeffs = u_coeffs[::-1]

            # rzm: extrapolation failing for u; suspect bc negative values
            #logging.info("real = {0}, total_ni = {1}".format(polyval(total_ni,
            #    u_coeffs,tensor=False)-self.zero_atom_energies[:,u_idx],
            #                                                 total_ni))
            energies += polyval(total_ni, u_coeffs,tensor=False) -\
                        self.zero_atom_energies[:,u_idx]

        return energies

    def compute_forces(self):
        pass

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
            index of spline interval"""

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
