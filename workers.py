import numpy as np
import lammpsTools
import meam

from numpy.polynomial.polynomial import polyval
from ase.neighborlist import NeighborList

class Worker(object):

    def __init__(self):
        self.structs = []
        self.coefficients = []

    @property
    def structs(self):
        """List of structures; each structure is an array of coordinates"""
        return self._structs

    @structs.setter
    def structs(self,structs):
        self._structs = structs

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self,c):
        self._coefficients = c

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
        # nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
        #         self_interaction=False, bothways=True, skin=0.0)


        # energies[z] corresponds to the energy as calculated by the k-th
        # potential
        energies = np.zeros(len(self.potentials))

        for i in range(natoms):
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            neighbors_noboth, offsets = nl_noboth.get_neighbors(i)
            num_neighbors_noboth = len(neighbors_noboth)
            # TODO: build a proper distance matrix using shifted positions

            # Calculate pair interactions (phi)
            for j,offset in zip(neighbors_noboth,offsets):
                jtype = lammpsTools.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                rij = np.linalg.norm(ipos -jpos)
                #print(rij)
                #print(self.phis[0][0])
                #print(self.phis[0][0].cutoff)
                #print(self.cutoff)

                # Finds correct type of phi fxn
                pot_idx = meam.ij_to_potl(itype,jtype,self.ntypes)

                # Finds interval of spline
                h = self.potentials[0].phis[pot_idx].h
                spline_num = int(np.floor(rij/h))
                # rzm: interval search seems to be working properly, but isn't evaluating correctly
                # scipy doesn't order coeffs same way as numpy

                phis = self.phis

                #for grp in phis:
                #    for p in grp:
                #        p.plot()

                # TODO: splines will need extrapolation coefficients; meaning
                # there will be two additional splines for endpoints

                # Extracts coefficients
                p = phis[0][pot_idx]
                rij -= p.knotsx[spline_num] # Spline.c coefficients assume spline starts at 0
                phi_coeffs =np.array([phis[potnum][pot_idx].c[:,spline_num]\
                        for potnum in range(len(phis))])

                phi_coeffs = phi_coeffs.transpose() # ordering for polyval()
                phi_coeffs = phi_coeffs[::-1]

                val = polyval(rij, phi_coeffs)
                energies += polyval(rij, phi_coeffs)

                # TODO: is it actually faster to create matrix of coefficients,
                # then use np.polynomial.polynomial.polyval() (Horner's) than to just use
                # CubicSpline.__call__() (unknown method; from LAPACK)?

        return energies

    def compute_forces(self):
        pass
