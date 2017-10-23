import numpy as np
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

    def compute_energies(self):
        raise NotImplementedError

    def compute_forces(self):
        raise NotImplementedError

class WorkerManyPotentialsOneStruct(Worker):
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials"""

    # TODO: instead of multiple worker types, could just do special evals based
    # on if A, X are matrices or not? Downside: can't manually decide method

    # TODO: an assumption is made that all potentials have the same cutoffs

    def __init__(self,x=[],potentials=[]):
        # TODO: c should instead be a list of MEAM potentials
        super(Worker,self).__init__()
        
        x = np.asarray(x)
        self.struct = x
        self.potentials = potentials

    @property
    def structs(self):
        """ASE Atoms object defining the system"""
        return self._struct

    @struct.setter
    def structs(self, struct):
        self._struct = struct

    @property
    def potentials(self):
        """Set of MEAM objects of various potentials"""
        return self._potentials

    @potentials.setter
    def potentials(self, p):
        self._potentials = p
        self.cutoff = p[0].cutoff
        self.types = p[0].types

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, c):
        raise AttributeError, "cutoff can only be set by setting the 'potentials' property"

    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, c):
        raise AttributeError, "types can only be set by setting the 'potentials' property"

    @property
    def ntypes(self):
        return self._ntypes

    @ntypes.setter
    def ntypes(self, n):
        raise AttributeError, "ntypes can only be set by setting the 'potentials' property"


    def compute_energies(self):
        atoms = self.struct
        natoms = len(atoms)

        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
                self_interaction=False, bothways=False, skin=0.0)
        nl.build(atoms)

        # TODO: only getting phi working first; bothways=True not needed yet
        # nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2.),\
        #         self_interaction=False, bothways=True, skin=0.0)


        # energies[z] corresponds to the energy as calculated by the k-th
        # potential
        energies = np.zeros(len(self.potentials))

        for i in range(natoms):
            itype = meam.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            neighbors, offsets = nl_noboth.get_neighbors(i)
            num_neighbors_noboth = len(neighbors_noboth)
            # TODO: build a proper distance matrix using shifted positions

            # Calculate pair interactions (phi)
            for j,offset in zip(neighbors,offsets):
                jtype = meam.symbol_to_type(atoms[j].symbol, self.types)
                jpos = atoms[j].position + np.dot(offset,atoms.get_cell())

                rij = np.linalg.norm(ipos -jpos)

                # Finds correct type of phi fxn
                pot_idx = meam.ij_to_potl(itype,jtype,self.ntypes)

                # Finds interval of spline
                h = self.potentials[0].phis[pot_idx].h
                spline_num = int(np.floor(rij,h))

                phis = [self.potentials[l].phis[pot_idx] for l in\
                        range(len(self.potentials))]

                # Extracts coefficients
                phi_coeffs= np.array([phis[l].c[:,spline_num] for l in range(len(phis))])
                phi_coeffs = phi_coeffs.transpose() # ordering for polyval()

                energies += polyval(rij, phi_coeffs)

                # TODO: is it actually faster to create matrix of coefficients,
                # then use np.polynomial.polynomial.polyval() (Horner's) than to just use
                # CubicSpline.__call__() (unknown method; from LAPACK)?

        return energies

    def compute_forces(self):
        pass
