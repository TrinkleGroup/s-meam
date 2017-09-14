"""Authors: Josh Vita (University of Illinois at Urbana-Champaign), Dallas
            Trinkle (UIUC)
Contact: jvita2@illinois.edu"""

# TODO: will need to be able to set individual splines, not just the whole set

import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from potential import Potential
from spline import Spline
from ase.neighborlist import NeighborList

class MEAM(Potential):
    """MEAM potential object
       
    Note:
        For indexing of splines, see notes in eval() method"""

    def  __init__(self, fname=None, fmt='lammps', types=[]):
        Potential.__init__(self)

        if fname:
            self.read_from_file(fname, fmt)
        else:
            ntypes = len(types)
            # self.ntypes is set in @setter.types
            self.types = types
            self.phis = [None]*((ntypes+1)*ntypes/2)
            self.rhos = [None]*ntypes
            self.us = [None]*ntypes
            self.fs = [None]*ntypes
            self.gs = [None]*((ntypes+1)*ntypes/2)
            self.zero_atom_energies = [None]*ntypes

    @property
    def phis(self):
        """N pair interaction spline functions"""
        return self._phis

    @phis.setter
    def phis(self,phis):
        self._phis = phis

    @property
    def rhos(self):
        """N electronic density spline functions"""
        return self._rhos

    @rhos.setter
    def rhos(self,rhos):
        self._rhos = rhos

    @property
    def us(self):
        """N embedding spline functions"""
        return self._us

    @us.setter
    def us(self,us):
        self._us = us

    @property
    def fs(self):
        """N additional spline functions"""
        return self._fs

    @fs.setter
    def fs(self,fs):
        self._fs = fs

    @property
    def gs(self):
        """N angular term spline functions"""
        return self._gs

    @gs.setter
    def gs(self,gs):
        self._gs = gs

    @property
    def types(self):
        """Ordered list of elemental string names OR atomic numbers"""
        return self._types

    @types.setter
    def types(self, types):
        self._types = types
        self._ntypes = len(types)

    @property
    def ntypes(self):
        """The number of unique elements in the system"""
        return self._ntypes

    @ntypes.setter
    def ntypes(self, n):
        raise AttributeError, "To set the number of elements in the system, specify the types using the MEAM.types attribute"

    def eval(self, atoms):
        """Evaluates the energies for the given system using the MEAM potential,
        following the notation specified in the LAMMPS spline/meam documentation
        Args:
            atoms (Atoms):
                ASE Atoms object containing all atomic information
        
        Note that splines are indexed as follows:
            phis, gs: 11,12,...,1N,22,...,2N,...,NN
            rhos, us, fs: 1,2,...,N
            
            e.g. for Ti-O where Ti is atom type 1, O atom type 2
                phis, gs: Ti-Ti, Ti-O, O-O
                rhos, us, fs: Ti, O"""

        # TODO: currently, this is the WORST case scenario in terms of runtime
        # TODO: avoid passing whole array? generator?
        # TODO: Check that all splines are not None
        # TODO: how to compute per-atom forces
        # TODO: is there anywhere that map() could help?

	# nl allows double-counting of bonds, nl_noboth does not
        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=True, skin=0.0)
        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=False, skin=0.0)
        nl.build(atoms)
        nl_noboth.build(atoms)

        total_pe = 0.0
        natoms = len(atoms)
        cellx,celly,cellz = atoms.get_cell()
        
        for i in xrange(natoms):
            itype = symbol_to_type(atoms[i].symbol, self.types)
            iposx = atoms[i].position

            # Pull atom-specific neighbor lists
            neighbors = nl.get_neighbors(i)
            neighbors_noboth = nl_noboth.get_neighbors(i)

            num_neighbors = len(neighbors[0])
            num_neighbors_noboth = len(neighbors_noboth[0])

            # Build the list of shifted positions for atoms outside of unit cell
            neighbor_shifted_positions = []
            for l in xrange(len(neighbors[0])):
                shiftx,shifty,shiftz = neighbors[1][l]
                neigh_pos = atoms[neighbors[0][l]].position + shiftx*cellx + shifty*celly +\
                        shiftz*cellz

                neighbor_shifted_positions.append(neigh_pos)
            # end shifted positions loop

            # TODO: workaround for this if branch; do we even need it??
            if len(neighbors[0]) > 0:
                tripcounter = 0
                total_phi = 0.0
                total_u = 0.0
                total_rho = 0.0
                total_ni = 0.0

                u = self.us[i_to_potl(itype)]

                # Calculate pair interactions (phi)
                for j in xrange(num_neighbors_noboth): # j = index for neighbor list
                    jtype = symbol_to_type(\
                            atoms[neighbors_noboth[0][j]].symbol, self.types)
                    r_ij = np.linalg.norm(iposx-neighbor_shifted_positions[j])

                    phi = self.phis[ij_to_potl(itype,jtype,self.ntypes)]

                    total_phi += phi(r_ij)
                # end phi loop

                for j in xrange(num_neighbors):
                    jtype = symbol_to_type(atoms[neighbors[0][j]].symbol, self.types)
                    r_ij = np.linalg.norm(iposx-neighbor_shifted_positions[j])

                    rho = self.rhos[i_to_potl(jtype)]
                    fj = self.fs[i_to_potl(jtype)]

                    # Used for triplet calculations
                    a = neighbor_shifted_positions[j]-iposx
                    na = np.linalg.norm(a)
                    
                    partialsum = 0.0
                    for k in xrange(j,num_neighbors):
                        if k != j:
                            ktype = symbol_to_type(\
                                    atoms[neighbors[0][k]].symbol, self.types)
                            r_ik = np.linalg.norm(iposx-\
                                    neighbor_shifted_positions[k])

                            b = neighbor_shifted_positions[k]-iposx

                            fk = self.fs[i_to_potl(ktype)]
                            g = self.gs[ij_to_potl(jtype,ktype,self.ntypes)]
                            
                            nb = np.linalg.norm(b)

                            # TODO: try get_dihedral() for angles
                            cos_theta = np.dot(a,b)/na/nb

                            fk_val = fk(r_ik)
                            g_val = g(cos_theta)

                            partialsum += fk_val*g_val
                            tripcounter += 1
                    # end triplet loop

                    fj_val = fj(r_ij)
                    total_ni += fj_val*partialsum
                    total_ni += rho(r_ij)
                # end u loop

                total_pe += total_phi + u(total_ni) -\
                        self.zero_atom_energies[i_to_potl(itype)]
            # end atom loop

        return total_pe

    def read_from_file(self, fname, fmt='lammps'):
        """Builds MEAM potential using spline information from the given LAMMPS
        potential file.
        
        Args:
            fname (str)
                the name of the input file
            fmt (str)
                currently implemented: ['lammps'], default = 'lammps'
            
        Returns:
            phis (list):
                pair interaction Splines
            rhos (list):
                electron density Splines
            us (list):
                embedding Splines
            fs (list):
                three-body corrector Splines
            gs (list):
                angular Splines
            types (list):
                elemental names of system components (e.g. ['Ti', 'O'])"""

        if fmt == 'lammps':
            try:
                f = open(fname, 'r')
                f.readline()                    # Remove header
                temp = f.readline().split()     # 'meam/spline ...' line
                types = temp[2:]
                ntypes = len(types)

                nsplines = ntypes*(ntypes+4)    # Based on how fxns in MEAM are defined

                # Calculate the number of splines for phi/g each
                nphi = (ntypes+1)*ntypes/2

                # Build all splines; separate into different types later
                splines = []
                for i in xrange(nsplines):
                    f.readline()                # throwaway 'spline3eq' line
                    nknots = int(f.readline())

                    d0,dN = [float(el) for el in f.readline().split()]
                    
                    xcoords = []                # x-coordinates of knots
                    ycoords = []                # y-coordinates of knots
                    for j in xrange(nknots):
                        # still unsure why y2 is in the file... why store if it's
                        # recalculated again later??
                        x,y,y2 = [np.float(el) for el in f.readline().split()]
                        xcoords.append(x)
                        ycoords.append(y)

                    if i < nphi+ntypes:             # phi and rho
                        bc = ('natural', (1,0.0))
                    elif i < nphi+2*ntypes:         # u
                        bc = 'natural'
                    elif i < nphi+3*ntypes:         # f
                        bc = ('natural', (1,0.0))
                    else:                           # g
                        bc = 'natural'

                    temp = Spline(xcoords,ycoords,bc_type =((1,d0),(1,dN)),\
                            derivs=(d0,dN))

                    temp.cutoff = (xcoords[0],xcoords[len(xcoords)-1])
                    splines.append(temp)

                # Separate splines for each unique function
                idx = 0                         # bookkeeping
                phis = splines[0:nphi]          # first nphi are phi
                idx += nphi
                rhos = splines[idx:idx+ntypes]  # next ntypes are rho
                idx += ntypes
                us = splines[idx:idx+ntypes]    # next ntypes are us
                idx += ntypes
                fs = splines[idx:idx+ntypes]    # next ntypes are fs
                idx += ntypes
                gs = splines[idx:]              # last nphi are gs

                # Using cutoff to be largest endpoint of radial functions
                splines = [phis,rhos,fs]
                endpoints = [max(fxn.x) for ftype in splines for fxn in ftype]
                cutoff = max(endpoints)

                self.types = types
                self.phis = phis
                self.rhos = rhos
                self.us = us
                self.fs = fs
                self.gs = gs
                self.cutoff = cutoff

                self.zero_atom_energies =  [None]*self.ntypes
                for i in xrange(self.ntypes):
                    self.zero_atom_energies[i] = self.us[i](0.0)

            except IOError as error:
                raise IOError("Could not open file '" + fname + "'")

def symbol_to_type(symbol, types):
    """Returns the numerical atom type, given its chemical symbol
    
    Args:
        symbol (str):
            Elemental symbol
        types (list):
            Ordered list of chemical symbols to search
            
    Returns:
        LAMMPS style atom type
        
        e.g. for ['Ti', 'O'], symbol_to_type('Ti',types) returns 1"""

    return np.where(np.array(types)==symbol)[0][0] + 1

def ij_to_potl(itype,jtype,ntypes):
    """Maps i and j element numbers to a single index of a 1D list; used for
    indexing spline functions. Taken directly from pair_meam_spline.h
    
    Args:
        itype (int):
            the i-th element in the system (e.g. in Ti-O, Ti=1)
        jtype (int):
            the j-the element in the system (e.g. in Ti-O, O=2)
        ntypes (int):
            the number of unique element types in the system
        
    Returns:
        The mapping of ij into an index of a 1D 0-indexed list"""

    return jtype - 1 + (itype-1)*ntypes - (itype-1)*itype/2

def i_to_potl(itype):
    """Maps element number i to an index of a 1D list; used for indexing spline
    functions. Taken directly from pair_meam_spline.h
    
    Args:
        itype (int):
            the i-the element in the system (e.g. in Ti-O, Ti=1)
        
    Returns:
        The array index for the given element"""

    return itype-1

if __name__ == "__main__":
    import lammpsTools

    p = MEAM('TiO.meam.spline')
    atoms = lammpsTools.atoms_from_file('Ti_only_crowd.Ti', ['Ti'])
    
    print(p.eval(atoms))
