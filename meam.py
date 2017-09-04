"""Authors: Josh Vita (University of Illinois at Urbana-Champaign), Dallas
            Trinkle (UIUC)
Contact: jvita2@illinois.edu"""

# TODO: will need to be able to set individual splines, not just the whole set
# TODO: personal Spline that also has cutoffs?

import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from potential import Potential
#from itertools import permutations
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
        # TODO: double check indexing is correct; consult ij_to_potl
        # TODO: also need array of angles b/w triplets
        # TODO: check that it's only iterating over lower triangle of
        #       distances array

        # TODO: Check that all splines are not None #raise NotImplementedError, "eval() not implemented yet" #pair_counter_phi = np.zeros((len(atoms),len(atoms))) #pair_counter_rho = np.zeros((len(atoms),len(atoms)))
        #trip_counter = np.zeros((len(atoms),len(atoms),len(atoms)))

        # Build the neighbor list and distances matrix
        # nl should not be counting an atom as its own neighbor OR double
        # counting 'bonds'
        # Using cutoff/2 because NeighborList builds a sphere around each atom
        # and checks if they intersect
        #print("Cutoff = %f" % self.cutoff)

        atoms.set_pbc(True)

        r = atoms.get_all_distances(mic=True)

        #for i in xrange(len(atoms)):
        #    print(nl.get_neighbors(i)[0])
        #    print(len(nl.get_neighbors(i)[0]))

        # TODO: how to compute per-atom forces
        # TODO: is there anywhere that map() could help?
        # TODO: for i in natoms: calc phipairs, rhopairs, triplets

        # Calculate for each atom; i = atom index
        total_pe = 0.0
        natoms = len(atoms)

        # Calculate pair terms
        #for i in xrange(natoms):
        #    itype = symbol_to_type(atoms[i].symbol, self.types)

        #    phi_val = 0.0
        #    #for j in nl_noboth.get_neighbors(i)[0]:
        #    #print("Atom %d's neighbors: " % i),
        #    for j in xrange(i):
        #        if r[i][j] <= self.cutoff:
        #            #print(str(j)),
        #            jtype = symbol_to_type(atoms[j].symbol, self.types)

        #            phi = self.phis[ij_to_potl(itype,jtype,self.ntypes)]

        #            #print("phi_val = %f" % phi(r[i][j]))
        #            total_pe += phi(r[i][j])
        #    #print("\n"),

        #plot_x = np.arange(natoms)
        #plot_y = np.ones(plot_x.shape)

        ## Calculate three-body terms
        #for i in xrange(natoms):
        #    itype = symbol_to_type(atoms[i].symbol, self.types)

        #    ni_val = 0.0
        #    total_rho_val = 0.0
        #    total_threebody_val = 0.0

        #    # Calculate rho contribution to ni
        #    for k in xrange(natoms):
        #        r_ik = r[i][k]

        #        if (k != i) and (r_ik <= self.cutoff):
        #            ktype = symbol_to_type(atoms[k].symbol, self.types)

        #            rho = self.rhos[i_to_potl(ktype)]

        #            total_rho_val += rho(r_ik)

        #    # Calculate threebody contribution to ni
        #    for k in xrange(natoms):
        #        r_ik = r[i][k]

        #        if (k != i) and (r_ik <= self.cutoff):
        #            ktype = symbol_to_type(atoms[k].symbol, self.types)
        #            for j in xrange(k):
        #                r_ij = r[i][j]

        #                if (j != i) and (r_ij <= self.cutoff):
        #                    jtype = symbol_to_type(atoms[j].symbol, self.types)

        #                    fj = self.fs[i_to_potl(jtype)]
        #                    fk = self.fs[i_to_potl(ktype)]
        #                    gkj = self.gs[ij_to_potl(ktype,jtype,self.ntypes)]

        #                    a = atoms[j].position-atoms[i].position
        #                    b = atoms[k].position-atoms[i].position

        #                    na = np.linalg.norm(a)
        #                    nb = np.linalg.norm(b)

        #                    cos_theta = np.dot(a,b)/na/nb

        #                    #if (r_ij < fj.cutoff[0]) or (r_ik < fk.cutoff[0]):
        #                    #    print("Extrap LHS value of %f" % r_ij)

        #                    fj_val = fj(r_ij)
        #                    fk_val = fk(r_ik)
        #                    gkj_val = gkj(cos_theta)

        #                    total_threebody_val += fj_val*fk_val*gkj_val

        #    ni_val = total_rho_val + total_threebody_val
        #    #print(ni_val)

        #    u = self.us[i_to_potl(ktype)]
        #    #if (ni_val > u.cutoff[1]):
        #    #    print("Extrap RHS value of %f" % ni_val)

        #    u_val = u(ni_val)
        #    #print(u_val)

        #    total_pe += u_val

        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=True, skin=0.0)
        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=False, skin=0.0)
        nl.build(atoms)
        nl_noboth.build(atoms)

        plot_three = []
        ucounts = 0
        for i in xrange(natoms):
            itype = symbol_to_type(atoms[i].symbol, self.types)

            neighbors = nl.get_neighbors(i)[0]
            neighbors_noboth = nl_noboth.get_neighbors(i)[0]

            pairs = itertools.product([i], neighbors)
            pairs_noboth = itertools.product([i], neighbors_noboth)
            neighbors_without_j = neighbors
            #print(type(neighbors_without_j))

            # TODO: workaround for this if branch
            if len(neighbors) > 0:
                tripcounter = 0
                total_phi = 0.0
                total_u = 0.0
                total_rho = 0.0
                total_ni = 0.0

                u = self.us[i_to_potl(itype)]

                # Calculate pair interactions (phi)
                for pair in pairs_noboth:
                    _,j = pair
                    r_ij = r[i][j]
                    jtype = symbol_to_type(atoms[j].symbol, self.types)

                    phi = self.phis[ij_to_potl(itype,jtype,self.ntypes)]

                    total_phi += phi(r_ij)
                # end phi loop

                # Calculate embedding term (rho)
                #for pair in pairs:
                #    _,j = pair
                #    r_ij= r[i][j]
                #    jtype = symbol_to_type(atoms[j].symbol, self.types)

                #    rho = self.rhos[i_to_potl(jtype)]

                #    total_rho += rho(r_ij)
                ## end rho loop
            
                ## Re-instantiate since this is a generator
                #pairs_noboth = itertools.product([i], neighbors)
                    
                # TODO: problem; times that u(ni_val) is calculated should be
                # once for every atom, INDEPENDENT of # pairs
                # Calculate threebody interactions (u)
                for pair in pairs:

                    _,j = pair
                    r_ij = r[i][j]
                    jtype = symbol_to_type(atoms[j].symbol, self.types)

                    rho = self.rhos[i_to_potl(jtype)]

                    total_ni += rho(r_ij)

                    # Iteratively kill atoms; avoid 2x counting triplets
                    neighbors_without_j = np.delete(neighbors_without_j,\
                            np.where(neighbors_without_j==j))

                    # TODO: only delete after calculation, so that j==k is
                    # included?
                    triplets = itertools.product(pair,neighbors_without_j)

                    for _,k in triplets:
                        r_ik = r[i][k]
                        ktype = symbol_to_type(atoms[k].symbol, self.types)

                        fj = self.fs[i_to_potl(jtype)]
                        fk = self.fs[i_to_potl(ktype)]
                        g = self.gs[ij_to_potl(jtype,ktype,self.ntypes)]

                        a = atoms[j].position-atoms[i].position
                        b = atoms[k].position-atoms[i].position

                        na = np.linalg.norm(a)
                        nb = np.linalg.norm(b)

                        # TODO: try get_dihedral() for angles
                        cos_theta = np.dot(a,b)/na/nb

                        fj_val = fj(r_ij)
                        fk_val = fk(r_ik)
                        g_val = g(r_ik)

                        total_ni += fj_val*fk_val*g_val
                        tripcounter += 1
                    # end triplet loop

                    #plot_three.append(total_threebody)
                    #ni_val = total_rho + total_threebody
                    #u_val = u(ni_val)

                    #total_u += u_val
                # end u loop

                print("ni_val = %f || " % total_ni),
                print("%f" % u(total_ni))
                #print("zero_atom_energy = %f" % self.zero_atom_energies[i_to_potl(itype)])
                ucounts += 1
                total_pe += total_phi + u(total_ni) -\
                        self.zero_atom_energies[i_to_potl(itype)]

        #plt.figure()
        #plt.plot(np.arange(len(plot_three)), plot_three)
        #plt.show()
            #print("%d trips for atom %i" %(tripcounter,i))
        # end atoms loop

        # Pairs
        #for k in neighbors:
        #paircounter = 0
        #total_threebody = 0.0
        #for k in xrange(natoms):
        #    rho_val = 0.0
        #    r_ik = r[i][k]
        #    if (k != i) and (r_ik < self.cutoff):
        #        #print(r_ik)
        #        # Don't need to check if k != i; nl does not include self
        #        ktype = symbol_to_type(atoms[k].symbol, self.types)

        #        rho = self.rhos[i_to_potl(ktype)]

        #        rho_val += rho(r_ik)

        #        # Triplets
        #        threebody_val = 0.0
        #        for j in xrange(k):
        #            paircounter += 1
        #            # DO need to check j != k since nl wasn't built for k
        #            r_ij = r[i][j]

        #            if (j != i) and (r_ij < self.cutoff):
        #                jtype = symbol_to_type(atoms[j].symbol, self.types)

        #                fj = self.fs[i_to_potl(jtype)]
        #                fk = self.fs[i_to_potl(ktype)]
        #                gkj = self.gs[ij_to_potl(ktype,jtype,self.ntypes)]

        #                a = atoms[j].position-atoms[i].position
        #                b = atoms[k].position-atoms[i].position

        #                na = np.linalg.norm(a)
        #                nb = np.linalg.norm(b)

        #                cos_theta = np.dot(a,b)/na/nb

        #                fj_val = fj(r_ij)
        #                fk_val = fk(r_ik)
        #                gkj_val = gkj(cos_theta)

        #                threebody_val += fj_val*fk_val*gkj_val
        #                total_threebody += threebody_val
        #                #print('temp = %f' % (fj_val*fk_val*gkj_val))

        #    #if abs(rho_val)>35.0: print("rho_val = %f" % rho_val)
        #    #if abs(threebody_val)>0: print("threebody_val = %f" % threebody_val)
        #    ni_val += rho_val + threebody_val
        #    paircounter += 1
        #
        #print("total_threebody = %f" % total_threebody)
        ##plot_y[i] = rho_val
        #u = self.us[i_to_potl(ktype)]
        #u_val = u(ni_val)
        ##print('rho_val = %f' % rho_val)
        ##print('ni_val = %f' % ni_val)
        ##print('u_val = %f' % u_val)
        #total_pe += u_val

        #plt.plot(plot_y,'o')
        #plt.show()
        #print("ucounts = %d" % ucounts)
        return(total_pe)

        ## Calculate manybody terms
        #for i in xrange(len(atoms)):
        #    itype = symbol_to_type(atoms[i].symbol, self.types)

        #    neighbors = nl.get_neighbors(i)[0]
        #    pairs = itertools.product([i], neighbors)

        #    # Get chemical symbol to index number

        #    # TODO: for triplets, do all atoms need to be in cutoff of i?
        #    # TODO: ensure no double-counting of triplets/pairs
        #        # maybe do some kind of array counter

        #    if len(neighbors)>0:
        #        neighbors_without_j = neighbors
        #        rho_val = 0.0
        #        counter = 0
        #        threebody_val = 0.0
        #        for pair in pairs:
        #            _,j = pair
        #            r_ij = r[i][j]
        #            jtype = symbol_to_type(atoms[j].symbol, self.types)

        #            rho = self.rhos[i_to_potl(jtype)]
        #            rho_val += rho(r_ij)
        #            #print("rho = %f" % rho(r_ij))

        #            #pair_counter_rho[i][j] += 1

        #            neighbors_without_j =np.delete(neighbors_without_j,np.where(neighbors_without_j==j))
        #            triplets = itertools.product(pair,neighbors_without_j)

        #            for _,k in triplets:
        #                r_ik = r[i][k]
        #                ktype = symbol_to_type(atoms[k].symbol, self.types)

        #                #print("r_ik = %f" % r_ik)

        #                a = atoms[j].position-atoms[i].position
        #                b = atoms[k].position-atoms[i].position

        #                na = np.linalg.norm(a)
        #                nb = np.linalg.norm(b)

        #                cos_theta = np.dot(a,b)/na/nb

        #                #print("cos_theta = %f" % cos_theta)

        #                fj = self.fs[i_to_potl(jtype)]
        #                fk = self.fs[i_to_potl(ktype)]
        #                gjk = self.gs[ij_to_potl(jtype,ktype,self.ntypes)]

        #                fj_val = fj(r_ij)

        #                fk_val = fk(r_ik)

        #                gjk_val = gjk(r_ik)

        #                #fj_val = fj(r_ij) if fj.in_range(r_ij) else 0.0
        #                #fk_val = fk(r_ik) if fk.in_range(r_ik) else 0.0
        #                #gjk_val=gjk(cos_theta)if gjk.in_range(cos_theta)else 0.0

        #                #fj_val = fj(r_ij)
        #                #fk_val = fk(r_ik)
        #                #gjk_val=gjk(cos_theta)

        #                #print("cos_theta = %s" % cos_theta)
        #                #print("fj.cutoff = %f %f" % fj.cutoff)
        #                #print("fj_val = %f" % fj_val)
        #                #print("fk_val= %f" % fk_val)
        #                #print("gjk_val = %f" % gjk_val)

        #                threebody_val += fj_val*fk_val*gjk_val

        #                #trip_counter[i][j][k] += 1
        #        
        #            ni = rho_val + threebody_val
        #            #print("U.cutoff = %f %f" % self.us[i_to_potl(itype)].cutoff)
        #            #print("ni = %f" % ni)
        #            #print("U(ni) = %.16f" % self.us[i_to_potl(itype)](ni))
        #            #print("Phi = %f" %\
        #            #        self.phis[ij_to_potl(itype,jtype,self.ntypes)](r_ij))
        #            #val =\
        #            #        self.phis[ij_to_potl(itype,jtype,self.ntypes)](r_ij)+\
        #            #        self.us[i_to_potl(itype)](ni)
        #            #print(i,val)

        #            phi = self.phis[ij_to_potl(itype,jtype,self.ntypes)]
        #            u = self.us[ij_to_potl(itype,jtype,self.ntypes)]

        #            if phi.in_range(r_ij):
        #                phi_val = phi(r_ij)
        #            else:
        #                phi_val = phi.extrap(r_ij)

        #            if u.in_range(ni):
        #                u_val = u(ni)
        #            else:
        #                u_val = u.extrap(ni)

        #            counter += phi_val+u_val

        #            total_pe += phi_val + u_val
        #            #pair_counter_phi[i][j] += 1
        #        #print("Atom %d contribution: %f" % (i, counter))

        #print(total_pe)
        #return total_pe

    def read_from_file(self, fname, fmt='lammps'):
        """Builds MEAM potential using spline information from the given LAMMPS
        potential file.
        
        Args:
            fname (str)
                the name of the input file
            fmt (str)
                currently implemented: ['lammps'], default = 'lammps'
            
        Returns:
            phis    -   (list) N pair interaction spline functions
            rhos    -   (list) N atom electronic density spline functions
            us      -   (list) N embedding spline functions
            fs      -   (list) N additional spline functions
            gs      -   (list) N angular term spline functions
            types   -   (list) names of elements in system (e.g. ['Ti', 'O']"""

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

                    if (i<nphi+ntypes) or ((i>=nphi+2*ntypes) and\
                            (i<nsplines-nphi)):
                        temp = Spline(xcoords,ycoords,bc_type =((1,d0),(1,dN)))
                    else:
                        #temp = Spline(xcoords,ycoords)
                        temp = Spline(xcoords,ycoords,bc_type =((1,d0),(1,dN)))

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
                    #print(self.us[i](0.0))

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
        itype   -   (int) the i-th element in the system (e.g. in Ti-O, Ti=1)
        jtype   -   (int) the j-th element in the system (e.g. in Ti-O, O=2)
        ntypes  -   (int) the number of unique element types in the system
        
    Returns:
        The mapping of ij into an index of a 1D 0-indexed list"""

    return jtype - 1 + (itype-1)*ntypes - (itype-1)*itype/2

def i_to_potl(itype):
    """Maps element number i to an index of a 1D list; used for indexing spline
    functions. Taken directly from pair_meam_spline.h
    
    Args:
        itype   -   (int) the i-the element in the system (e.g. in Ti-O, Ti=1)
        
    Returns:
        The array index for the given element"""

    return itype-1

def calc_theta(j,i,k):
    """Calculates the angle between points j,i,k where i is the central atom.
    
    Args:
        i,j,k (np.arr):
            The 3D cartesian coordinates of the three points
            
    Returns:
        theta (float):
            The angle between points j,i,k in RADIANS"""

    a = j-i
    b = k-i

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    return np.arccos()

if __name__ == "__main__":
    import lammpsTools

    p = MEAM('TiO.meam.spline')
    atoms = lammpsTools.atoms_from_lammps_data('Ti_only_crowd.Ti', ['Ti'])
    
    print(p.eval(atoms))
