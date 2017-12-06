"""Authors: Josh Vita (University of Illinois at Urbana-Champaign), Dallas
            Trinkle (UIUC)
Contact: jvita2@illinois.edu"""

# TODO: will need to be able to set individual splines, not just the whole set

import numpy as np
import lammpsTools
import logging

from potential import Potential
from spline import Spline, ZeroSpline
from ase.neighborlist import NeighborList

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MEAM(Potential):
    """Represents a spline-MEAM potential as a collection of spline
    interpolants for the phi, rho, u, f, and g functions for each type of
    interaction in the system. In total, this is represented as a 3D array
    with indices as follows:

    Index number: value represented
        0:  spline number (e.g. phi_Ti, phi_TiO, ... , g_TiO, g_O)
        1:  knot number (each spline is determined by its knot coordinates)
        2:  the coordinate direction (x=0, y=1)

    Important notes:
        1)  the variable 'nphi' represents the number of phi pair interactions
        in the system and is used for indexing the splines
        2)  splines are ordered as [phis, rhos, us, fs, gs] where there are
        nphi # of phis, ntype rhos, ntype us, ntype fs, and nphi gs.
        3)  not all splines have the same number of knots; the number of
        knots will be set to the maximum of any of the splines, but splines
        with fewer knots will be zero-padded to fit the shape

        TODO: SHOULD we zero-pad??
        """

    def  __init__(self, fname=None, types=[], fmt='lammps', splines=[],
                  d0=[], dN=[]):
        Potential.__init__(self)

        if fname:
            """Used to read in a LAMMPS-style spline-meam file"""
            # raise NotImplementedError("Reading splines from a file has not "
            #                           "been implemented yet.")
            if splines:
                raise AttributeError("Can only specify either fname or splines")
            self.read_from_file(fname, fmt)
            nphi = int((self.ntypes+1)*self.ntypes/2)
            self.nphi = nphi
        elif len(splines) > 0:
            self.splines = splines
            self.d0 = d0
            self.dN = dN
            self.types = types  # sets self.ntypes too
            ntypes = self.ntypes

            if len(splines) != ((ntypes+4)*ntypes):
                raise AttributeError("Incorrect number of splines for given number of system components")

            # Calculate the number of splines for phi/g each
            nphi = int((ntypes+1)*ntypes/2)
            self.nphi = nphi

            self.y2 = second_derivatives(self.splines[:,:,0], self.splines[:,
                                                              :,1], d0, dN)

        else:
            ntypes = len(types)
            nsplines = ntypes*(ntypes+4) # Based on how fxns in MEAM are defined
            nphi = int((ntypes+1)*ntypes/2)
            self.nphi = nphi

            self.types = types # also sets self.ntypes
            self.splines = np.zeros((nsplines,1))

            self.zero_atom_energies = [None]*ntypes
            self.uprimes = None
            self.forces = None
            self.energies = None
            return

        # Find the maximum range of the potential

        # rzm: chop out radial functions
        # splits into [phis, rhos], [us], [fs], [gs]
        blocks = np.split(self.splines, (nphi+self.ntypes,
                        nphi+2*self.ntypes, nphi+3*self.ntypes))
        radial_fxns = np.concatenate((blocks[0], blocks[2]))
        # endpoints = [max(fxn[]) for ftype in splines for fxn in ftype]
        # self.cutoff = max(endpoints)
        self.cutoff = np.max(np.max(radial_fxns[:,:,0], axis=1))
        # TODO: ensure this is getting the proper cutoffs

        u_shift = nphi + 2
        self.zero_atom_energies =  [None]*self.ntypes
        for i in range(self.ntypes):
            # self.zero_atom_energies[i] = self.us[i](0.0)
            self.zero_atom_energies[i] = self.__call__(u_shift+i, 0.0)

        self.uprimes = None
        self.forces = None
        self.energies = None

    def __call__(self, i, x, deriv=0):
        """Evaluates the i-th spline at position x"""

        xknots = self.splines[i,:,0]
        y = self.splines[i,:,1]
        y2 = self.y2[i]

        xmin = xknots[0]
        xmax_shifted = xknots[-1]-xmin

        x -= xmin

        if x < 0.0:
            if deriv: return self.d0[i]

            return y[0] + self.d0[i]*x
        elif x >= xmax_shifted:
            if deriv: return self.dN[i]

            return y[-1] + self.dN[i]*(x-xmax_shifted)
        else:
            h = xknots[1]-xknots[0]

            # logging.info("x = {0}".format(x))
            # klo = int(np.floor((x-xknots[0])/h)) + 1
            klo = int(x/h)
            khi = klo+1

            a = xknots[khi] - x
            b = h - a

            if deriv:
                return (y[klo]-y[khi])/h + ((3.0*b*b - h*h)*y2[khi] - (
                        3.0*a*a - h*h) * y2[klo])

            return y[khi] - a*((y[klo]-y[khi])/h) + ((a*a - h*h)*a*y2[klo] +
                                                     (b*b - h*h)*b*y2[khi])

    @property
    def d0(self):
        """First derivatives of all splines at first knot"""
        return self._d0

    @d0.setter
    def d0(self, c):
        self._d0 = c

    @property
    def dN(self):
        """First derivatives of all splines at last knot"""
        return self._dN

    @dN.setter
    def dN(self, c):
        self._dN = c

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
        raise AttributeError("To set the number of elements in the system, specify the types using the MEAM.types attribute")

    def compute_forces(self, atoms):
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

        self.compute_energies(atoms)

        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=True, skin=0.0)
        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=False, skin=0.0)
        nl.build(atoms)
        nl_noboth.build(atoms)

        natoms = len(atoms)

        self.forces = np.zeros((natoms,3))
        cellx,celly,cellz = atoms.get_cell()
        
        for i in range(natoms):
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position
            print("Atom {0} of type {1}----------------------".format(i, itype))

            Uprime_i = self.uprimes[i]

            # Pull atom-specific neighbor lists
            neighbors = nl.get_neighbors(i)
            neighbors_noboth = nl_noboth.get_neighbors(i)

            num_neighbors = len(neighbors[0])
            num_neighbors_noboth = len(neighbors_noboth[0])

            # Build the list of shifted positions for atoms outside of unit cell
            neighbor_shifted_positions = []
            bonds = []
            for l in range(num_neighbors):
                shiftx,shifty,shiftz = neighbors[1][l]
                neigh_pos = atoms[neighbors[0][l]].position + shiftx*cellx + shifty*celly +\
                        shiftz*cellz

                neighbor_shifted_positions.append(neigh_pos)
            # end shifted positions loop

            # TODO: workaround for this if branch
            # TODO: add more thorough in-line documentation

            forces_i = np.zeros((3,))
            if len(neighbors[0]) > 0:
                for j in range(num_neighbors):
                    jtype = lammpsTools.symbol_to_type(atoms[neighbors[0][j]].symbol, self.types)

                    # TODO: make a j_tag variable; too easy to make mistakes
                    jpos = neighbor_shifted_positions[j]
                    jdel = jpos - ipos 
                    r_ij = np.linalg.norm(jdel)
                    jdel /= r_ij

                    fj_val = self.fs[i_to_potl(jtype, 'f', self.ntypes)](r_ij)
                    fj_prime = self.fs[i_to_potl(jtype, 'f', self.ntypes)](
                        r_ij,1)

                    forces_j = np.zeros((3,))

                    # print("fj_val = %.3f" % fj_val)
                    # print("fj_prime = %.3f" % fj_prime)

                    # Used for triplet calculations
                    a = neighbor_shifted_positions[j]-ipos
                    na = np.linalg.norm(a)
                    
                    for k in range(j,num_neighbors):
                        if k != j:
                            ktype = lammpsTools.symbol_to_type(\
                                    atoms[neighbors[0][k]].symbol, self.types)
                            kpos = neighbor_shifted_positions[k]
                            kdel = kpos - ipos
                            r_ik = np.linalg.norm(kdel)
                            kdel /= r_ik

                            # TODO: b == kdel
                            b = neighbor_shifted_positions[k]-ipos
                            nb = np.linalg.norm(b)

                            # TODO: try get_dihedral() for angles
                            cos_theta = np.dot(a,b)/na/nb

                            fk_val = self.fs[i_to_potl(ktype, 'f',
                                                       self.ntypes)](r_ik)
                            g_val = self.gs[
                                ij_to_potl(jtype, ktype, 'g', self.ntypes)\
                                ](cos_theta)

                            fk_prime = self.fs[i_to_potl(ktype, 'f',
                                                         self.ntypes)](r_ik,1)
                            g_prime = self.gs[
                                ij_to_potl(jtype, ktype, 'g', self.ntypes)](
                                cos_theta, 1)

                            # print("fk_val = %.3f" % fk_val)
                            # print("fk_prime = %.3f" % fk_prime)

                            # print("g_val = %.3f" % g_val)
                            # print("g_prime = %.3f" % g_prime)

                            fij = -Uprime_i*g_val*fk_val*fj_prime
                            fik = -Uprime_i*g_val*fj_val*fk_prime

                            # print("fij = {0}".format(fij))
                            # print("fik = {0}".format(fik))
                            # print("cos_values = {0}".format(cos_theta))
                            # print("rij_values = {0}".format(r_ij))
                            # print("rik_values = {0}".format(r_ik))

                            prefactor = Uprime_i*fj_val*fk_val*g_prime
                            prefactor_ij = prefactor / r_ij
                            prefactor_ik = prefactor / r_ik
                            fij += prefactor_ij * cos_theta
                            fik += prefactor_ik * cos_theta


                            # print("prefactor = {0}".format(prefactor))
                            # print("prefactor_ij = {0}".format(prefactor_ij))
                            # print("prefactor_ik = {0}".format(prefactor_ik))

                            fj = jdel*fij - kdel*prefactor_ij
                            forces_j += fj

                            fk = kdel*fik - jdel*prefactor_ik
                            forces_i -= fk


                            self.forces[neighbors[0][k]] += fk
                    # end triplet loop

                    self.forces[i] -= forces_j
                    self.forces[neighbors[0][j]] += forces_j
                # end pair loop

                self.forces[i] += forces_i
                print("forces_j = {0}".format(forces_j))
                print("forces_k = {0}".format(forces_i))

                # Calculate pair interactions (phi)
                for j in range(num_neighbors_noboth): # j = index for neighbor list
                    jtype = lammpsTools.symbol_to_type(\
                            atoms[neighbors_noboth[0][j]].symbol, self.types)
                    jpos = neighbor_shifted_positions[j]
                    jdel = jpos - ipos 
                    r_ij = np.linalg.norm(jdel)

                    rho_prime_i = self.rhos[i_to_potl(itype, 'rho',
                                                      self.ntypes)](r_ij,1)
                    rho_prime_j = self.rhos[i_to_potl(jtype, 'rho',
                                                      self.ntypes)](r_ij,1)

                    fpair = rho_prime_j*self.uprimes[i] +\
                            rho_prime_i*self.uprimes[neighbors_noboth[0][j]]

                    phi_prime = self.phis[ij_to_potl(itype, jtype, 'phi',
                                          self.ntypes)](r_ij,1)

                    fpair += phi_prime
                    fpair /= r_ij

                    self.forces[i] += jdel*fpair
                    self.forces[neighbors_noboth[0][j]] -= jdel*fpair
                # end phi loop

            # end atom loop

        return self.forces

    def compute_energies(self, atoms):
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
        self.energies = np.zeros((natoms,))
        self.uprimes = [None]*natoms

        cellx,celly,cellz = atoms.get_cell()
        
        for i in range(natoms):
            logging.info("Computing for atom {0}".format(i))
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            # Pull atom-specific neighbor lists
            neighbors = nl.get_neighbors(i)
            neighbors_noboth = nl_noboth.get_neighbors(i)

            num_neighbors = len(neighbors[0])
            num_neighbors_noboth = len(neighbors_noboth[0])

            # Build the list of shifted positions for atoms outside of unit cell
            neighbor_shifted_positions = []
            #for l in range(len(neighbors[0])):
            #    shiftx,shifty,shiftz = neighbors[1][l]
            #    neigh_pos = atoms[neighbors[0][l]].position + shiftx*cellx + shifty*celly +\
            #            shiftz*cellz

            #    neighbor_shifted_positions.append(neigh_pos)
            indices, offsets = nl.get_neighbors(i)
            for idx, offset in zip(indices,offsets):
                neigh_pos = atoms.positions[idx] + np.dot(offset,\
                        atoms.get_cell())
                neighbor_shifted_positions.append(neigh_pos)
            # end shifted positions loop

            # TODO: workaround for this if branch; do we even need it??
            if len(neighbors[0]) > 0:
                tripcounter = 0
                total_phi = 0.0
                total_u = 0.0
                total_rho = 0.0
                total_ni = 0.0

                u_idx = i_to_potl(itype, 'u', self.ntypes)

                # Calculate pair interactions (phi)
                for j in range(num_neighbors_noboth): # j = index for neighbor list
                    jtype = lammpsTools.symbol_to_type(\
                            atoms[neighbors_noboth[0][j]].symbol, self.types)
                    r_ij = np.linalg.norm(ipos-neighbor_shifted_positions[j])

                    phi_idx = ij_to_potl(itype, jtype, 'phi', self.ntypes)

                    total_phi += self.__call__(phi_idx, r_ij)
                # end phi loop
                logging.info("phi ----- {0}".format(total_phi))

                for j in range(num_neighbors):
                    jtype = lammpsTools.symbol_to_type(atoms[neighbors[0][j]].symbol, self.types)
                    r_ij = np.linalg.norm(ipos-neighbor_shifted_positions[j])

                    rho_idx = i_to_potl(jtype, 'rho', self.ntypes)
                    fj_idx = i_to_potl(jtype, 'f', self.ntypes)

                    # Used for triplet calculations
                    a = neighbor_shifted_positions[j]-ipos
                    na = np.linalg.norm(a)
                    
                    partialsum = 0.0
                    for k in range(j,num_neighbors):
                        if k != j:
                            ktype = lammpsTools.symbol_to_type(\
                                    atoms[neighbors[0][k]].symbol, self.types)
                            r_ik = np.linalg.norm(ipos-\
                                    neighbor_shifted_positions[k])

                            b = neighbor_shifted_positions[k]-ipos

                            fk_idx = i_to_potl(ktype, 'f', self.ntypes)
                            g_idx = ij_to_potl(jtype, ktype, 'g', self.ntypes)
                            
                            nb = np.linalg.norm(b)

                            # TODO: try get_dihedral() for angles
                            cos_theta = np.dot(a,b)/na/nb

                            fk_val = self.__call__(fk_idx, r_ik)
                            g_val = self.__call__(g_idx, cos_theta)

                            partialsum += fk_val*g_val
                            tripcounter += 1
                    # end triplet loop

                    fj_val = self.__call__(fj_idx, r_ij)
                    total_ni += fj_val*partialsum
                    total_ni += self.__call__(rho_idx, r_ij)
                # end u loop

                tmp_idx = i_to_potl(itype, 'u', self.ntypes) - (self.nphi +
                                                               self.ntypes)
                atom_e = total_phi + self.__call__(u_idx, total_ni) -\
                    self.zero_atom_energies[tmp_idx]
                self.energies[i] = atom_e
                total_pe += atom_e

                self.uprimes[i] = self.__call__(u_idx, total_ni,1)
            # end atom loop

        return total_pe

    def write_to_file(self, fname, fmt='lammps'):
        """Writes the potential to a file"""
        types = self.types
        ntypes = len(types)

        with open(fname, 'w') as f:
            # Write header lines
            f.write("# meam/spline potential parameter file produced by MEAM object\n")
            f.write("meam/spline %d %s\n" % (ntypes, " ".join(types)))

            def write_spline(s, y2, d0, dN):

                # Write additional spline info
                nknots = len(s)
                # knotsx = s.knotsx              # x-pos of knots
                knotsx = s[:,0]         # x-pos of knots
                # knotsy = s.knotsy             # y-pos of knots
                knotsy = s[:,1]         # y-pos of knots
                # y2 = s.y2               # second derivative at knots
                f.write("spline3eq\n")
                f.write("%d\n" % nknots)

                # der = s.knotsy1
                # d0 = der[0]; dN = der[nknots-1]
                #f.write("%.16f %.16f\n" % (d0,dN))

                # TODO: need to ensure precision isn't changed between read/write

                str1 = ("%.16f" % d0).rstrip('0').rstrip('.')
                str2 = ("%.16f" % dN).rstrip('0').rstrip('.')
                f.write(str1 + ' ' + str2 + '\n')

                # Write knot info
                for i in range(nknots):
                    str1 = ("%.16f" % knotsx[i]).rstrip('0').rstrip('.')
                    str2 = ("%.16f" % knotsy[i]).rstrip('0').rstrip('.')
                    str3 = ("%.16f" % y2[i]).rstrip('0').rstrip('.')
                    f.write(str1 + ' ' + str2 + ' ' + str3 + '\n')

            # # Output all splines
            # for fxn in self.phis:
            #     write_spline(fxn)
            # for fxn in self.rhos:
            #     write_spline(fxn)
            # for fxn in self.us:
            #     write_spline(fxn)
            # for fxn in self.fs:
            #     write_spline(fxn)
            # for fxn in self.gs:
            #     write_spline(fxn)

            for i in range(len(self.splines)):
                write_spline(self.splines[i,:,:], self.y2[i], self.d0[i],
                             self.dN[i])

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

                # Build all splines; separate into different types later
                all_xcoords = []
                all_ycoords = []
                all_d0      = [ ]
                all_dN      = [ ]

                max_nknots = 0
                for i in range(nsplines):
                    f.readline()                # throwaway 'spline3eq' line
                    nknots = int(f.readline())

                    if nknots > max_nknots: max_nknots = nknots

                    d0,dN = [float(el) for el in f.readline().split()]
                    all_d0.append(d0)
                    all_dN.append(dN)

                    xcoords = []                # x-coordinates of knots
                    ycoords = []                # y-coordinates of knots
                    for j in range(nknots):
                        # still unsure why y2 is in the file... why store if it's
                        # recalculated again later??
                        x,y,y2 = [np.float(el) for el in f.readline().split()]
                        xcoords.append(x)
                        ycoords.append(y)

                    # all_xcoords.append(xcoords)
                    # all_ycoords.append(ycoords)
                    all_xcoords.append(np.array(xcoords))
                    all_ycoords.append(np.array(ycoords))


                all_xcoords = np.array(all_xcoords)
                all_ycoords = np.array(all_ycoords)

                self.d0         = np.array(all_d0)
                self.dN         = np.array(all_dN)
                self.y2         = second_derivatives(all_xcoords,
                                                     all_ycoords,
                                                     self.d0, self.dN)
                self.types      = types
                self.nphi       = int((ntypes+1)*ntypes/2)

                # TODO: in reality; can't do this since input to MEAM will be
                #  a whole collection of knot points
                tmp_splinesx = []
                tmp_splinesy = []
                for i in range(nsplines):
                    nknots = len(all_xcoords[i])

                    diff = max_nknots - nknots

                    tmp_splinesx.append(np.pad(all_xcoords[i], (0,diff),
                                         'constant', constant_values=0))
                    tmp_splinesy.append(np.pad(all_ycoords[i], (0,diff),
                                              'constant', constant_values=0))

            # rzm set self.splines
            except IOError as error:
                raise IOError("Could not open file '" + fname + "'")

            tmp_splinesx = np.array(tmp_splinesx)
            tmp_splinesy = np.array(tmp_splinesy)

            # self.splines = self.splines.reshape((nsplines,max_nknots,2))
            self.splines = np.stack((tmp_splinesx, tmp_splinesy), axis=2)
            return

    def plot(self):
        """Generates plots of all splines"""

        ntypes = self.ntypes
        splines = self.phis + self.rhos + self.us + self.fs + self.gs

        for i,s in enumerate(splines):
            s.plot(saveName=str(i+1)+'.png')
                # TODO: finish this for generic system, not just binary/unary


# TODO: all subtype functions are assuming a 2-component system
def phionly_subtype(pot):
    """Returns the phionly version of pot

    Args:
        pot (MEAM):
            the original potential

    Returns:
        phionly (MEAM):
            the phionly version of pot"""

    N           = pot.ntypes        # number of components in the system
    nsplines    = N*(N+4)           # number of splines
    nphi        = int((N+1)*N/2)    # number of each phi and g splines

    # splines = np.zeros(pot.splines.shape)
    splines = self.splines.copy()
    d0      = np.zeros(pot.d0.shape)
    dN      = np.zeros(pot.dN.shape)

    for i in range(N*(N+4)):
        if i < nphi:
            splines[i]  = pot.splines[i]
            d0[i]       = pot.d0[i]
            dN[i]       = pot.dN[i]
        else:
            splines[i,:,1] = np.zeros(len(splines[i,:,1]))

    return MEAM(splines=splines, types=pot.types, d0=d0, dN=dN)

def nophi_subtype(pot):
    """Returns the nophi version of pot

    Args:
        pot (MEAM):
            the original potential

    Returns:
        nophi (MEAM):
            the nophi version of pot"""

    N           = pot.ntypes        # number of components in the system
    nsplines    = N*(N+4)           # number of splines
    nphi        = int((N+1)*N/2)    # number of each phi and g splines

    # splines = np.zeros(pot.splines.shape)
    splines = self.splines.copy()
    d0      = np.zeros(pot.d0.shape)
    dN      = np.zeros(pot.dN.shape)

    for i in range(N*(N+4)):
        if i >= nphi:
            splines[i]  = pot.splines[i]
            d0[i]       = pot.d0[i]
            dN[i]       = pot.dN[i]
        else:
            splines[i,:,1] = np.zeros(len(splines[i,:,1]))

    return MEAM(splines=splines, types=pot.types, d0=d0, dN=dN)

def rhophi_subtype(pot):
    """Returns the rhophi version of pot

    Args:
        pot (MEAM):
            the original potential

    Returns:
        rhophi (MEAM):
            the rhophi version of pot"""

    N           = pot.ntypes        # number of components in the system
    nsplines    = N*(N+4)           # number of splines
    nphi        = int((N+1)*N/2)    # number of each phi and g splines

    # splines = np.zeros(pot.splines.shape)
    splines = self.splines.copy()
    d0      = np.zeros(pot.d0.shape)
    dN      = np.zeros(pot.dN.shape)

    for i in range(N*(N+4)):
        if i < (nphi+N+N):
            splines[i]  = pot.splines[i]
            d0[i]       = pot.d0[i]
            dN[i]       = pot.dN[i]
        else:
            splines[i,:,1] = np.zeros(len(splines[i,:,1]))

    return MEAM(splines=splines, types=pot.types, d0=d0, dN=dN)

def norhophi_subtype(pot):
    """Returns the norhophi version of pot

    Args:
        pot (MEAM):
            the original potential

    Returns:
        norhophi (MEAM):
            the norhophi version of pot"""

    N           = pot.ntypes        # number of components in the system
    nsplines    = N*(N+4)           # number of splines
    nphi        = int((N+1)*N/2)    # number of each phi and g splines

    # splines = np.zeros(pot.splines.shape)
    splines = self.splines.copy()
    d0      = np.zeros(pot.d0.shape)
    dN      = np.zeros(pot.dN.shape)

    for i in range(N*(N+4)):
        if i >= (nphi+N):
            splines[i]  = pot.splines[i]
            d0[i]       = pot.d0[i]
            dN[i]       = pot.dN[i]
        else:
            splines[i,:,1] = np.zeros(len(splines[i,:,1]))

    return MEAM(splines=splines, types=pot.types, d0=d0, dN=dN)

def norho_subtype(pot):
    """Returns the norhos version of pot

    Args:
        pot (MEAM):
            the original potential

    Returns:
        norhos (MEAM):
            the norhos version of pot"""

    N           = pot.ntypes        # number of components in the system
    nsplines    = N*(N+4)           # number of splines
    nphi        = int((N+1)*N/2)    # number of each phi and g splines

    # splines = np.zeros(pot.splines.shape)
    splines = self.splines.copy()
    d0      = np.zeros(pot.d0.shape)
    dN      = np.zeros(pot.dN.shape)

    for i in range(N*(N+4)):
        if (i < nphi) or (i >= (nphi+N)):
            splines[i]  = pot.splines[i]
            d0[i]       = pot.d0[i]
            dN[i]       = pot.dN[i]
        else:
            splines[i,:,1] = np.zeros(len(splines[i,:,1]))

    return MEAM(splines=splines, types=pot.types, d0=d0, dN=dN)

def rho_subtype(pot):
    """Returns the rhos version of pot

    Args:
        pot (MEAM):
            the original potential

    Returns:
        rhos (MEAM):
            the rhos version of pot"""

    N           = pot.ntypes        # number of components in the system
    nsplines    = N*(N+4)           # number of splines
    nphi        = int((N+1)*N/2)    # number of each phi and g splines

    # splines = np.zeros(pot.splines.shape)
    splines = self.splines.copy()
    d0      = np.zeros(pot.d0.shape)
    dN      = np.zeros(pot.dN.shape)

    for i in range(N*(N+4)):
        if (i >= nphi) and (i < (nphi+N+N)):
            splines[i]  = pot.splines[i]
            d0[i]       = pot.d0[i]
            dN[i]       = pot.dN[i]
        else:
            splines[i,:,1] = np.zeros(len(splines[i,:,1]))

    return MEAM(splines=splines, types=pot.types, d0=d0, dN=dN)

def nog_subtype(pot):
    # TODO: need to set g=1
    """Returns the nog version of pot

    Args:
        pot (MEAM):
            the original potential

    Returns:
        nog (MEAM):
            the nog version of pot"""

    N           = pot.ntypes        # number of components in the system
    nsplines    = N*(N+4)           # number of splines
    nphi        = int((N+1)*N/2)    # number of each phi and g splines

    # splines = np.zeros(pot.splines.shape)
    splines = self.splines.copy()
    d0      = np.zeros(pot.d0.shape)
    dN      = np.zeros(pot.dN.shape)

    for i in range(N*(N+4)):
        if i < (nphi+3*N):
            splines[i]  = pot.splines[i]
            d0[i]       = pot.d0[i]
            dN[i]       = pot.dN[i]
        else:
            splines[i,:,1] = np.zeros(len(splines[i,:,1]))

    return MEAM(splines=splines, types=pot.types, d0=d0, dN=dN)

def ij_to_potl(itype, jtype, pot_type, ntypes):
    """Maps i and j element numbers to a single index of a 1D list; used for
    indexing spline functions. Taken directly from pair_meam_spline.h
    
    Args:
        itype (int):
            the i-th element in the system (e.g. in Ti-O, Ti=1)
        jtype (int):
            the j-the element in the system (e.g. in Ti-O, O=2)
        pot_type (str):
            'phi' or 'g'
        ntypes (int):
            the number of unique element types in the system
        
    Returns:
        The mapping of ij into an index of a 1D 0-indexed list"""

    pot_type = pot_type.lower()
    nphi = int((ntypes+1)*ntypes/2)

    if pot_type == 'phi': shift = 0
    elif pot_type == 'g': shift = nphi + int(3*ntypes)
    else: raise ValueError("Invalid choice of pot_type; must be 'phi' or 'g'.")

    return shift + int(jtype - 1 + (itype-1)*ntypes - (itype-1)*itype/2)

def i_to_potl(itype, pot_type, ntypes):
    """Maps element number i to an index of a 1D list; used for indexing spline
    functions. Taken directly from pair_meam_spline.h
    
    Args:
        itype (int):
            the i-the element in the system (e.g. in Ti-O, Ti=1)
        pot_type (str):
            'rho', 'u', or 'f'
        ntypes (int):
            the number of unique element types in the system

    Returns:
        The array index for the given element"""

    pot_type = pot_type.lower()
    nphi = int((ntypes+1)*ntypes/2)

    if pot_type == 'rho': shift = nphi
    elif pot_type == 'u': shift = nphi + ntypes
    elif pot_type == 'f': shift = nphi + 2*ntypes

    return shift + int(itype-1)

def second_derivatives(all_x, all_y, all_d0, all_dN, boundary=None):
    """Computes the second derivatives of the interpolating function at the
    knot points. Follows the algorithm in the classic Numerical Recipes
    textbook.

    Args:
        all_x (np.arr):
            (P,N) array, knot x-positions
        all_y (np.arr):
            (P,N) array, knot y-positions
        all_d0 (np.arr):
            length P array of first derivatives at first knot
        all_dN (np.arr):
            length P array of first derivatives at last knot
        boundary (str):
            the type of boundary conditions to use. 'natural', 'fixed', ...

    Returns:
        y2 (np.arr):
            (P,N) second derivative at knots for every spline"""

    P = len(all_x) # number of splines

    max_nknots = max([len(all_x[j]) for j in range(len(all_x))])

    all_y2 = np.zeros((P,max_nknots))

    for j in range(P):
        x = all_x[j]
        y = all_y[j]
        d0 = all_d0[j]
        dN = all_dN[j]

        N = len(x) # number of knots

        y2  = np.zeros(N)
        u   = np.zeros(N)

        if boundary == 'natural':
            y2[0] = -0.5
        else:
            y2[0] = -0.5
            u[0] = (3.0 / (x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - d0)

        for i in range(1, N-1):
            sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1])
            p = sig*y2[i-1] + 2.0

            y2[i] = (sig - 1.0) / p
            u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1])
            u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1])/p

        # rzm: how to compute y2 for a set of splines simultaneously
        qn = 0.5
        un = (3.0/(x[N-1]-x[N-2])) * (dN - (y[N-1]-y[N-2])/(x[N-1]-x[N-2]))
        y2[N-1] = (un - qn*u[N-2]) / (qn*y2[N-2] + 1.0)

        for k in range(N-2, -1, -1): # loops over [N-2, 0] inclusive
            y2[k] = y2[k]*y2[k+1] + u[k]

        diffx = max_nknots - N
        y2 = np.pad(y2, (0,diffx), 'constant', constant_values=0)

        all_y2[j] = y2

    # TODO: use solve_banded to quickly do tridiag system
    # TODO: fix for y multi-dimensional (multiple potentials at once)
    return all_y2

if __name__ == "__main__":
    import lammpsTools

    p = MEAM('TiO.meam.spline')
    atoms = lammpsTools.atoms_from_file('Ti_only_crowd.Ti', ['Ti'])
    
    logging.info("{0}".format(p.compute_energies(atoms)))
