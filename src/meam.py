#TODO: will need to be able to set individual splines, not just the whole set

import numpy as np
import logging

from spline import Spline, ZeroSpline
from ase.neighborlist import NeighborList

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: assume that an x_pvec does NOTTTTTTT have bc in it. should be in y_pvec

class MEAM:
    """An object for organizing the information needed for a spline meam
    potential in various different representations

    Properties:
        types (list):
            list of str representations of unique elements in the system (
            e.g. ['H', 'He'])
        ntypes (int):
            len(types)
        nphi (int):
            the number of bond types in the system (e.g. H-H, H-He, He-He)

            nphi = (ntypes+1)*ntypes/2)

        phis (list):
            pair interaction splines, nphi in total
        rhos (list):
            electron density splines, ntypes in total
        us (list):
            full embedding energy splines, ntypes in total
        fs (list):
            three-body correction splines, ntypes in total
        gs (list):
            angular contribution splines, nphi in total"""

    def __init__(self, splines, types):
        """Base constructor"""

        ntypes = len(types)

        if ntypes < 1:
            raise ValueError("must specify at least one atom type")
        elif ntypes > 2:
            raise NotImplementedError("only unary and binary systems are supported")
        elif len(splines) != ntypes*(ntypes+4):
            raise ValueError("incorrect number of splines for given number of"
                             "atom types")

        self.types = types
        self.ntypes = ntypes
        nphi = int((ntypes+1)*ntypes/2)
        self.nphi = nphi

        split_indices = [nphi, nphi+ntypes, nphi+2*ntypes, nphi+3*ntypes]
        self.phis, self.rhos, self.us, self.fs, self.gs = np.split(splines,
                                                                split_indices)

        self.phis = list(self.phis)
        self.rhos = list(self.rhos)
        self.us = list(self.us)
        self.fs = list(self.fs)
        self.gs = list(self.gs)

        self.splines = self.phis + self.rhos + self.us + self.fs + self.gs

        radials = [self.phis,self.rhos,self.fs]
        endpoints = [max(fxn.x) for ftype in radials for fxn in ftype]
        self.cutoff = max(endpoints)

        self.zero_atom_energies =  [None]*self.ntypes
        for i in range(self.ntypes):
            self.zero_atom_energies[i] = self.us[i](0.0)

        self.uprimes = None
        self.forces = None
        self.energies = None

    @classmethod
    def from_pvec(cls, x_pvec, y_pvec, x_indices, types):
        """Builds a MEAM object from 1D vectors of parameters.

        Args:
            see WorkerManyPotentialsOneStruct.__init()__ for details
            concerning formatting of input arguments

        Returns:
            MEAM object"""

        x_pvec = np.array(x_pvec)
        y_pvec = np.array(y_pvec)

        x_indices = np.array(x_indices)
        y_indices = [x_indices[i-1]-2*i for i in range(1, len(x_indices)+1)]

        split_x = np.split(x_pvec, x_indices)
        split_y = np.split(y_pvec, y_indices)

        nsplines = len(types)*(len(types)+4)
        splines = [None]*nsplines

        for i in range(len(split_x)):
            x, bc = np.split(split_x[i], [-2])

            splines[i] = Spline(x, split_y[i], end_derivs=bc)

        return cls(splines, types)

    @classmethod
    def  from_file(cls, fname):
        """Builds MEAM potential using spline information from the given file

        Args:
            fname (str):
                the name of the input file

        Returns:
            MEAM object"""

        with open(fname, 'r') as f:
            f.readline()  # Remove header
            temp = f.readline().split()  # 'meam/spline ...' line
            types = temp[2:]
            ntypes = len(types)

            # Based on how fxns in MEAM are defined
            nsplines = ntypes * (ntypes + 4)

            # Build all splines; separate into different types later
            splines = []
            for i in range(nsplines):
                f.readline()  # throwaway 'spline3eq' line
                nknots = int(f.readline())

                d0, dN = [float(el) for el in f.readline().split()]

                xcoords = []  # x-coordinates of knots
                ycoords = []  # y-coordinates of knots
                for j in range(nknots):
                    # still unsure why y2 is in the file... why store if it's
                    # recalculated again later??
                    x, y, y2 = [np.float(el) for el in f.readline().split()]
                    xcoords.append(x)
                    ycoords.append(y)

                splines.append(Spline(xcoords, ycoords, end_derivs=(d0,dN)))

        return cls(splines, types)

    def write_to_file(self, fname):
        """Writes the potential to a file

        Args:
            fname (str):
                name of file to write to

        Returns:
            None; writes to file with name <fname>"""

        types = self.types
        ntypes = len(types)

        with open(fname, 'w') as f:
            # Write header lines
            f.write("# meam/spline potential parameter file produced by MEAM object\n")
            f.write("meam/spline %d %s\n" % (ntypes, " ".join(types)))

            def write_spline(s):

                # Write additional spline info
                nknots = len(s.x)
                f.write("spline3eq\n")
                f.write("%d\n" % nknots)

                der = s(s.x, 1)
                d0 = der[0]; dN = der[nknots-1]

                str1 = ("%.16f" % d0).rstrip('0').rstrip('.')
                str2 = ("%.16f" % dN).rstrip('0').rstrip('.')
                f.write(str1 + ' ' + str2 + '\n')

                # Write knot info
                for i in range(nknots):
                    str1 = ("%.16f" % s.x[i]).rstrip('0').rstrip('.')
                    str2 = ("%.16f" % s(s.x)[i]).rstrip('0').rstrip('.')
                    str3 = ("%.16f" % s(s.x, 2)[i]).rstrip('0').rstrip('.')
                    f.write(str1 + ' ' + str2 + ' ' + str3 + '\n')

            # Output all splines
            for fxn in self.phis:
                write_spline(fxn)
            for fxn in self.rhos:
                write_spline(fxn)
            for fxn in self.us:
                write_spline(fxn)
            for fxn in self.fs:
                write_spline(fxn)
            for fxn in self.gs:
                write_spline(fxn)

    def plot(self, fname=None):
        """Generates plots of all splines"""

        splines = self.phis + self.rhos + self.us + self.fs + self.gs

        for i,s in enumerate(splines):
            s.plot(saveName=fname+str(i+1)+'.png')
                # TODO: finish this for generic system, not just binary/unary

    def phionly_subtype(pot):
        """Returns the phionly version of pot

        Args:
            pot (MEAM):
                the original potential

        Returns:
            phionly (MEAM):
                the phionly version of pot"""

        N = pot.ntypes          # number of components in the system
        nphi = int((N+1)*N/2)   # number of each phi and u splines

        original = pot.phis + pot.rhos + pot.us + pot.fs + pot.gs

        splines = [original[i] if (i<nphi) else ZeroSpline(original[i].x)
                   for i in range(N*(N+4))]

        return MEAM(splines=splines, types=pot.types)

    def nophi_subtype(pot):
        """Returns the nophi version of pot

        Args:
            pot (MEAM):
                the original potential

        Returns:
            nophi (MEAM):
                the nophi version of pot"""

        N = pot.ntypes          # number of components in the system
        nphi = int((N+1)*N/2)   # number of each phi and u splines

        original = pot.phis + pot.rhos + pot.us + pot.fs + pot.gs

        splines = [original[i] if (i>=nphi) else ZeroSpline(original[i].x)
                   for i in range(N*(N+4))]

        return MEAM(splines=splines, types=pot.types)

    def rhophi_subtype(pot):
        """Returns the rhophi version of pot

        Args:
            pot (MEAM):
                the original potential

        Returns:
            rhophi (MEAM):
                the rhophi version of pot"""

        N = pot.ntypes          # number of components in the system
        nphi = int((N+1)*N/2)   # number of each phi and u splines

        original = pot.phis + pot.rhos + pot.us + pot.fs + pot.gs

        splines = [original[i] if (i<(nphi+N+N)) else ZeroSpline(original[
                                            i].x) for i in range(N*(N+4))]

        return MEAM(splines=splines, types=pot.types)

    def norhophi_subtype(pot):
        """Returns the norhophi version of pot

        Args:
            pot (MEAM):
                the original potential

        Returns:
            norhophi (MEAM):
                the norhophi version of pot"""

        N = pot.ntypes          # number of components in the system
        nphi = int((N+1)*N/2)   # number of each phi and u splines

        original = pot.phis + pot.rhos + pot.us + pot.fs + pot.gs

        splines = [original[i] if (i>=(nphi+N)) else ZeroSpline(original[
                                            i].x) for i in range(N*(N+4))]

        return MEAM(splines=splines, types=pot.types)

    def norho_subtype(pot):
        """Returns the norhos version of pot

        Args:
            pot (MEAM):
                the original potential

        Returns:
            norhos (MEAM):
                the norhos version of pot"""

        N = pot.ntypes          # number of components in the system
        nphi = int((N+1)*N/2)   # number of each phi and u splines

        original = pot.phis + pot.rhos + pot.us + pot.fs + pot.gs

        splines = [original[i] if ((i<nphi) or (i>=(nphi+N))) else ZeroSpline(
                        original[i].x) for i in range(N*(N+4))]

        return MEAM(splines=splines, types=pot.types)

    def rho_subtype(pot):
        """Returns the rhos version of pot

        Args:
            pot (MEAM):
                the original potential

        Returns:
            rhos (MEAM):
                the rhos version of pot"""

        N = pot.ntypes          # number of components in the system
        nphi = int((N+1)*N/2)   # number of each phi and u splines

        original = pot.phis + pot.rhos + pot.us + pot.fs + pot.gs

        splines = [original[i] if ((i>=nphi) and (i<nphi+N+N)) else ZeroSpline(
                        original[i].x) for i in range(N*(N+4))]

        return MEAM(splines=splines, types=pot.types)

    def nog_subtype(pot):
        # TODO: need to set g=1
        """Returns the nog version of pot

        Args:
            pot (MEAM):
                the original potential

        Returns:
            nog (MEAM):
                the nog version of pot"""

        N = pot.ntypes          # number of components in the system
        nphi = int((N+1)*N/2)   # number of each phi and u splines

        original = pot.phis + pot.rhos + pot.us + pot.fs + pot.gs

        splines = [original[i] if ((i>=nphi) and (i<nphi+N+N)) else ZeroSpline(
                        original[i].x) for i in range(N*(N+4))]

        return MEAM(splines=splines, types=pot.types)

def ij_to_potl(itype, jtype, ntypes):
    """Maps i and j element numbers to a single index of a 1D list; used for
    indexing spline functions. Currently only works for binary systems;
    ternary systems causes overlap with current algorithm (e.g. 2-1 bond maps to
    same as 1-3 bond)
    
    Args:
        itype (int):
            the i-th element in the system (e.g. in Ti-O, Ti=1)
        jtype (int):
            the j-the element in the system (e.g. in Ti-O, O=2)
        ntypes (int):
            the number of unique element types in the system
        
    Returns:
        The mapping of ij into an index of a 1D 0-indexed list"""

    if (itype<1) or (jtype<1):
        raise ValueError("atom types must be positive and non-zero")
    elif ntypes != 2:
        # remove the unit test in meamTests.py once you implement this
        raise NotImplementedError("currently, only binary systems are "
                                  "supported")
    else:
        return int(jtype - 1 + (itype-1)*ntypes - (itype-1)*itype/2)

def i_to_potl(itype):
    """Maps element number i to an index of a 1D list; used for indexing spline
    functions. Taken directly from pair_meam_spline.h
    
    Args:
        itype (int):
            the i-the element in the system (e.g. in Ti-O, Ti=1)
        
    Returns:
        The array index for the given element"""

    if itype < 1:
        raise ValueError("atom types must be positive and non-zero")
    else:
        return int(itype-1)

def splines_to_pvec(splines):
    """Converts a list of splines into a large 1D vector of parameters.
    Spline ordering is unchanged.

    Note:
        this formatting for the output was chosen to correspond with how a
        Worker object is built and evaluated

    Args:
        splines (list [Spline]):
            an ordered list of Spline objects

    Returns:
        x_pvec (np.arr):
            a group of parameters corresponding to a single spline consists
            of the x-positions of all knot points

        y_pvec (np.arr):
            y-positions of all knot points plus two additional boundary
            conditions

        x_indices (list [int]):
            indices deliminating spline knots"""

    x_pvec = np.array([])
    y_pvec = np.array([])
    x_indices = []

    idx_tracker = 0
    for s in splines:
        x_pvec = np.append(x_pvec, s.x)
        y_pvec = np.append(y_pvec, s(s.x))

        der = s(s.x, 1)
        bc = [der[0], der[-1]]

        y_pvec = np.append(y_pvec, bc)

        x_indices.append(idx_tracker)
        idx_tracker += len(s.x)

    return x_pvec, y_pvec, x_indices

def splines_from_pvec(x_pvec, y_pvec, x_indices):
    """Builds splines out of the given knot coordinates and boundary
    conditions.

    Note:
        this formatting for the output was chosen to correspond with how a
        Worker object is built and evaluated

    Args:
        x_pvec (np.arr):
            a group of parameters corresponding to a single spline consists
            of the x-positions of all knot points as well as two additional
            boundary conditions

        y_pvec (np.arr):
            y-positions of all knot points

        x_indices (list [int]):
            indices deliminating spline parameters in vector

    Returns:
        splines (list [Spline]):
            an ordered list of Spline objects"""

    x_indices = np.array(x_indices)
    y_indices = [x_indices[i-1]-2*i for i in range(1, len(x_indices)+1)]
    y_indices = np.array(y_indices)

    x_split = np.split(x_pvec, x_indices)
    y_split = np.split(y_pvec, y_indices)

    splines = []

    for i in range(len(x_split)):
        x_knots, bc = np.split(x_split[i], [-2])
        y_knots = y_split[i]

        splines.append(Spline(x_knots, y_knots, end_derivs=bc))

    return splines

if __name__ == "__main__":
    import lammpsTools

    p = MEAM('TiO.meam.spline')
    atoms = lammpsTools.atoms_from_file('Ti_only_crowd.Ti', ['Ti'])
    
    #logging.info("{0}".format(p.compute_energies_self(atoms)))