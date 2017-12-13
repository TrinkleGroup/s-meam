"""Helper functions for reading/writing/manipulating LAMMPS input/output to be
used for spline-MEAM potential fitting

Author: Josh Vita, University of Illinois at Urbana-Champaign
Date:   8/14/17"""

import numpy as np
import glob
import ase.io
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from scipy.interpolate import CubicSpline
from decimal import Decimal

# LAMMPS atom_style data file formats
STYLE_FORMATS = {'atomic':'%d %d %f %f %f', 'charge':'%d %d %f %f %f %f',
        'full':'%d %d %d %f %f %f %f'}

def read_spline_meam(fname):
    """Builds MEAM potential using spline information from the given file
    
    Args:
        fname (str):
            the name of the input file
        
    Returns:
        knot_x_indices (np.arr):
            ordered list of x-coordinates of all knots

        knot_y_indices (np.arr):
            ordered list of y-coordinates of all knots

        indices (list):
            list of indices deliminating groups of splines"""

    # TODO: convert types to lowercase by default

    try:
        f = open(fname, 'r')

        f.readline()                    # Remove header
        temp = f.readline().split()     # 'meam/spline ...' line
        types = temp[2:]
        ntypes = len(types)

        nsplines = ntypes*(ntypes+4)    # Based on how fxns in MEAM are defined

        # Calculate the number of splines for phi/g each
        nphi = (ntypes + 1) * ntypes / 2

        # Build all splines; separate into different types later
        knot_x_points = []
        knot_y_points = []
        indices     = [] # tracks ends of phi, rho, u, ,f, g groups
        idx         = 0

        for i in range(nsplines):

            if i == nphi:               indices.append(idx)
            elif i == nphi+ntypes:      indices.append(idx)
            elif i == nphi+2*ntypes:    indices.append(idx)
            elif i == nphi+3*ntypes:    indices.append(idx)

            f.readline()                # throwaway 'spline3eq' line
            nknots  = int(f.readline())
            idx     += nknots

            d0, dN = [float(el) for el in f.readline().split()]
            # rzm: do we need to keep d0, dN? also, move this all to worker.py

            for j in range(nknots):
                x,y,y2 = [np.float(el) for el in f.readline().split()]
                knot_x_points.append(x)
                knot_y_points.append(y)

    except IOError as error:
        raise IOError("Could not open potential file: {0}".format(fname))

    return knot_x_points, knot_y_points, indices

def write_spline_meam(fname, phis, rhos, us, fs, gs, types):
    """Writes the splines of a MEAM potential into a LAMMPS style potential file
    for a system with N elements. Potential files have the following format:

        # comment line
        meam/spline <N> <element 1> ... <element N>
        [phis]
        [rhos]
        [us]
        [fs]
        [gs]

        Where each [*] is a block of spline functions with the following format:
            
            spline3eq
            <number of knots>
            <1st derivative @ 1st knot> <1st derivative at last knot>
            <1st knot coordinates>
            .
            .
            .
            <last knot coordinates>

    Each <coordinate> is a set (x,y,2nd deriv of y). See i_to_potl() and
    ij_to_potl for ordering of function blocks.

    See scipy.interpolate.CubicSpline  and spline.py for documentation of Spline
    object
    
    Args:
        fname (str):
            the name of the output potential file
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
            elemental names of system components (e.g. ['Ti', 'O'])
        
    Returns:
        None; output is a potential file with the given fname"""

    ntypes = len(types)

    with open(fname, 'w') as f:
        # Write header lines
        f.write("# meam/spline potential parameter file produced by Python\n")
        f.write("meam/spline %d %s\n" % (ntypes, " ".join(types)))

        def write_spline(s):

            # Write additional spline info
            nknots = len(s.x)
            knotsx = s.x              # x-pos of knots
            knotsy = s(knotsx)             # y-pos of knots
            knotsy2 = s(knotsx,2)          # second derivative at knots
            f.write("spline3eq\n")
            f.write("%d\n" % nknots)

            der = s(knotsx,1)
            d0 = der[0]; dN = der[nknots-1]
            #f.write("%.16f %.16f\n" % (d0,dN))

            # TODO: need to ensure precision isn't changed between read/write

            str1 = ("%.16f" % d0).rstrip('0').rstrip('.')
            str2 = ("%.16f" % dN).rstrip('0').rstrip('.')
            f.write(str1 + ' ' + str2 + '\n')

            # Write knot info
            for i in range(nknots):
                str1 = ("%.16f" % knotsx[i]).rstrip('0').rstrip('.')
                str2 = ("%.16f" % knotsy[i]).rstrip('0').rstrip('.')
                str3 = ("%.16f" % knotsy2[i]).rstrip('0').rstrip('.')
                f.write(str1 + ' ' + str2 + ' ' + str3 + '\n')

        # Output all splines
        for fxn in phis:
            write_spline(fxn)
        for fxn in rhos:
            write_spline(fxn)
        for fxn in us:
            write_spline(fxn)
        for fxn in fs:
            write_spline(fxn)
        for fxn in gs:
            write_spline(fxn)

def read_data(fname, style):
    """Reads in atomic information from a LAMMPS data file, where the
    information in the data file is written with atom_style 'style'
    
    Args:
        fname (str):
            the name of the input file
        style (str):
            LAMMPS atom_style in input file
        
    Returns:
        data (np.arr):
            atomic information in atom_style style"""

    headerSize = 0
    with open(fname, 'r') as f:
        line = " "

        while line and ('Atoms' not in line):
            headerSize += 1
            line = f.readline()

        headerSize += 1 # Catches blank line after 'Atoms' OR first comment line

    return np.genfromtxt(fname, skip_header=headerSize)

def read_box_data(fname, tilt):
    """Reads in simulation box information from a LAMMPS data file.
    
    Args:
        fname (str):
            the name of the input file
        filt (bool):
            True if a tilt factor line should be read
        
    Returns:
        dims (tuple):
            tuple of (lo,hi) for each dimension
        tlt (tuple):
            xy,xz,yz"""

    with open(fname, 'r') as f:
        line = f.readline()

        xs = False; ys = False; zs = False

        # Loop until all three dimensions have been read in
        while ((not xs) or (not ys) or (not zs)):
            if 'xlo' in line:
                temp = line.split()
                xlo, xhi = temp[0:2]
                xlo = float(xlo); xhi = float(xhi)
                xs = True
            elif 'ylo' in line:
                temp = line.split()
                ylo, yhi = temp[0:2]
                ylo = float(ylo); yhi = float(yhi)
                ys = True
            elif 'zlo' in line:
                temp = line.split()
                zlo, zhi = temp[0:2]
                zlo = float(zlo); zhi = float(zhi)
                zs = True

            line = f.readline()

        # Read tilt factors if needed
        # TODO: Infinite looping if orthogonal box
        if tilt:
            tlt = None

            while not tlt:
                if 'xy' in line:
                    temp = line.split()
                    xy, xz, yz = temp[0:3]
                    xy = float(xy); xz = float(xz); yz = float(yz)
                    tlt = (xy,xz,yz)

                line = f.readline()

    dims = ((xlo,xhi), (ylo,yhi), (zlo,zhi))

    if tilt:
        return dims, tlt
    else:
        return dims

def atoms_from_file(fname, types, fmt='lammps-data', style='atomic', pbc=True):
    """Wrapper for ase.io.read() function that also sets chemical symbols.
    Creates an ASE Atoms object using the positions and types from a LAMMPS
    data file.
    
    Args:
        fname (str):
            the name of the input file
        types (list):
            a list of strings specifying atom types; should be ordered to match
            LAMMPS types (e.g. LAMMPS type 1 == 'Ti', type 2 == 'Au')
        fmt (str):
            ASE data file format. Default 'lammps-data'
        style (str):
            LAMMPS atom_style. Default 'atomic'
        pbc (bool):
            Boolean or len()==3 list of booleans defining periodic boundary
            conditions along each cell basis vector
                    
    Returns:
        atoms (Atoms):
            collection of atoms"""

    # TODO: use **kwargs, like in ase.io.read()
    atoms = ase.io.read(fname, format=fmt, style=style)
    atoms.set_tags(range(1,len(atoms)+1))
    atoms.set_chemical_symbols([types[i-1] for i in atoms.get_atomic_numbers()])
    atoms.set_pbc(pbc)

    return atoms

def atoms_to_LAMMPS_file(fname, atoms):
    """Writes atoms to a LAMMPS style data file. Assumes box starts at origin"""

    if not atoms.get_cell().any():
        raise AttributeError("Must specify cell size")

    types = atoms.get_chemical_symbols()
    types_idx = range(1, len(types)+1)
    p = atoms.positions

    # Converts cell vectors to LAMMPS format
    print(atoms.get_cell())
    a,b,c = atoms.get_cell()
    x = a[0]; y = b[1]; z = c[2]
    xy = b[0]; xz = c[0]; yz = c[1]

    with open(fname, 'w') as f:
        f.write('Written using lammpsTools.py\n\n')

        f.write('%d atoms\n' % len(atoms))
        f.write('%d atom types\n\n' % len(set(types)))

        f.write("0.0 %.16f xlo xhi\n" % x)
        f.write("0.0 %.16f ylo yhi\n" % y)
        f.write("0.0 %.16f zlo zhi\n" % z)
        f.write("%.16f %.16f %.16f xy xz yz\n\n" % (xy,xz,yz))

        f.write("Atoms\n\n")

        for i in range(len(types_idx)):
            f.write("%d %d %.16f %.16f %.16f\n" % (i+1,\
                symbol_to_type(types[i],types),p[i][0],p[i][1],p[i][2]))

def read_forces(fname):
    """Reads in the atomic forces from a file with the following format:
    <file name>
    w1 Fx1 Fy1 Fz1
    w2 Fx2 Fy2 Fz2
    .
    .
    .
    
    where each row is the force on the respective atom and w_i is the force
    weight used for potential fitting, either 0 or 1 determined by the cutoff
    potential.

    Args:
        fname   -   (str) the name of the input file

    Returns:
        w       -   (np.arr) Nx1 weighting vector
        data    -   (np.arr) Nx3 array of force vectors
    """

    data = np.genfromtxt(fname, skip_header=1)

    return data[:,0], data[:,1:]    # w, data

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

def main():

    allFiles = glob.glob('./Database-Structures/crowd_*.Ti')
    allFiles = ['Database-Structures/force_crowd.Ti']

    types = ['Ti', 'O']
    
    for f in allFiles:
        w,d = read_forces(f)
        print(w)
        print(d)

if __name__ == '__main__':
    main()
