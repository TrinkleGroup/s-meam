"""Helper functions for reading/writing/manipulating LAMMPS input/output to be
used for spline-MEAM potential fitting

Author: Josh Vita, University of Illinois at Urbana-Champaign
Date:   8/14/17"""

import numpy as np
import glob
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
        fname   -   (str) the name of the input file
        
    Returns:
        phis    -   (list) N pair interaction spline functions
        rhos    -   (list) N atom electronic density spline functions
        us      -   (list) N embedding spline functions
        fs      -   (list) N additional spline functions
        gs      -   (list) N angular term spline functions
        types   -   (list) names of elements in system (e.g. ['Ti', 'O']"""

    print('WARNING: this method may not be up to date!')

    # TODO: convert types to lowercase by default

    with open(fname, 'r') as f:
        f.readline()                    # Remove header
        temp = f.readline().split()     # 'meam/spline ...' line
        types = temp[2:]
        ntypes = len(types)

        nsplines = ntypes*(ntypes+4)    # Based on how fxns in MEAM are defined

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

            # Create a 'natural' spline with endpoint derivatives d0,dN
            splines.append(CubicSpline(xcoords,ycoords,bc_type=((1,d0),(1,dN))))
        print xcoords

    # Calculate the number of splines for phi/g each
    nphi = (ntypes+1)*ntypes/2

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

    return phis, rhos, us, fs, gs, types

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
    
    Args:
        fname   -   (str) the name of the output potential file
        phis    -   (list) N pair interaction spline functions
        rhos    -   (list) N atom electronic density spline functions
        us      -   (list) N embedding spline functions
        fs      -   (list) N additional spline functions
        gs      -   (list) N angular term spline functions
        types   -   (list) names of elements in system (e.g. ['Ti', 'O'])
        
    Returns:
        None; output is a potential file with the given fname"""

    ntypes = len(types)

    with open(fname, 'w') as f:
        # Write header lines
        f.write("# meam/spline potential parameter file produced by Python\n")
        f.write("meam/spline %d %s\n" % (ntypes, " ".join(types)))

        def write_spline(s):
            # See scipy.interpolate.CubicSpline for documentation of fxn object

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
            for i in xrange(nknots):
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
        fname   -   (str) the name of the input file
        style   -   (str) LAMMPS atom_style in input file
        
    Returns:
        data    -   (np.arr) atomic information in atom_style 'style' """

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
        fname   -   (str) the name of the input file
        tilt    -   (bool) True if tilt factor line should be read
        
    Returns:
        dims    -   (tuple) tuple of tuples of (lo,hi) for each dimension
        tlt     -   (tuple) xy,xz,yz"""

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

def atoms_from_lammps_data(fname, types):
    """Creates an ASE Atoms object using the positions and types from a LAMMPS
    data file
    
    Args:
        fname   -   (str) the name of the input file
        types   -   (list) a list of strings specifying atom types; should be
                    ordered to match LAMMPS types (e.g. LAMMPS type 1 == 'Ti',
                    type 2 == 'Au')
                    
    Returns:
        atoms   -   (Atoms) collection of atoms"""

    data = read_data(fname, style='atomic')

    ids = [types[int(x)-1] for x in data[:,1]]  # Convert atom types to symbols
    posx = data[:,2:5]                     # Extract position vectors

    atoms = Atoms(symbols=ids, positions=posx)
    box, tlt = read_box_data(fname,True)

    # Build a right-handed coordinate system; tlt = xy,xz,yz
    a = (box[0][1]-box[0][0],0,0) # vector on x-axis
    b = (tlt[0],box[1][1]-box[1][0],0) # vector in xy-plane
    c = (tlt[1],tlt[2],box[2][1]-box[2][0]) # vector with positive z-component

    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    normc = np.linalg.norm(c)

    # Find angles between basis vectors (degrees)
    theta1 = (180/np.pi)*np.arccos(np.dot(b,c)/normb/normc)
    theta2 = (180/np.pi)*np.arccos(np.dot(a,c)/norma/normc)
    theta3 = (180/np.pi)*np.arccos(np.dot(a,b)/norma/normb)

    atoms.set_cell([norma, normb, normc, theta1, theta2, theta3])

    return atoms

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

def calc_forces(atoms, calc):
    """Calculates the atomic forces using the given ASE calculator.
    
    Args:
        atoms   -   (ase.Atoms) the set of ase.Atom objects defining the system
        calc    -   (ase.calculator) the MD calculator
        
    Returns:
        forces  -   (np.arr) Nx3 array of force vectors"""

    atoms.set_cal
    forces = atoms.get_forces()

def main():

    allFiles = glob.glob('./Database-Structures/crowd_*.Ti')
    #allFiles = ['Database-Structures/crowd.Ti']
    allFiles = ['Database-Structures/force_crowd.Ti']

    types = ['Ti', 'O']
    
    for f in allFiles:
        #print(len(read_data(f, 'atomic')))
        #dims,tilt = read_box_data(f, True)
        #print("Box dimensions = " + str(dims))
        #print("Tilt factors = " + str(tilt))
        #print(len(atoms_from_lammps_data(f, types)))
        w,d = read_forces(f)
        print(w)
        print(d)

if __name__ == '__main__':
    main()
