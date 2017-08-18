import os, shutil
import numpy as np
import time
import lammpsTools
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS

fileDir = 'Database-Structures/'

# Read/write potential file
phis, rhos, us, fs, gs, types = lammpsTools.read_spline_meam('TiO.meam.spline')
lammpsTools.write_spline_meam('poop.meam.spline',phis,rhos,us,fs,gs,types)

# Empty tmp folder
folder = './tmp'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
	elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

# Input file
params = {}
params['boundary'] = 'p p p'
params['mass'] = ['1 47.867', '2 15.999']
params['pair_style'] = 'meam/spline'
params['pair_coeff'] = ['* * poop.meam.spline Ti O']
params['newton'] = 'on'
params['min_style'] = 'cg'
params['minimize'] = '1.0e-8 1.0e-8 1000 1000'

allFiles = []
filePairs = []
expected = []

with open(os.path.join(fileDir, 'FittingDataEnergy.dat')) as f:
    numStructs = int(f.readline().split()[1])

    i = 0
    for line in f:
        if not line.startswith('#'):
            split = line.split()

            if len(split) == 4:
                name1 = split[0]
                name2 = split[3]
                filePairs.append((name1, name2))
                allFiles.append(name1)
                allFiles.append(name2)

            elif len(split) == 1:
                val = float(line)
                expected.append(val)

allFiles = set(allFiles)

start = time.time()

energies = {}
for fname in allFiles:
    # Initialize calculator
    calc = LAMMPS(no_data_file=True, parameters=params,
            keep_tmp_files=True, specorder=['Ti','O'],
            tmp_dir=os.path.join('tmp',fname), files=['poop.meam.spline'], keep_alive=True)

    fullPath = os.path.join(fileDir, fname)

    # Read in atoms and box information
    atoms = lammpsTools.atoms_from_lammps_data(fullPath, ['Ti', 'O'])
    atoms.set_pbc(True)

    box, tlt = lammpsTools.read_box_data(fullPath, tilt=True)

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

    # Define unit cell
    atoms.set_cell([norma, normb, normc, theta1, theta2, theta3])
    atoms.set_calculator(calc)

    # Calculate energies
    calc.update(atoms)

    energies[fname] = calc.get_potential_energy(atoms)

end = time.time()

print('')
print("'Expected' = difference in energy between structures according to database")
print("'Actual' = actual difference in energies according to LAMMPS runs")

counter = 0
differences = []
print('')
for i,j in filePairs:
    val = abs(energies[i] - energies[j])
    print("Structs: %s %s" % (i,j))
    print("\tExpected: %f" % expected[counter])
    print("\tActual: %f" % val)

    diff = val-expected[counter]
    differences.append(diff)
    print("\tError: %f" % diff)
    print('')
    counter += 1 

print("Max error: %f" % max(differences))
print("Min error: %f" % min(differences))
print("Mean error: %f" % (sum(differences)/len(differences)))
print('')
print("Runtime: %s seconds" % (end-start))
