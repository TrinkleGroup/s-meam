import os, shutil
import numpy as np
import time
import lammpsTools
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS

#fileDir = 'Database-Structures/'
#
## Read/write potential file
#phis, rhos, us, fs, gs, types = lammpsTools.read_spline_meam('TiO.meam.spline')
#lammpsTools.write_spline_meam('poop.meam.spline',phis,rhos,us,fs,gs,types)
#
## Empty tmp folder
#folder = './tmp'
#for the_file in os.listdir(folder):
#    file_path = os.path.join(folder, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#	elif os.path.isdir(file_path): shutil.rmtree(file_path)
#    except Exception as e:
#        print(e)
#
## Input file
#params = {}
#params['boundary'] = 'p p p'
#params['mass'] = ['1 47.867', '2 15.999']
#params['pair_stxle'] = 'meam/spline'
#params['pair_coeff'] = ['* * poop.meam.spline Ti O']
#params['newton'] = 'on'
#params['min_stxle'] = 'cg'
#params['minimize'] = '1.0e-8 1.0e-8 1000 1000'
#
#allFiles = []
#filePairs = []
#expected = []
#
## Read information for force matching
#with open(os.path.join(fileDir, 'FittingDataEnergy.dat')) as f:
#    numStructs = int(f.readline().split()[1])
#
#    i = 0
#    for line in f:
#        if not line.startswith('#'):
#            split = line.split()
#
#            if len(split) == 4:
#                name1 = split[0]
#                name2 = split[3]
#                filePairs.append((name1, name2))
#                allFiles.append(name1)
#                allFiles.append(name2)
#
#            elif len(split) == 1:
#                val = float(line)
#                expected.append(val)
#
#allFiles = set(allFiles)
#
#start = time.time()
#
#energies = {}
#for fname in allFiles:
#    params['read_data'] = fname
#    # Initialize calculator
#    calc = LAMMPS(no_data_file=True, parameters=params,
#            keep_tmp_files=True, specorder=['Ti','O'],
#            tmp_dir=os.path.join('tmp',fname), files=['poop.meam.spline'], keep_alive=True)
#
#    fullPath = os.path.join(fileDir, fname)
#
#    # Read in atoms and box information
#    #atoms = lammpsTools.atoms_from_lammps_data(fullPath, ['Ti', 'O'])
#    #atoms.set_pbc(True)
#
#    #box, tlt = lammpsTools.read_box_data(fullPath, tilt=True)
#
#    # Build a right-handed coordinate system; tlt = xy,xz,yz
#    a = (box[0][1]-box[0][0],0,0) # vector on x-axis
#    b = (tlt[0],box[1][1]-box[1][0],0) # vector in xy-plane
#    c = (tlt[1],tlt[2],box[2][1]-box[2][0]) # vector with positive z-component
#
#    norma = np.linalg.norm(a)
#    normb = np.linalg.norm(b)
#    normc = np.linalg.norm(c)
#
#    # Find angles between basis vectors (degrees)
#    theta1 = (180/np.pi)*np.arccos(np.dot(b,c)/normb/normc)
#    theta2 = (180/np.pi)*np.arccos(np.dot(a,c)/norma/normc)
#    theta3 = (180/np.pi)*np.arccos(np.dot(a,b)/norma/normb)
#
#    # Define unit cell
#    atoms.set_cell([norma, normb, normc, theta1, theta2, theta3])
#    atoms.set_calculator(calc)
#
#    # Calculate energies
#    calc.update(atoms)
#
#    energies[fname] = calc.get_potential_energy(atoms)
#
#end = time.time()

#print('')
#print("'Expected' = difference in energy between structures according to database")
#print("'Actual' = actual difference in energies according to LAMMPS runs")
#
#counter = 0
#differences = []
#print('')
#for i,j in filePairs:
#    val = abs(energies[i] - energies[j])
#    print("Structs: %s %s" % (i,j))
#    print("\tExpected: %f" % expected[counter])
#    print("\tActual: %f" % val)
#
#    diff = val-expected[counter]
#    differences.append(diff)
#    print("\tError: %f" % diff)
#    print('')
#    counter += 1
#
#print("Max error: %f" % max(differences))
#print("Min error: %f" % min(differences))
#print("Mean error: %f" % (sum(differences)/len(differences)))
#print('')
#print("Runtime: %s seconds" % (end-start))

import lammpsTools
from meam import MEAM

atoms = lammpsTools.atoms_from_lammps_data('data.post_min_crowd.Ti', ['Ti'])
p = MEAM('TiO.zeroed.spline')
print("PE = %f" % p.eval(atoms))

splines = [p.phis, p.rhos, p.us, p.fs, p.gs]
splines = [el for grp in splines for el in grp]

plotting = False

if plotting:
    name = "phi_Ti.png"
    xr = [1.5,5.75]
    yr = [-1,4]
    yl = "$\phi_{Ti}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[0].plot(xr,yr,xl,yl,name)

    name = "phi_TiO.png"
    xr = [1.9,5.5]
    yr = [-.6,.6]
    yl = "$\phi_{TiO}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[1].plot(xr,yr,xl,yl,name)

    name = "phi_O.png"
    yl = "$\phi_{O}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[2].plot(yl=yl,xl=xl,saveName=name)

    name = "rho_Ti.png"
    xr = [1.75,5.5]
    yr = [-9,3]
    yl = "$\\rho_{Ti}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[3].plot(xr,yr,xl,yl,name)

    name = "rho_O.png"
    xr = [1.9,5.5]
    yr = [-30,15]
    yl = "$\\rho_{O}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[4].plot(xr,yr,xl,yl,name)

    name = "u_Ti.png"
    xr = [-60,-20]
    yr = [-.5,1]
    yl = "$U_{Ti}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[5].plot(xr,yr,xl,yl,name)

    name = "u_O.png"
    xr = [-25,10]
    yr = [-.15,.35]
    yl = "$U_{O}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[6].plot(xr,yr,xl,yl,name)

    name = "f_Ti.png"
    xr = [1.75,5.5]
    yr = [-2,3]
    yl = "$f_{Ti}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[7].plot(xr,yr,xl,yl,name)

    name = "f_O.png"
    xr = [1.9,5.5]
    yr = [-7.5,12.5]
    yl = "$f_{O}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[8].plot(xr,yr,xl,yl,name)

    name = "g_Ti"
    xr = [-1.25,1.25]
    yr = [-7,2]
    yl = "$g_{Ti}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[9].plot(xr,yr,xl,yl,name)

    name = "g_TiO"
    xr = [-1,1]
    yr = [-.8,.1]
    yl = "$g_{TiO}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[10].plot(xr,yr,xl,yl,name)

    name = "g_O"
    yl = "$g_{O}(r)$ [eV]"
    xl = "r [$\AA$]"
    splines[11].plot(yl=yl,xl=xl,saveName=name)
#
#import timeit
#from spline import Phi
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.interpolate import CubicSpline
#
#xi = np.arange(1,10)
#yi = np.sin(xi)
#x = np.linspace(0,10,100000)
#
#p = Phi(xi,yi,0,0)
#cs = CubicSpline(xi,yi)
#
#def wrapper(func, *args, **kwargs):
#    def wrapped():
#        return func(*args, **kwargs)
#    return wrapped
#
#def fxn1(p, x):
#    map(lambda e: p(e), x)
#
#def fxn2(cs, x):
#    #cs(x)
#    map(lambda e: cs(e),x)
#
#wrapped1 = wrapper(fxn1,p,x)
#wrapped2 = wrapper(fxn2,cs,x)
#
#t1 = float(timeit.timeit(wrapped1,number=1))
#t2 = float(timeit.timeit(wrapped2,number=1))
#
#print("Phi time = %f" % t1)
#print("CubicSpline time = %f" % t2)
#print("Ratio t1/t2 = %f" % (t1/t2))
