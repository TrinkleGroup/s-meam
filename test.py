import os, shutil
import numpy as np
import time
import lammpsTools
import ase.io
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from meam import MEAM

types = ['Ti','O']

atoms = ase.io.read("test-files/data.trimer.Ti", format="lammps-data",\
        style="atomic")
atoms.set_chemical_symbols([types[i-1] for i in atoms.get_atomic_numbers()])
p = MEAM('test-files/TiO.nophi.spline')
#print("PE = %.16f" % p.compute_energies(atoms))
print(p.compute_forces(atoms))

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
