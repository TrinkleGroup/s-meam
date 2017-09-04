import os, shutil
import numpy as np
import time
import lammpsTools
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from meam import MEAM

atoms = lammpsTools.atoms_from_lammps_data('data.post_min_crowd.Ti', ['Ti'])
p = MEAM('TiO.meam.spline')
print("PE = %f" % p.eval(atoms))

splines = [p.phis, p.rhos, p.us, p.fs, p.gs]
splines = [el for grp in splines for el in grp]

plotting = True

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
