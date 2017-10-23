from nose_parameterized import parameterized
import numpy as np
import time
import pickle
import os
import logging
import meam

import lammpsTools

np.random.seed(42)

from workers import WorkerManyPotentialsOneStruct as worker
from ase.calculators.lammpsrun import LAMMPS
from .structs import allstructs
from .globalVars import ATOL

################################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def loader(group_name, calculated, lammps):
    """Assumes 'lammps' and 'calculated' are dictionaries where key:value =
    <struct name>:<list of calculations> where each entry in the list of
    calculations corresponds to a single potential."""

    tests = []
    for name in calculated.keys():
        test_name = group_name + '_' + name
        tests.append((test_name, calculated[name], lammps[name]))

    return tests

def runner(pots, structs):

    calculated = {}
    for name in structs.keys():
        atoms = structs[name]

        w = worker(atoms,pots)
        start = time.time()
        calculated[name] = w.compute_energies()
        py_calcduration += time.time() - start

    return calculated

def getLammpsResults(pots, structs):

    start = time.time()

    # Builds dictionaries where dict[i]['ptype'][j] is the energy or force matrix of
    # struct i using potential j for the given 'ptype', according to LAMMPS
    energies = {}
    forces = {}

    types = ['H','He']
    #types = ['He','H']

    params = {}
    params['units'] = 'metal'
    params['boundary'] = 'p p p'
    params['mass'] =  ['1 1.008', '2 4.0026']
    params['pair_style'] = 'meam/spline'
    params['pair_coeff'] = ['* * test.meam.spline ' + ' '.join(types)]
    params['newton'] = 'on'

    calcduration = 0

    for key in structs.keys():
        energies[key] = np.zeros(len(pots))

    for pnum,p in enumerate(pots):

        p.write_to_file('test.meam.spline')

        calc = LAMMPS(no_data_file=True, parameters=params, \
                      keep_tmp_files=True,specorder=types,files=['test.meam.spline'])

        for name in structs.keys():
            atoms = structs[name]

            cstart = time.time()
            energies[name][pnum] = calc.get_potential_energy(atoms)
            calcduration += float(time.time() - cstart)

            # TODO: LAMMPS runtimes are inflated due to ASE internal read/write
            # TODO: need to create shell script to get actual runtimes

        calc.clean()
        #os.remove('test.meam.spline')

    #logging.info("Time spent writing potentials: {}s".format(round(writeduration,3)))
    logging.info("Time spent calculating in LAMMPS: {}s".format(round(calcduration,3)))

    return calcduration, energies, forces

################################################################################
#from .potentials import zero_potential as p
#
#_, energies, _ = getLammpsResults([p], allstructs)
#
#calculated = runner([p], allstructs)
#
#@parameterized.expand(loader('', calculated, energies))
#def test_zero_potential(name, a, b):
#    np.testing.assert_allclose(a,b,atol=ATOL)

    #str = 'lammps = {0}\npython = {1}'.format(a,b)
    #logging.info(str)

################################################################################
"""Constant potentials"""

from .potentials import constant_potential as p

py_calcduration = 0
#rzm: Timing tests

"""meam subtype"""
_, energies, _ = getLammpsResults([p], allstructs)

#logging.info("meam subtype")
calculated = runner([p], allstructs)

@parameterized.expand(loader('', calculated, energies))
def test_constant_potential_meam(name, a, b):
    np.testing.assert_allclose(a,b,atol=ATOL)

    str = 'lammps = {0} python = {1}'.format(a,b)
    #logging.info(str)

"""phionly subtype"""
#meam.phionly_subtype(p).write_to_file('test.phionly.poop.spline')
_, energies, _ = getLammpsResults([meam.phionly_subtype(p)], allstructs)

#meam.nophi_subtype(p).write_to_file('test.poop.spline')
meam.nophi_subtype(p).plot()
#p.plot()
#lammpsTools.atoms_to_LAMMPS_file('data.pooper.lammps', allstructs[list(allstructs.keys())[0]])
#
#logging.info("phionly subtype")
calculated = runner([meam.phionly_subtype(p)], allstructs)

@parameterized.expand(loader('', calculated, energies))
def test_constant_potential_phionly(name, a, b):
    np.testing.assert_allclose(a,b,atol=ATOL)

    str = 'lammps = {0} python = {1}'.format(a,b)
    #logging.info(str)

"""rhophi subtype"""
_, energies, _ = getLammpsResults([meam.rhophi_subtype(p)], allstructs)

#logging.info("rhophi subtype")
calculated = runner([meam.rhophi_subtype(p)], allstructs)

@parameterized.expand(loader('', calculated, energies))
def test_constant_potential_rhophi(name, a, b):
    np.testing.assert_allclose(a,b,atol=ATOL)

    str = 'lammps = {0} python = {1}'.format(a,b)
    #logging.info(str)

"""nophi subtype"""
_, energies, _ = getLammpsResults([meam.nophi_subtype(p)], allstructs)

#logging.info("nophi subtype")
calculated = runner([meam.nophi_subtype(p)], allstructs)

@parameterized.expand(loader('', calculated, energies))
def test_constant_potential_nophi(name, a, b):
    np.testing.assert_allclose(a,b,atol=ATOL)

    str = 'lammps = {0} python = {1}'.format(a,b)
    #logging.info(str)

"""rho subtype"""
_, energies, _ = getLammpsResults([meam.rho_subtype(p)], allstructs)

#logging.info("rho subtype")
calculated = runner([meam.rho_subtype(p)], allstructs)

@parameterized.expand(loader('', calculated, energies))
def test_constant_potential_rho(name, a, b):
    np.testing.assert_allclose(a,b,atol=ATOL)

    str = 'lammps = {0} python = {1}'.format(a,b)
    #logging.info(str)

"""norho subtype"""
_, energies, _ = getLammpsResults([meam.norho_subtype(p)], allstructs)

#logging.info("norho subtype")
calculated = runner([meam.norho_subtype(p)], allstructs)

@parameterized.expand(loader('', calculated, energies))
def test_constant_potential_norho(name, a, b):
    np.testing.assert_allclose(a,b,atol=ATOL)

    str = 'lammps = {0} python = {1}'.format(a,b)
    #logging.info(str)

"""norhophi subtype"""
_, energies, _ = getLammpsResults([meam.norhophi_subtype(p)], allstructs)

#logging.info("norhophi subtype")
calculated = runner([meam.norhophi_subtype(p)], allstructs)
meam.norhophi_subtype(p).plot()

@parameterized.expand(loader('', calculated, energies))
def test_constant_potential_norhophi(name, a, b):
    np.testing.assert_allclose(a,b,atol=ATOL)

    str = 'lammps = {0} python = {1}'.format(a,b)
    #logging.info(str)

logging.info("Time spent calculating in Python: {}s".format(round(
    py_calcduration,3)))

################################################################################

def load_energies():
    if os.path.isfile('lammps_energies.dat'):
        # TODO: check if energies matches number of potentials; else rebuild
        duration, energies = pickle.load(open('lammps_energies.dat', 'rb'))
        #cls.forces = pickle.load(open('lammps_forces.dat', 'r'))
    else:
        duration, energies,_  = getLammpsResults()

        pickle.dump((duration, energies), open('lammps_energies.dat', 'wb'))
        #pickle.dump(cls.forces, open('lammps_forces.dat', 'w'))

    #logger.info('LAMMPS build time (accounting for file writing time):({'
    #            '}s)'.format(round(duration,3)))

    return energies
