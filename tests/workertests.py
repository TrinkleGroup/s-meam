import nose
import numpy as np
import time
import pickle
import os
import logging
import meam
import sys
import lammpsTools
import tests.potentials

np.random.seed(42)

from nose_parameterized import parameterized
from workers import WorkerManyPotentialsOneStruct as worker
from ase.calculators.lammpsrun import LAMMPS
from tests.structs import dimers, trimers, bulk_vac_ortho, \
    bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo
from tests.globalVars import ATOL

N = 7

zero_pots_flag  = False
const_pots_flag = False
rand_pots_flag  = True

meam_flag       = True
phionly_flag    = False
rhophi_flag     = False
nophi_flag      = False
rho_flag        = False
norho_flag      = False
norhophi_flag   = False

dimers_flag  = False
trimers_flag = True
bulk_flag    = False

allstructs = {}

if dimers_flag:
    allstructs = {**allstructs, **dimers}
if trimers_flag:
    allstructs = {**allstructs, **trimers}
if bulk_flag:
    allstructs = {**allstructs, **bulk_vac_ortho, **bulk_periodic_ortho,
                  **bulk_vac_rhombo, **bulk_periodic_rhombo}

################################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.terminator = ""

py_calcduration = 0
lammps_calcduration = 0

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

        global py_calcduration

        w = worker(atoms,pots)
        start = time.time()
        #calculated[name] = w.compute_energies()
        # TODO: to optimize, preserve workers for each struct
        calculated[name] = w(pots)
        #logging.info("{0}".format(calculated[name]))
        py_calcduration += time.time() - start

    return calculated

def getLammpsResults(pots, structs):

    start = time.time()

    # Builds dictionaries where dict[i]['ptype'][j] is the energy or force matrix of
    # struct i using potential j for the given 'ptype', according to LAMMPS
    energies = {}
    forces = {}

    types = ['H','He']

    params = {}
    params['units'] = 'metal'
    params['boundary'] = 'p p p'
    params['mass'] =  ['1 1.008', '2 4.0026']
    params['pair_style'] = 'meam/spline'
    params['pair_coeff'] = ['* * test.meam.spline ' + ' '.join(types)]
    params['newton'] = 'on'

    global lammps_calcduration

    for key in structs.keys():
        energies[key] = np.zeros(len(pots))

    for pnum,p in enumerate(pots):

        p.write_to_file('test.meam.spline')

        calc = LAMMPS(no_data_file=True, parameters=params, \
                      keep_tmp_files=False,specorder=types,files=['test.meam.spline'])

        for name in structs.keys():
            atoms = structs[name]

            cstart = time.time()
            energies[name][pnum] = calc.get_potential_energy(atoms)
            lammps_calcduration += float(time.time() - cstart)

            # TODO: LAMMPS runtimes are inflated due to ASE internal read/write
            # TODO: need to create shell script to get actual runtimes

        calc.clean()
        #os.remove('test.meam.spline')

    #logging.info("Time spent writing potentials: {}s".format(round(writeduration,3)))
    #logging.info("Time spent calculating in LAMMPS: {}s".format(round(
    # calcduration,3)))

    return energies, forces

################################################################################
if zero_pots_flag:
    """Zero potentials"""
    p = tests.potentials.get_zero_potential()

    energies, _ = getLammpsResults([p], allstructs)

    calculated = runner([p], allstructs)

    @parameterized.expand(loader('', calculated, energies))
    def test_zero_potential(name, a, b):
        np.testing.assert_allclose(a,b,atol=ATOL)
################################################################################
if const_pots_flag:
    """Constant potentials"""

    p = tests.potentials.get_constant_potential()

    if meam_flag:
        """meam subtype"""
        energies, _ = getLammpsResults([p], allstructs)

        calculated = runner([p], allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_constant_potential_meam(name, a, b):
           np.testing.assert_allclose(a,b,atol=ATOL)

    if phionly_flag:
        """phionly subtype"""
        energies, _ = getLammpsResults([meam.phionly_subtype(p)], allstructs)

        calculated = runner([meam.phionly_subtype(p)], allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_constant_potential_phionly(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)

    if rhophi_flag:
        """rhophi subtype"""
        energies, _ = getLammpsResults([meam.rhophi_subtype(p)], allstructs)

        calculated = runner([meam.rhophi_subtype(p)], allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_constant_potential_rhophi(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)

    if nophi_flag:
        """nophi subtype"""
        energies, _ = getLammpsResults([meam.nophi_subtype(p)], allstructs)

        calculated = runner([meam.nophi_subtype(p)], allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_constant_potential_nophi(name, a, b):
           np.testing.assert_allclose(a,b,atol=ATOL)

    if rho_flag:
        """rho subtype"""
        energies, _ = getLammpsResults([meam.rho_subtype(p)], allstructs)

        calculated = runner([meam.rho_subtype(p)], allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_constant_potential_rho(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)

    if norho_flag:
        """norho subtype"""
        energies, _ = getLammpsResults([meam.norho_subtype(p)], allstructs)

        calculated = runner([meam.norho_subtype(p)], allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_constant_potential_norho(name, a, b):
           np.testing.assert_allclose(a,b,atol=ATOL)

    if norhophi_flag:
        """norhophi subtype"""
        energies, _ = getLammpsResults([meam.norhophi_subtype(p)], allstructs)

        calculated = runner([meam.norhophi_subtype(p)], allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_constant_potential_norhophi(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)
#################################################################################
if rand_pots_flag:
    """Random potentials"""

    if meam_flag:
        """meam subtype"""
        p = tests.potentials.get_random_pots(N)['meams']

        energies, _ = getLammpsResults(p, allstructs)

        calculated = runner(p, allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_random_potential_meam(name, a, b):
           np.testing.assert_allclose(a,b,atol=ATOL)

    if phionly_flag:
        """phionly subtype"""
        p = tests.potentials.get_random_pots(N)['phionlys']

        energies, _ = getLammpsResults(p, allstructs)

        calculated = runner(p, allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_random_potential_phionly(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)

    if rhophi_flag:
        """rhophi subtype"""
        p = tests.potentials.get_random_pots(N)['rhophis']

        energies, _ = getLammpsResults(p, allstructs)

        calculated = runner(p, allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_random_potential_rhophi(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)

    if nophi_flag:
        """nophi subtype"""
        p = tests.potentials.get_random_pots(N)['nophis']

        energies, _ = getLammpsResults(p, allstructs)

        calculated = runner(p, allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_random_potential_nophi(name, a, b):
           np.testing.assert_allclose(a,b,atol=ATOL)

    if rho_flag:
        """rho subtype"""
        p = tests.potentials.get_random_pots(N)['rhos']

        p[0].plot()
        energies, _ = getLammpsResults(p, allstructs)

        calculated = runner(p, allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_random_potential_rho(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)

    if norho_flag:
        """norho subtype"""
        p = tests.potentials.get_random_pots(N)['norhos']

        energies, _ = getLammpsResults(p, allstructs)

        calculated = runner(p, allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_random_potential_norho(name, a, b):
           np.testing.assert_allclose(a,b,atol=ATOL)

    if norhophi_flag:
        """norhophi subtype"""
        p = tests.potentials.get_random_pots(N)['norhophis']

        energies, _ = getLammpsResults(p, allstructs)

        calculated = runner(p, allstructs)

        @parameterized.expand(loader('', calculated, energies))
        def test_random_potential_norhophi(name, a, b):
           np.testing.assert_allclose(a,b,atol=ATOL)
################################################################################

#logging.info("Time spent calculating in LAMMPS: {} s".format(round(
#    lammps_calcduration,3)))
#logging.info("Time spent calculating in Python: {} s".format(round(
#    py_calcduration,3)))
