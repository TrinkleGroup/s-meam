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

N = 1

# Flags for what tests to run
energy_flag = True*0
forces_flag = True*1

zero_pots_flag  = True*0
const_pots_flag = True*0
rand_pots_flag  = True*1

meam_flag       = True*0
phionly_flag    = True*0
rhophi_flag     = True*0
nophi_flag      = True*0
rho_flag        = True*0
norho_flag      = True*0
norhophi_flag   = True*1

dimers_flag  = True*0
trimers_flag = True*1
bulk_flag    = True*0

allstructs = {}

if dimers_flag:
    allstructs = {**allstructs, **dimers}
if trimers_flag:
    allstructs = {**allstructs, **trimers}
if bulk_flag:
    allstructs = {**allstructs, **bulk_vac_ortho, **bulk_periodic_ortho,
                  **bulk_vac_rhombo, **bulk_periodic_rhombo}

key = 'bab'

allstructs = {key:trimers[key]}

################################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.terminator = ""

py_calcduration = 0
lammps_calcduration = 0

def loader_energy(group_name, calculated, lammps):
    """Assumes 'lammps' and 'calculated' are dictionaries where key:value =
    <struct name>:<list of calculations> where each entry in the list of
    calculations corresponds to a single potential."""

    tests = []
    for name in calculated.keys():
        test_name = group_name + '_' + name + '_energy'
        tests.append((test_name, calculated[name], lammps[name]))

    return tests

def loader_forces(group_name, calculated, lammps):
    """Assumes 'lammps' and 'calculated' are dictionaries where key:value =
    <struct name>:<list of calculations> where each entry in the list of
    calculations corresponds to a single potential."""

    tests = []
    for name in calculated.keys():
        test_name = group_name + '_' + name + '_forces'
        #logging.info("calc{0}".format(calculated.keys()))
        #logging.info("lammps{0}".format(lammps.keys()))
        tests.append((test_name, calculated[name], lammps[name]))

    return tests

def runner_energy(pots, structs):

    energies = {}
    for name in structs.keys():
        atoms = structs[name]

        global py_calcduration

        w = worker(atoms,pots)
        start = time.time()
        #calculated[name] = w.compute_energies()
        # TODO: to optimize, preserve workers for each struct
        energies[name] = w.compute_energies(pots)
        py_calcduration += time.time() - start

        #logging.info("{0}".format(calculated[name]))

    return energies

def runner_forces(pots, structs):

    forces = {}
    for name in structs.keys():
        atoms = structs[name]

        global py_calcduration

        w = worker(atoms,pots)
        start = time.time()
        #calculated[name] = w.compute_energies()
        # TODO: to optimize, preserve workers for each struct
        forces[name] = w.compute_forces(pots)
        py_calcduration += time.time() - start

        #logging.info("{0}".format(forces[name]))

    return forces

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
        forces[key] = []

    for pnum,p in enumerate(pots):

        p.write_to_file('test.meam.spline')

        calc = LAMMPS(no_data_file=True, parameters=params, \
                      keep_tmp_files=False,specorder=types,files=['test.meam.spline'])

        for name in structs.keys():
            atoms = structs[name]

            cstart = time.time()
            energies[name][pnum] = calc.get_potential_energy(atoms)
            forces[name].append(calc.get_forces(atoms))
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

    energies, forces = getLammpsResults([p], allstructs)

    if energy_flag:
        calc_energies = runner_energy([p], allstructs)

        @parameterized.expand(loader_energy('', calc_energies, energies))
        def test_zero_potential_energy(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)

    if forces_flag:
        calc_forces = runner_forces([p], allstructs)
        @parameterized.expand(loader_forces('',calc_forces, forces))
        def test_zero_potential_forces(name, a, b):
            np.testing.assert_allclose(a,b,atol=ATOL)
################################################################################
# TODO: const_pot test needs multiple potentials at once
if const_pots_flag:
    """Constant potentials"""

    p = tests.potentials.get_constant_potential()

    if meam_flag:
        """meam subtype"""
        energies, forces = getLammpsResults([p], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_meam_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces([p], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_meam_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if phionly_flag:
        """phionly subtype"""
        energies, forces = getLammpsResults([meam.phionly_subtype(p)], allstructs)

        if energy_flag:
            calc_energies = runner_energy([meam.phionly_subtype(p)], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_phionly_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces([meam.phionly_subtype(p)], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_phionly_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if rhophi_flag:
        """rhophi subtype"""
        energies, forces = getLammpsResults([meam.rhophi_subtype(p)], allstructs)

        if energy_flag:
            calc_energies = runner_energy([meam.rhophi_subtype(p)], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rhophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces([meam.rhophi_subtype(p)], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if nophi_flag:
        """nophi subtype"""
        energies, forces = getLammpsResults([meam.nophi_subtype(p)], allstructs)

        if energy_flag:
            calc_energies = runner_energy([meam.nophi_subtype(p)], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_nophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces([meam.nophi_subtype(p)], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_nophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if rho_flag:
        """rho subtype"""
        energies, forces = getLammpsResults([meam.rho_subtype(p)], allstructs)

        if energy_flag:
            calc_energies = runner_energy([meam.rho_subtype(p)], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rho_energy(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces([meam.rho_subtype(p)], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if norho_flag:
        """norho subtype"""
        energies, forces = getLammpsResults([meam.norho_subtype(p)], allstructs)

        if energy_flag:
            calc_energies = runner_energy([meam.norho_subtype(p)], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norho_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces([meam.norho_subtype(p)], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if norhophi_flag:
        """norhophi subtype"""
        energies, forces = getLammpsResults([meam.norhophi_subtype(p)], allstructs)

        if energy_flag:
            calc_energies = runner_energy([meam.norhophi_subtype(p)], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norhophi_energy(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces([meam.norhophi_subtype(p)], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)
    #################################################################################
if rand_pots_flag:
    """Random potentials"""

    if meam_flag:
        """meam subtype"""
        p = tests.potentials.get_random_pots(N)['meams']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_meam_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_meam_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if phionly_flag:
        """phionly subtype"""
        p = tests.potentials.get_random_pots(N)['phionlys']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_phionly_energy(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_phionly_forces(name, a, b):
                #rzm: basic triplet failing; second atom is correct, but for all pots?
                np.testing.assert_allclose(a,b,atol=ATOL)

    if rhophi_flag:
        """rhophi subtype"""
        p = tests.potentials.get_random_pots(N)['rhophis']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rhophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if nophi_flag:
        """nophi subtype"""
        p = tests.potentials.get_random_pots(N)['nophis']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_nophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_nophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if rho_flag:
        """rho subtype"""
        p = tests.potentials.get_random_pots(N)['rhos']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rho_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if norho_flag:
        """norho subtype"""
        p = tests.potentials.get_random_pots(N)['norhos']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norho_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)

    if norhophi_flag:
        """norhophi subtype"""
        p = tests.potentials.get_random_pots(N)['norhophis']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norhophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=ATOL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=ATOL)
################################################################################

#logging.info("Time spent calculating in LAMMPS: {} s".format(round(
#    lammps_calcduration,3)))
#logging.info("Time spent calculating in Python: {} s".format(round(
#    py_calcduration,3)))
