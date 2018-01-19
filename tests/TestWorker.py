import unittest
import numpy as np
import logging
import time

from nose_parameterized import parameterized

import meam

from worker import Worker

import tests.testPotentials

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
    bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo, extra

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#logging.disable(logging.CRITICAL)

EPS = 1e-15
np.random.seed(42)

N = 1

# Flags for what tests to run
energy_flag = True*0
forces_flag = True*1

zero_pots_flag  = True*0
const_pots_flag = True*1
rand_pots_flag  = True*0

meam_flag       = True*0
phionly_flag    = True*1
rhophi_flag     = True*0
nophi_flag      = True*0
rho_flag        = True*0
norho_flag      = True*0
norhophi_flag   = True*0

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
                  **bulk_vac_rhombo, **bulk_periodic_rhombo, **extra}

allstructs = {'aba':trimers['aba']}

################################################################################
# Helper functions

def loader_energy(group_name, calculated, lammps):
    """Assumes 'lammps' and 'calculated' are dictionaries where key:value =
    <struct name>:<list of calculations> where each entry in the list of
    calculations corresponds to a single potential."""

    tests = []
    for name in calculated.keys():
        logging.info("LAMMPS: {0} = {1}".format(name, lammps[name]))
        logging.info("WORKER: {0} = {1}".format(name, calculated[name]))
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
        tests.append((test_name, calculated[name], lammps[name]))

def getLammpsResults(pots, structs):

    start = time.time()

    # Builds dictionaries where dict[i]['ptype'][j] is the energy or force matrix of
    # struct i using potential j for the given 'ptype', according to LAMMPS
    energies = {}
    forces = {}

    global lammps_calcduration

    for key in structs.keys():
        energies[key] = np.zeros(len(pots))
        forces[key] = []

    for pnum,p in enumerate(pots):

        for name in structs.keys():
            atoms = structs[name]

            val = p.compute_energy(atoms)

            # cstart = time.time()
            results = p.get_lammps_results(atoms)
            energies[name][pnum] = results['energy']
            forces[name].append(results['forces'])
            # lammps_calcduration += float(time.time() - cstart)

            # TODO: LAMMPS runtimes are inflated due to ASE internal read/write
            # TODO: need to create shell script to get actual runtimes

    return energies, forces

def runner_energy(pots, structs):

    x_pvec, y_pvec, indices = meam.splines_to_pvec(pots[0].splines)

    energies = {}
    for name in structs.keys():
        atoms = structs[name]

        global py_calcduration

        w = Worker(atoms, x_pvec, indices, pots[0].types)
        start = time.time()
        # TODO: to optimize, preserve workers for each struct
        energies[name] = w.compute_energies(y_pvec)
        # py_calcduration += time.time() - start

    return energies

def runner_forces(pots, structs):

    x_pvec, y_pvec, indices = meam.splines_to_pvec(pots[0].splines)

    forces = {}
    for name in structs.keys():
        atoms = structs[name]

        global py_calcduration

        w = Worker(atoms, x_pvec, indices, pots[0].types)
        start = time.time()
        # TODO: to optimize, preserve workers for each struct
        forces[name] = w.compute_forces(pots[0])
        # py_calcduration += time.time() - start

    return forces

################################################################################

if zero_pots_flag:
    """Zero potentials"""
    p = tests.testPotentials.get_zero_potential()

    energies, forces = getLammpsResults([p], allstructs)

    if energy_flag:
        calc_energies = runner_energy([p], allstructs)
        print()

        @parameterized.expand(loader_energy('', calc_energies, energies))
        def test_zero_potential_energy(name, a, b):
            np.testing.assert_allclose(a,b,atol=EPS)

    if forces_flag:
        calc_forces = runner_forces([p], allstructs)
        @parameterized.expand(loader_forces('',calc_forces, forces))
        def test_zero_potential_forces(name, a, b):
            np.testing.assert_allclose(a,b,atol=EPS)
################################################################################
# TODO: const_pot test needs multiple potentials at once
if const_pots_flag:
    """Constant potentials"""

    p = tests.testPotentials.get_constant_potential()

    if meam_flag:
        """meam subtype"""
        energies, forces = getLammpsResults([p], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_meam_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([p], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_meam_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if phionly_flag:
        """phionly subtype"""
        energies, forces = getLammpsResults([p.phionly_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.phionly_subtype()], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_phionly_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([p.phionly_subtype()], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_phionly_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if rhophi_flag:
        """rhophi subtype"""
        energies, forces = getLammpsResults([p.rhophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.rhophi_subtype()], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rhophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([p.rhophi_subtype()], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if nophi_flag:
        """nophi subtype"""
        energies, forces = getLammpsResults([p.nophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.nophi_subtype()], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_nophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([p.nophi_subtype()], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_nophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if rho_flag:
        """rho subtype"""
        energies, forces = getLammpsResults([p.rho_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.rho_subtype()], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rho_energy(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([p.rho_subtype()], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if norho_flag:
        """norho subtype"""
        energies, forces = getLammpsResults([p.norho_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.norho_subtype()], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norho_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([p.norho_subtype()], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if norhophi_flag:
        """norhophi subtype"""
        energies, forces = getLammpsResults([p.norhophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.norhophi_subtype()], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norhophi_energy(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([p.norhophi_subtype()], allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)
    #################################################################################
if rand_pots_flag:
    """Random potentials"""

    if meam_flag:
        """meam subtype"""
        p = tests.testPotentials.get_random_pots(N)['meams']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_meam_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_meam_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if phionly_flag:
        """phionly subtype"""
        p = tests.testPotentials.get_random_pots(N)['phionlys']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_phionly_energy(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_phionly_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if rhophi_flag:
        """rhophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['rhophis']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rhophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if nophi_flag:
        """nophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['nophis']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_nophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_nophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if rho_flag:
        """rho subtype"""
        p = tests.testPotentials.get_random_pots(N)['rhos']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)
            print()

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rho_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if norho_flag:
        """norho subtype"""
        p = tests.testPotentials.get_random_pots(N)['norhos']

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norho_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norho_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

    if norhophi_flag:
        """norhophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['norhophis']
        p[0].write_to_file("random.norhophi")

        energies, forces = getLammpsResults(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norhophi_energy(name, a, b):
               np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)

            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norhophi_forces(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

################################################################################
