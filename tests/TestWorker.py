import numpy as np
import logging
import time

from nose_parameterized import parameterized

import meam

from worker import Worker

np.random.seed(42)

import tests.testPotentials

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
    bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo, extra

logging.basicConfig(filename='test_results.dat', level=logging.DEBUG)
# logging.disable(logging.CRITICAL)

EPS = 1e-12

N = 1

# Flags for what tests to run
energy_flag = True * 0
forces_flag = True * 1

zero_pots_flag = True * 0
const_pots_flag = True * 0
rand_pots_flag = True * 1

meam_flag = True * 1
phionly_flag = True * 0
rhophi_flag = True * 0
nophi_flag = True * 0
rho_flag = True * 0
norho_flag = True * 0
norhophi_flag = True * 0

dimers_flag = True * 1
trimers_flag = True * 1
bulk_flag = True * 1

allstructs = {}

if dimers_flag:
    allstructs = {**allstructs, **dimers}
if trimers_flag:
    allstructs = {**allstructs, **trimers}
if bulk_flag:
    allstructs = {**allstructs, **bulk_vac_ortho, **bulk_periodic_ortho,
                  **bulk_vac_rhombo, **bulk_periodic_rhombo, **extra}


# allstructs = {'bulk_vac_rhombo_mixed':bulk_vac_rhombo[
#     'bulk_vac_rhombo_mixed']}

################################################################################
# Helper functions

def loader_energy(group_name, calculated, lammps):
    """Assumes 'lammps' and 'calculated' are dictionaries where key:value =
    <struct name>:<list of calculations> where each entry in the list of
    calculations corresponds to a single potential."""

    load_tests = []
    for name in calculated.keys():
        test_name = group_name + '_' + name + '_energy'
        load_tests.append((test_name, calculated[name], lammps[name]))

    return load_tests


def loader_forces(group_name, calculated, lammps):
    """Assumes 'lammps' and 'calculated' are dictionaries where key:value =
    <struct name>:<list of calculations> where each entry in the list of
    calculations corresponds to a single potential."""

    load_tests = []
    for name in calculated.keys():
        test_name = group_name + '_' + name + '_forces'
        load_tests.append((test_name, calculated[name], lammps[name][0]))

    return load_tests


def get_lammps_results(pots, structs):
    """
    Builds dictionaries where dict[i]['ptype'][j] is the energy or force
    matrix of struct i using potential j for the given ptype, according to
    LAMMPS
    """

    lmp_energies = {}
    lmp_forces = {}

    for key in structs.keys():
        lmp_energies[key] = np.zeros(len(pots))
        lmp_forces[key] = []

    for pnum, lmp_p in enumerate(pots):

        for name in structs.keys():
            atoms = structs[name]

            results = lmp_p.get_lammps_results(atoms)
            lmp_energies[name][pnum] = results['energy'] / len(atoms)
            lmp_forces[name].append(results['lmp_forces'])

            # TODO: LAMMPS runtimes are inflated due to ASE internal read/write
            # TODO: need to create shell script to get actual runtimes

    return lmp_energies, lmp_forces


def runner_energy(pots, structs):
    x_pvec, y_pvec, indices = meam.splines_to_pvec(pots[0].splines)

    wrk_energies = {}
    for name in structs.keys():
        atoms = structs[name]

        w = Worker(atoms, x_pvec, indices, pots[0].types)
        # TODO: to optimize, preserve workers for each struct
        wrk_energies[name] = w.compute_energies(y_pvec) / len(atoms)

    return wrk_energies


def runner_forces(pots, structs):
    x_pvec, y_pvec, indices = meam.splines_to_pvec(pots[0].splines)

    wrk_forces = {}
    for name in structs.keys():
        start = time.time()
        atoms = structs[name]

        w = Worker(atoms, x_pvec, indices, pots[0].types)
        wrk_forces[name] = w.compute_forces(y_pvec)
        logging.info(" ...... {0} second(s)".format(time.time() - start))

    return forces


################################################################################

if zero_pots_flag:
    """Zero potentials"""
    p = tests.testPotentials.get_zero_potential()

    energies, forces = get_lammps_results([p], allstructs)

    if energy_flag:
        calc_energies = runner_energy([p], allstructs)


        @parameterized.expand(loader_energy('', calc_energies, energies))
        def test_zero_potential_energy(name, a, b):
            np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if forces_flag:
        calc_forces = runner_forces([p], allstructs)


        @parameterized.expand(loader_forces('', calc_forces, forces))
        def test_zero_potential_forces(name, a, b):
            np.testing.assert_allclose(a, b, atol=EPS, rtol=0)
################################################################################
# TODO: const_pot test needs multiple potentials at once
if const_pots_flag:
    """Constant potentials"""

    p = tests.testPotentials.get_constant_potential()

    if meam_flag:
        """meam subtype"""
        energies, forces = get_lammps_results([p], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_meam_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces([p], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_meam_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if phionly_flag:
        """phionly subtype"""
        energies, forces = get_lammps_results([p.phionly_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.phionly_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_phionly_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces([p.phionly_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_phionly_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if rhophi_flag:
        """rhophi subtype"""
        energies, forces = get_lammps_results([p.rhophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.rhophi_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rhophi_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces([p.rhophi_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rhophi_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if nophi_flag:
        """nophi subtype"""
        energies, forces = get_lammps_results([p.nophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.nophi_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_nophi_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces([p.nophi_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_nophi_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if rho_flag:
        """rho subtype"""
        energies, forces = get_lammps_results([p.rho_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.rho_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rho_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces([p.rho_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rho_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if norho_flag:
        """norho subtype"""
        energies, forces = get_lammps_results([p.norho_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.norho_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norho_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces([p.norho_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norho_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if norhophi_flag:
        """norhophi subtype"""
        energies, forces = get_lammps_results(
            [p.norhophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.norhophi_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norhophi_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces([p.norhophi_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norhophi_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)
################################################################################

if rand_pots_flag:
    """Random potentials"""

    if meam_flag:
        """meam subtype"""
        p = tests.testPotentials.get_random_pots(N)['meams']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_meam_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_meam_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if phionly_flag:
        """phionly subtype"""
        p = tests.testPotentials.get_random_pots(N)['phionlys']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_phionly_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_phionly_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if rhophi_flag:
        """rhophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['rhophis']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rhophi_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rhophi_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if nophi_flag:
        """nophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['nophis']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_nophi_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_nophi_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if rho_flag:
        """rho subtype"""
        p = tests.testPotentials.get_random_pots(N)['rhos']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rho_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rho_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if norho_flag:
        """norho subtype"""
        p = tests.testPotentials.get_random_pots(N)['norhos']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norho_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norho_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

    if norhophi_flag:
        """norhophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['norhophis']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norhophi_energy(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norhophi_forces(name, a, b):
                np.testing.assert_allclose(a, b, atol=EPS, rtol=0)

################################################################################
