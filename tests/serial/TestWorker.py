import sys
sys.path.append('/home/jvita/scripts/s-meam/project/')
import numpy as np
np.random.seed(42)

import time
import unittest
import collections

from nose_parameterized import parameterized

import src.meam

from src.worker import Worker

#np.random.seed(237907459)

import tests.testPotentials
from tests.testVars import a0

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
    bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo, extra

# logging.disable(logging.CRITICAL)

DECIMAL = 12
GRAD_DECIMAL = 4

N = 1

# Flags for what tests to run
energy_flag = True * 1
forces_flag = True * 1

zero_pots_flag  = True * 0
const_pots_flag = True * 0
rand_pots_flag  = True * 1

meam_flag       = True * 1
phionly_flag    = True * 0
rhophi_flag     = True * 0
nophi_flag      = True * 0
rho_flag        = True * 0
norho_flag      = True * 0
norhophi_flag   = True * 0

dimers_flag     = True * 1
trimers_flag    = True * 1
bulk_flag       = True * 0
extra_flag      = True * 1

allstructs = {}

if dimers_flag:
    allstructs = {**allstructs, **dimers}
if trimers_flag:
    allstructs = {**allstructs, **trimers}
if bulk_flag:
    allstructs = {**allstructs, **bulk_vac_ortho, **bulk_periodic_ortho,
                  **bulk_vac_rhombo, **bulk_periodic_rhombo, **extra}
if extra_flag:
    allstructs = {**allstructs, **extra}

full_start = time.time()

################################################################################
# Helper functions

def loader_energy(group_name, calculated, lammps):
    """Assumes 'lammps' and 'calculated' are dictionaries where key:value =
    <struct name>:<list of calculations> where each entry in the list of
    calculations corresponds to a single potential."""

    load_tests = []
    for name in calculated.keys():

        np.set_printoptions(precision=16)

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
        np.set_printoptions(precision=16)
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
            # lmp_energies[name][pnum] = results['energy'] / len(atoms)
            # lmp_forces[name].append(results['forces'] / len(atoms))

            lmp_energies[name][pnum] = lmp_p.compute_energy(atoms)[0] / len(atoms)
            lmp_forces[name].append(lmp_p.compute_forces(atoms) / len(atoms))

    return lmp_energies, lmp_forces


# @profile
def runner_energy(pots, structs, u_ranges):
    x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pots[0].splines)
    # logging.info("x_pvec.shape: {0}".format(x_pvec.shape))
    # logging.info("x_pvec: {0}".format(x_pvec))
    # logging.info("a0: {0}".format(a0))

    wrk_energies = {}
    for name in structs.keys():
        atoms = structs[name]

        # logging.info("COMPUTING: {0}".format(name))
        w = Worker(atoms, x_pvec, indices, pots[0].types)
        # TODO: to optimize, preserve workers for each struct
        wrk_energies[name] = w.compute_energy(y_pvec, u_ranges)[0] / len(atoms)

    return wrk_energies

# @profile
def runner_forces(pots, structs, u_ranges):
    x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pots[0].splines)

    wrk_forces = {}
    start = time.time()
    for name in structs.keys():
        atoms = structs[name]

        w = Worker(atoms, x_pvec, indices, pots[0].types)
        wrk_forces[name] = np.array(w.compute_forces(y_pvec, u_ranges) / len(atoms))
    # logging.info(" ...... {0} second(s)".format(time.time() - start))

    return wrk_forces


################################################################################

if zero_pots_flag:
    """Zero potentials"""
    p = tests.testPotentials.get_zero_potential()

    energies, forces = get_lammps_results([p], allstructs)

    if energy_flag:
        calc_energies = runner_energy([p], allstructs)


        @parameterized.expand(loader_energy('', calc_energies, energies))
        def test_zero_potential_energy(name, a, b):
            diff = np.abs(a - b)
            np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

    if forces_flag:
        calc_forces = runner_forces([p], allstructs)


        @parameterized.expand(loader_forces('', calc_forces, forces))
        def test_zero_potential_forces(name, a, b):
            max_diff = np.max(np.abs(a - b))
            np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)
################################################################################
# TODO: const_pot test needs multiple potentials at once
if const_pots_flag:
    """Constant potentials"""

    p = tests.testPotentials.get_constant_potential()

    if meam_flag:
        """meam subtype"""
        energies, forces = get_lammps_results([p], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p], allstructs, [(1, a0 + 1)]*2)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_meam_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces([p], allstructs, [(1, a0 + 1)]*2)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_meam_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if phionly_flag:
        """phionly subtype"""
        energies, forces = get_lammps_results([p.phionly_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.phionly_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_phionly_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces([p.phionly_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_phionly_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if rhophi_flag:
        """rhophi subtype"""
        energies, forces = get_lammps_results([p.rhophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.rhophi_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rhophi_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces([p.rhophi_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rhophi_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if nophi_flag:
        """nophi subtype"""
        energies, forces = get_lammps_results([p.nophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.nophi_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_nophi_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces([p.nophi_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_nophi_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if rho_flag:
        """rho subtype"""
        energies, forces = get_lammps_results([p.rho_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.rho_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_rho_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces([p.rho_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_rho_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if norho_flag:
        """norho subtype"""
        energies, forces = get_lammps_results([p.norho_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.norho_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norho_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces([p.norho_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norho_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if norhophi_flag:
        """norhophi subtype"""
        energies, forces = get_lammps_results(
            [p.norhophi_subtype()], allstructs)

        if energy_flag:
            calc_energies = runner_energy([p.norhophi_subtype()], allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norhophi_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces([p.norhophi_subtype()], allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_constant_potential_norhophi_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)
################################################################################

if rand_pots_flag:
    """Random potentials"""

    if meam_flag:
        """meam subtype"""
        p = tests.testPotentials.get_random_pots(N)['meams']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs, [(0, a0)]*2)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_meam_energy(name, a, b):
                # logging.info("a: {0}".format(a))
                # logging.info("b: {0}".format(b))
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs, [(0, a0)]*2)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_meam_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if phionly_flag:
        """phionly subtype"""
        p = tests.testPotentials.get_random_pots(N)['phionlys']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_phionly_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_phionly_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if rhophi_flag:
        """rhophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['rhophis']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rhophi_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rhophi_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if nophi_flag:
        """nophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['nophis']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_nophi_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_nophi_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if rho_flag:
        """rho subtype"""
        p = tests.testPotentials.get_random_pots(N)['rhos']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_rho_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_rho_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if norho_flag:
        """norho subtype"""
        p = tests.testPotentials.get_random_pots(N)['norhos']

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norho_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norho_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

    if norhophi_flag:
        """norhophi subtype"""
        p = tests.testPotentials.get_random_pots(N)['norhophis']

        p[0].write_to_file('spline.poop')

        energies, forces = get_lammps_results(p, allstructs)

        if energy_flag:
            calc_energies = runner_energy(p, allstructs)


            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_random_potential_norhophi_energy(name, a, b):
                diff = np.abs(a - b)
                np.testing.assert_almost_equal(diff, 0.0, decimal=DECIMAL)

        if forces_flag:
            calc_forces = runner_forces(p, allstructs)


            @parameterized.expand(loader_forces('', calc_forces, forces))
            def test_random_potential_norhophi_forces(name, a, b):
                max_diff = np.max(np.abs(a - b))
                np.testing.assert_almost_equal(max_diff, 0.0, decimal=DECIMAL)

# def test_hdf5():
#     import h5py
#     from tests.testStructs import allstructs
#
#     p = tests.testPotentials.get_random_pots(1)['meams'][0]
#     x_pvec, y_pvec, indices = src.meam.splines_to_pvec(p.splines)
#     y_pvec = np.atleast_2d(y_pvec)
#
#     atoms = allstructs['8_atoms']
#
#     w1 = Worker(atoms, x_pvec, indices, ['H', 'He'])
#
#     hdf5_file = h5py.File("test.hdf5", 'w')
#
#     w1.add_to_hdf5(hdf5_file, 'worker')
#     w2 = Worker.from_hdf5(hdf5_file, 'worker')
#
#     np.testing.assert_almost_equal(w1.compute_energy(y_pvec),
#             w2.compute_energy(y_pvec), decimal=DECIMAL)
#
#     np.testing.assert_almost_equal(w1.compute_forces(y_pvec),
#             w2.compute_forces(y_pvec), decimal=DECIMAL)
#
#     os.remove("test.hdf5")

class GradientTests(unittest.TestCase):

    def setUp(self):
        phionly = tests.testPotentials.get_random_pots(1)['phionlys'][0]
        rho = tests.testPotentials.get_random_pots(1)['rhos'][0]
        rhophi = tests.testPotentials.get_random_pots(1)['rhophis'][0]
        norhophi = tests.testPotentials.get_random_pots(1)['norhophis'][0]
        full = tests.testPotentials.get_random_pots(1)['meams'][0]

        Potential = collections.namedtuple('Potential', 'x_pvec y_pvec indices')

        self.phionly = Potential(*src.meam.splines_to_pvec(phionly.splines))
        self.rho = Potential(*src.meam.splines_to_pvec(rho.splines))
        self.rhophi = Potential(*src.meam.splines_to_pvec(rhophi.splines))
        self.norhophi = Potential(*src.meam.splines_to_pvec(norhophi.splines))
        self.full = Potential(*src.meam.splines_to_pvec(full.splines))

        def fd_gradient(y_pvec, worker, type, h=1e-8):
            N = y_pvec.ravel().shape[0]

            cd_points = np.array([y_pvec] * N*2)

            for l in range(N):
                cd_points[2*l, l] += h
                cd_points[2*l+1, l] -= h

            if type == 'energy':
                cd_evaluated = worker.compute_energy(np.array(cd_points), [(0,a0)]*2)[0]
                fd_gradient = np.zeros(N)
            elif type == 'forces':
                cd_evaluated = worker.compute_forces(np.array(cd_points), [(0, a0)]*2)
                fd_gradient = np.zeros((worker.natoms, 3, worker.len_param_vec))

            for l in range(N):
                if type == 'energy': fd_gradient[l] = \
                    (cd_evaluated[2*l] - cd_evaluated[2*l+1]) / h / 2

                elif type == 'forces': fd_gradient[:, :, l] = \
                    (cd_evaluated[2*l] - cd_evaluated[2*l+1]) / h / 2

            return fd_gradient

        self.fd_gradient_eval = fd_gradient

        self.types = ['H', 'He']
        self.pots = [self.phionly, self.rho, self.rhophi,
                     self.norhophi, self.full]

    def test_aa(self):
        atoms = allstructs['aa']
        worker = Worker(atoms, self.phionly.x_pvec, self.phionly.indices,
                        self.types)
        
        for pot in self.pots:
            fd_grad_e = self.fd_gradient_eval(pot.y_pvec, worker, 'energy')
            fd_grad_f = self.fd_gradient_eval(pot.y_pvec, worker, 'forces')

            w_grad_e = worker.energy_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)
            w_grad_f = worker.forces_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)

            fd_grad_e = fd_grad_e.reshape(w_grad_e.shape)
            fd_grad_f = fd_grad_f.reshape(w_grad_f.shape)

            np.testing.assert_almost_equal(fd_grad_e, w_grad_e,
                                           decimal=GRAD_DECIMAL)

            np.testing.assert_almost_equal(fd_grad_f, w_grad_f,
                                           decimal=GRAD_DECIMAL)

    def test_ab(self):
        atoms = allstructs['ab']
        worker = Worker(atoms, self.phionly.x_pvec, self.phionly.indices,
                        self.types)

        for pot in self.pots:
            fd_grad_e = self.fd_gradient_eval(pot.y_pvec, worker, 'energy')
            fd_grad_f = self.fd_gradient_eval(pot.y_pvec, worker, 'forces')

            w_grad_e = worker.energy_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)
            w_grad_f = worker.forces_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)

            fd_grad_e = fd_grad_e.reshape(w_grad_e.shape)
            fd_grad_f = fd_grad_f.reshape(w_grad_f.shape)

            np.testing.assert_almost_equal(fd_grad_e, w_grad_e,
                                           decimal=GRAD_DECIMAL)

            np.testing.assert_almost_equal(fd_grad_f, w_grad_f,
                                           decimal=GRAD_DECIMAL)

    def test_aaa(self):
        atoms = allstructs['aaa']
        worker = Worker(atoms, self.phionly.x_pvec, self.phionly.indices,
                        self.types)

        for pot in self.pots:
            fd_grad_e = self.fd_gradient_eval(pot.y_pvec, worker, 'energy')
            fd_grad_f = self.fd_gradient_eval(pot.y_pvec, worker, 'forces')

            w_grad_e = worker.energy_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)
            w_grad_f = worker.forces_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)

            fd_grad_e = fd_grad_e.reshape(w_grad_e.shape)
            fd_grad_f = fd_grad_f.reshape(w_grad_f.shape)

            np.testing.assert_almost_equal(fd_grad_e, w_grad_e,
                                           decimal=GRAD_DECIMAL)

            np.testing.assert_almost_equal(fd_grad_f, w_grad_f,
                                           decimal=GRAD_DECIMAL)

    # @profile
    def test_aba(self):
        atoms = allstructs['aba']
        worker = Worker(atoms, self.phionly.x_pvec, self.phionly.indices,
                        self.types)

        for pot in self.pots:
            fd_grad_e = self.fd_gradient_eval(pot.y_pvec, worker, 'energy')
            fd_grad_f = self.fd_gradient_eval(pot.y_pvec, worker, 'forces')

            w_grad_e = worker.energy_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)
            w_grad_f = worker.forces_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)

            fd_grad_e = fd_grad_e.reshape(w_grad_e.shape)
            fd_grad_f = fd_grad_f.reshape(w_grad_f.shape)

            np.testing.assert_almost_equal(fd_grad_e, w_grad_e,
                                           decimal=GRAD_DECIMAL)

            np.testing.assert_almost_equal(fd_grad_f, w_grad_f,
                                           decimal=GRAD_DECIMAL)

    def test_8_atoms(self):
        atoms = allstructs['8_atoms']
        worker = Worker(atoms, self.phionly.x_pvec, self.phionly.indices,
                        self.types)

        for pot in self.pots:
            fd_grad_e = self.fd_gradient_eval(pot.y_pvec, worker, 'energy')
            fd_grad_f = self.fd_gradient_eval(pot.y_pvec, worker, 'forces')

            w_grad_e = worker.energy_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)
            w_grad_f = worker.forces_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)

            fd_grad_e = fd_grad_e.reshape(w_grad_e.shape)
            fd_grad_f = fd_grad_f.reshape(w_grad_f.shape)

            np.testing.assert_almost_equal(fd_grad_e, w_grad_e,
                                           decimal=GRAD_DECIMAL)

            np.testing.assert_almost_equal(fd_grad_f, w_grad_f,
                                           decimal=GRAD_DECIMAL)

    if bulk_flag:
        def test_bulk(self):
            atoms = allstructs['bulk_periodic_ortho_mixed']
            worker = Worker(atoms, self.phionly.x_pvec, self.phionly.indices,
                            self.types)

            for pot in self.pots:
                fd_grad_e = self.fd_gradient_eval(pot.y_pvec, worker, 'energy')
                fd_grad_f = self.fd_gradient_eval(pot.y_pvec, worker, 'forces')

                w_grad_e = worker.energy_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)
                w_grad_f = worker.forces_gradient_wrt_pvec(pot.y_pvec, [(0, a0)]*2)

                fd_grad_e = fd_grad_e.reshape(w_grad_e.shape)
                fd_grad_f = fd_grad_f.reshape(w_grad_f.shape)

                np.testing.assert_almost_equal(fd_grad_e, w_grad_e,
                                               decimal=GRAD_DECIMAL)

                np.testing.assert_almost_equal(fd_grad_f, w_grad_f,
                                               decimal=GRAD_DECIMAL)

################################################################################

# logging.info("Total runtime: {0}".format(time.time() - full_start))
