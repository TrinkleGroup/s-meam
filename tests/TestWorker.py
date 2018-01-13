import unittest
import numpy as np
import time

from nose_parameterized import parameterized
from scipy.interpolate import CubicSpline
from spline import Spline

import meam
import worker

from worker import Worker, WorkerSpline

import tests.testPotentials

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
    bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo

EPS = 1e-7
np.random.seed(42)

N = 1

# Flags for what tests to run
energy_flag = True*1
forces_flag = True*0

zero_pots_flag  = True*0
const_pots_flag = True*0
rand_pots_flag  = True*1

meam_flag       = True*0
phionly_flag    = True*0
rhophi_flag     = True*0
nophi_flag      = True*0
rho_flag        = True*1
norho_flag      = True*0
norhophi_flag   = True*0

dimers_flag  = True*1
trimers_flag = True*0
bulk_flag    = True*0

allstructs = {}

if dimers_flag:
    allstructs = {**allstructs, **dimers}
if trimers_flag:
    allstructs = {**allstructs, **trimers}
if bulk_flag:
    allstructs = {**allstructs, **bulk_vac_ortho, **bulk_periodic_ortho,
                  **bulk_vac_rhombo, **bulk_periodic_rhombo}

# allstructs =  {'aa':dimers['aa']}

################################################################################
# Helper functions

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

            # cstart = time.time()
            results = p.compute_lammps_results(atoms)
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
        py_calcduration += time.time() - start

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

            # rzm: overcounting for rho?
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
        energies, forces = getLammpsResults([meam.norhophi_subtype(p)], allstructs)

        if energy_flag:
            calc_energies = runner_energy([meam.norhophi_subtype(p)], allstructs)

            @parameterized.expand(loader_energy('', calc_energies, energies))
            def test_constant_potential_norhophi_energy(name, a, b):
                np.testing.assert_allclose(a,b,atol=EPS)

        if forces_flag:
            calc_forces = runner_forces([meam.norhophi_subtype(p)], allstructs)

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
class ConstructorTests(unittest.TestCase):

    def test_one_potential(self):
        pass

    def test_many_potentials(self):
        pass

    def test_spline_grouping(self):
        pass

    def test_knot_array_indexing(self):
        pass

    def test_build_deriv_matrix(self):
        pass

class MethodsTests(unittest.TestCase):

    def test_build_deriv_matrix(self):
        pass

    def test_spline_interval_search(self):
        # TODO in TestSpline.py?
        pass

    def test_write_one_to_file(self):
        pass

    def test_write_many_to_file(self):
        pass

    def test_write_atoms_to_file(self):
        pass

class WorkerSplineTests(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-1, 1, 5)
        self.dx = self.x[1] - self.x[0]

        d0 = dN = 0
        self.y = np.array([ 0.05138434,  0.01790244, -0.26065088, -0.19016379,
                       -0.76379542, d0, dN])

    def test_against_CubicSpline(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(-10, 20, 1000)

        results = np.zeros(test_x.shape)
        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        # rzm: plots show that splines are different
        results = ws(self.y)
        #ws.plot()
        #cs.plot()

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]))

    def test_constructor_bad_x(self):
        x = self.x.copy()
        x[1] = -1

        self.assertRaises(ValueError, WorkerSpline, x, ('fixed','fixed'))

    def test_constructor_bad_bc(self):
        self.assertRaises(ValueError, WorkerSpline, self.x, ('fixed','bad'))

    def test_get_abcd_basic(self):
        x = np.arange(3)
        r = 0.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(r)

        true = np.array([0.5, 0.5, 0, 0.125, -0.125, 0])
        np.testing.assert_allclose(ws.struct_vec, true)

    def test_get_abcd_lhs_extrap(self):
        x = np.arange(3)
        r = -0.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(r)

        true = np.array([1, 0, 0, -0.5, 0, 0])
        np.testing.assert_allclose(ws.struct_vec, true)

    def test_get_abcd_rhs_extrap(self):
        x = np.arange(3)
        r = 2.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(r)

        true = np.array([0, 0, 1, 0, 0, 0.5])
        np.testing.assert_allclose(ws.struct_vec, true)

    def test_eval_flat(self):
        x = np.arange(10, dtype=float)
        y = np.zeros(12)

        ws = WorkerSpline(x, ('natural', 'natural'))
        ws.add_to_struct_vec(4)

        self.assertEqual(ws(y), 0.)

    def test_eval_sloped(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float); y[-2] = y[-1] = 1

        test_x = np.linspace(0, 10, 100)

        # TODO: should be same with 'nat'
        ws = WorkerSpline(x, ('fixed', 'fixed'))

        for el in test_x:
            ws.add_to_struct_vec(el)

        for i in range(100):
            np.testing.assert_allclose(np.sum(ws(y)), np.sum(test_x))

    def test_eval_flat_lhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.zeros(12)

        ws = WorkerSpline(x, ('natural', 'natural'))
        ws.add_to_struct_vec(-1)

        self.assertEqual(ws(y), 0.)

    def test_eval_sloped_lhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float); y[-2] = y[-1] = 1

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(-1.)

        self.assertEqual(ws(y), -1.)

    def test_eval_flat_rhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.zeros(12)

        ws = WorkerSpline(x, ('natural', 'natural'))
        ws.add_to_struct_vec(10)

        self.assertEqual(ws(y), 0.)

    def test_eval_sloped_rhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float); y[-2] = y[-1] = 1

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(10.)

        self.assertEqual(ws(y), 10.)

    def test_eval_sin_fxn(self):
        pass

    def test_build_M_natural_natural(self):
        M = worker.build_M(len(self.x), self.dx, bc_type=('natural',
                                                                'natural'))

        true = np.array([ 0.13719161, -0.47527463, -0.10830441, -0.33990513,
                          -1.55094231])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_natural_fixed(self):
        M = worker.build_M(len(self.x), self.dx, bc_type=('natural',
                                                                'fixed'))

        true = np.array([0.15318071, -0.50725283, 0.00361927, -0.75562163, 0.])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_fixed_natural(self):
        M = worker.build_M(len(self.x), self.dx, bc_type=('fixed',
                                                                'natural'))

        true = np.array([0, -0.43850162, -0.11820483, -0.33707643, -1.55235667])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_fixed_fixed(self):
        M = worker.build_M(len(self.x), self.dx, bc_type=('fixed',
                                                                'fixed'))

        true = np.array([0.00000000e+00, -4.66222277e-01, -7.32221143e-03,
                            -7.52886257e-01, -8.88178420e-16])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_bad_LHS(self):
        self.assertRaises(ValueError, worker.build_M, len(self.x),
                          self.dx, bc_type=('bad', 'natural'))

    def test_build_M_bad_RHS(self):
        self.assertRaises(ValueError, worker.build_M, len(self.x),
                          self.dx, bc_type=('natural', 'bad'))

