import unittest
import numpy as np

import workerSplines

from spline import Spline
# from workerSplines import WorkerSpline, RhoSpline, USpline
import src.workerSplines2
from workerSplines import RhoSpline, USpline
from src.workerSplines2 import WorkerSpline

DIGITS = 15
EPS = 1e-12


class WorkerSplineTests(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-1, -0.5, 0, 0.5, 1])
        self.dx = self.x[1] - self.x[0]

        self.y = np.array([1, .25, 0, .25, 1, -2, 2])
        self.y = np.atleast_2d(self.y)

    def test_full_eval_extrap_double_range(self):
        d0, dN = self.y[0, -2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        spline_range = self.x[-1] - self.x[0]

        test_x = np.linspace(self.x[0] - spline_range/2., self.x[-1] +
                             spline_range/2., 10)

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y)[0], cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_single_rhs_extrap(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        test_x = 2

        ws.add_to_energy_struct_vec(test_x)

        self.assertAlmostEqual(ws.calc_energy(self.y), cs(test_x))

    def test_two_rhs_extrap(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        test_x = [2, 3]

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y), cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_multiple_pvecs(self):
        d0, dN = self.y[0, -2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))

        cs1 = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        cs2 = Spline(self.x, self.y[0,:-2] + 1, bc_type=((1, d0), (1, dN)),
                     end_derivs=(d0, dN))

        spline_range = self.x[-1] - self.x[0]

        test_x = np.linspace(self.x[0] - spline_range/2., self.x[-1] +
                             spline_range/2., 10)

        y = self.y.ravel()
        double_y = np.vstack([y, y])
        double_y[1,:-2] += 1

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(double_y)[0], cs1(x))
            self.assertAlmostEqual(ws.calc_energy(double_y)[1], cs2(x))
            ws.structure_vectors['energy'][:] = 0

    def test_rhs_extrap(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        test_x = np.linspace(self.x[-1], self.x[-1] + 2, 100)

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y)[0], cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_single_lhs_extrap(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        test_x = -2

        ws.add_to_energy_struct_vec(test_x)

        # spline is x^2, but extrapolation should be linear outside range
        self.assertAlmostEqual(ws.calc_energy(self.y), cs(test_x))

    def test_two_lhs_extrap(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        test_x = [-2, -3]

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y), cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_lhs_extrap_one_in_one_out(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        test_x = [-1.5, -0.25]

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y), cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_lhs_extrap_single(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))

        x = self.x[0] - 1
        ws.add_to_energy_struct_vec(x)

        self.assertAlmostEqual(ws.calc_energy(self.y), self.y[0][0] - d0)

    def test_lhs_extrap_two_seperate(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))

        x = self.x[0] - 1
        test_x = [x,x]

        for x in test_x:
            ws.add_to_energy_struct_vec(x)

        self.assertAlmostEqual(ws.calc_energy(self.y), 2*(self.y[0][0] - d0))

    def test_lhs_extrap_two_together(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))

        x = self.x[0] - 1
        test_x = [x,x]

        ws.add_to_energy_struct_vec(test_x)
        self.assertAlmostEqual(ws.calc_energy(self.y), 2*(self.y[0][0] - d0))

    def test_lhs_extrap(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0, dN))

        test_x = np.linspace(self.x[0] - 2, self.x[0], 1000)

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y)[0], cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_inner_intervals(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0,dN))

        test_x = np.linspace(self.x[1], self.x[-2], 1000)

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y)[0], cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_leftmost_interval(self):
        d0, dN = self.y[0,-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0,dN))

        test_x = np.linspace(self.x[0], self.x[1], 100)

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y)[0], cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_rightmost_interval(self):
        d0, dN = self.y[0, -2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[0,:-2], bc_type=((1, d0), (1, dN)),
                    end_derivs=(d0,dN))

        test_x = np.linspace(self.x[-2], self.x[-1] - 0.01, 100)

        for x in test_x:
            ws.add_to_energy_struct_vec(x)
            self.assertAlmostEqual(ws.calc_energy(self.y)[0], cs(x))
            ws.structure_vectors['energy'][:] = 0

    def test_constructor_bad_x(self):
        x = self.x.copy()
        x[1] = -1

        self.assertRaises(ValueError, WorkerSpline, x, ('fixed', 'fixed'))

    def test_constructor_bad_bc(self):
        self.assertRaises(ValueError, WorkerSpline, self.x, ('fixed', 'bad'))

    def test_get_abcd_basic(self):
        x = np.arange(3)
        r = 0.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        calc = ws.get_abcd(r)

        true_beta = ws.M * np.array([.125, -.125, 0])[:, np.newaxis]
        true = np.array([.5, .5, 0, 0, 0]) + np.sum(true_beta, axis=0)

        np.testing.assert_allclose(calc, true, atol=EPS, rtol=0)

    def test_get_abcd_lhs_extrap(self):
        x = np.arange(3)
        r = -0.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        calc = ws.get_abcd(r)

        true_alpha = np.array([1, 0, 0, 0, 0])
        true_beta = ws.M * np.array([-0.5, 0, 0])[:, np.newaxis]

        true = true_alpha + np.sum(true_beta, axis=0)

        np.testing.assert_allclose(calc, true, atol=EPS, rtol=0)

    def test_get_abcd_rhs_extrap(self):
        x = np.arange(3)
        r = 2.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        calc = ws.get_abcd(r)

        true_alpha = np.array([0, 0, 1, 0, 0])
        true_beta = ws.M * np.array([0, 0, 0.5])[:, np.newaxis]
        true = true_alpha + np.sum(true_beta, axis=0)

        np.testing.assert_allclose(calc, true, atol=EPS, rtol=0)

    def test_eval_flat(self):
        x = np.arange(10, dtype=float)
        y = np.zeros((1,12))

        ws = WorkerSpline(x, ('natural', 'natural'))
        ws.add_to_energy_struct_vec(4)

        self.assertEqual(ws.calc_energy(y), 0.)

    def test_eval_internal_sloped_single(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float).reshape((1,12))
        y[0,-2] = y[0,-1] = 1

        test_x = 5.5

        # TODO: should be same with 'nat'
        ws = WorkerSpline(x, ('fixed', 'fixed'))
        # ws = WorkerSpline(x, ('natural', 'natural'))

        ws.add_to_energy_struct_vec(test_x)

        np.testing.assert_allclose(ws.calc_energy(y), test_x, atol=EPS, rtol=0)

    def test_eval_internal_sloped(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float).reshape((1,12))
        y[0,-2] = y[0,-1] = 1

        test_x = np.linspace(0, 9, 100)

        # TODO: should be same with 'nat'
        ws = WorkerSpline(x, ('fixed', 'fixed'))

        ws.add_to_energy_struct_vec(test_x)

        for i in range(100):
            np.testing.assert_allclose(np.sum(ws.calc_energy(y)),
                                       np.sum(test_x), atol=EPS, rtol=0)

    def test_eval_flat_lhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.zeros((1,12))

        ws = WorkerSpline(x, ('natural', 'natural'))
        ws.add_to_energy_struct_vec(-1)

        self.assertEqual(ws.calc_energy(y), 0.)

    def test_eval_sloped_lhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float).reshape((1,12))
        y[0, -2] = y[0, -1] = 1

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_energy_struct_vec(-1.)

        self.assertAlmostEqual(ws.calc_energy(y), -1)

    def test_eval_flat_rhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.zeros((1,12))

        ws = WorkerSpline(x, ('natural', 'natural'))
        ws.add_to_energy_struct_vec(10)

        self.assertEqual(ws.calc_energy(y), 0.)

    def test_eval_sloped_rhs_extrap(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float).reshape((1,12))
        y[0,-2] = y[0,-1] = 1

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_energy_struct_vec(10.)

        self.assertEqual(ws.calc_energy(y), 10.)

    def test_build_M_natural_natural(self):
        d0 = dN = 0
        x = np.array([-1, -0.5, 0, 0.5, 1])
        dx = x[1] - x[0]

        y = np.array([0.05138434, 0.01790244, -0.26065088, -0.19016379,
                      -0.76379542, d0, dN])

        # M = workerSplines.build_M(len(x), dx, bc_type=('natural', 'natural'))
        M = src.workerSplines2.build_M(len(x), dx, bc_type=('natural','natural'))

        true = np.array([0.13719161, -0.47527463, -0.10830441,
                         -0.33990513, -1.55094231])

        np.testing.assert_allclose(true, M @ y, atol=1e-7, rtol=0)

    def test_build_M_natural_fixed(self):
        d0 = dN = 0
        x = np.linspace(-1, 1, 5)
        dx = x[1] - x[0]

        y = np.array([0.05138434, 0.01790244, -0.26065088, -0.19016379,
                      -0.76379542, d0, dN])

        M = workerSplines.build_M(len(x), dx, bc_type=('natural', 'fixed'))

        true = np.array([0.15318071, -0.50725283, 0.00361927, -0.75562163, 0.])

        np.set_printoptions(precision=15)
        np.testing.assert_allclose(true, M @ y, atol=1e-8, rtol=0)

    def test_build_M_fixed_natural(self):
        d0 = dN = 0
        x = np.array([-1, -0.5, 0, 0.5, 1])
        dx = x[1] - x[0]

        y = np.array([0.05138434, 0.01790244, -0.26065088, -0.19016379,
                      -0.76379542, d0, dN])

        # M = workerSplines.build_M(len(x), dx, bc_type=('fixed', 'natural'))
        M = src.workerSplines2.build_M(len(x), dx, bc_type=('fixed', 'natural'))

        true = np.array([0, -0.43850162, -0.11820483, -0.33707643, -1.55235667])

        np.testing.assert_allclose(true, M @ y, atol=1e-8, rtol=0)

    def test_build_M_fixed_fixed(self):
        d0 = dN = 0
        x = np.array([-1, -0.5, 0, 0.5, 1])
        dx = x[1] - x[0]
        y = np.array([0.05138434, 0.01790244, -0.26065088, -0.19016379,
                      -0.76379542, d0, dN])

        # M = workerSplines.build_M(len(x), dx, bc_type=('fixed', 'fixed'))
        M = src.workerSplines2.build_M(len(x), dx, bc_type=('fixed', 'fixed'))

        true = np.array([0.00000000e+00, -4.66222277e-01, -7.32221143e-03,
                         -7.52886257e-01, -8.88178420e-16])

        np.testing.assert_allclose(true, M @ y, atol=1e-8, rtol=0)

    def test_build_M_bad_LHS(self):
        self.assertRaises(ValueError, workerSplines.build_M, len(self.x),
                          self.dx, bc_type=('bad', 'natural'))

    def test_build_M_bad_RHS(self):
        self.assertRaises(ValueError, workerSplines.build_M, len(self.x),
                          self.dx, bc_type=('natural', 'bad'))

class USplineTests(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(10)
        self.y = np.arange(1, 13)

        d0 = dN = 1
        self.y[-2] = d0
        self.y[-1] = dN
        self.y = np.atleast_2d(self.y)

        self.s = USpline(self.x, ('fixed', 'fixed'), 5)
        self.s.y = self.y

    def test_reset(self):
        self.s.atoms_embedded = 100
        self.s.deriv_struct_vec[:] = 0
        self.s.energy_struct_vec[:] = 0

    def test_zero_point_energy(self):
        # Should evaluate to # of evaluations (e.g. 4 fake atoms = result of 4)
        self.s.atoms_embedded = 3
        self.assertEqual(np.sum(self.s.compute_zero_potential(self.y, 3)), 3)

class RhoSplineTests(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(10)
        self.y = np.arange(12)

        d0 = dN = 1
        self.y[-2] = d0
        self.y[-1] = dN
        self.y = np.atleast_2d(self.y)

        self.s = RhoSpline(self.x, ('fixed', 'fixed'), 5)
        self.s.y = self.y

    def test_compute_zeros(self):
        self.s.add_to_energy_struct_vec(np.zeros(5), [[0,1]]*2 + [[1,0]]*3)

        results = self.s.calc_energy(self.y)

        np.testing.assert_allclose(np.sum(results), 0.)

    def test_compute_ones(self):

        self.s.add_to_energy_struct_vec(np.ones(5),
                                 [[0,1], [0,1], [1,0], [1,0], [1,0]])

        results = self.s.calc_energy(self.y)

        np.testing.assert_allclose(np.sum(results[0]), 5.)
