import unittest
import numpy as np

import workerSplines

from spline import Spline
from workerSplines import WorkerSpline, RhoSpline

EPS = 1e-15

class WorkerSplineTests(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-1, 1, 5)
        self.dx = self.x[1] - self.x[0]

        d0 = dN = 0
        self.y = np.array([ 0.05138434,  0.01790244, -0.26065088, -0.19016379,
                       -0.76379542, d0, dN])

    def test_y_setter(self):
        ws = WorkerSpline(self.x, ('fixed', 'fixed'))

        ws.y = self.y

        np.testing.assert_allclose(ws.y, self.y[:-2])
        np.testing.assert_allclose(ws.end_derivs, self.y[-2:], EPS)
        np.testing.assert_allclose(ws.y1, np.array([ 0., -0.46622228,
                                 -0.00732221, -0.75288626, 0.]), atol=1e-7)

    def test_full_eval(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(-10, 20, 1000)

        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]))

    def test_rhs_extrap(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(self.x[-1], self.x[-1]*1.5, 1000)

        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]))

    def test_lhs_extrap(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(self.x[0]*0.5, self.x[0], 1000)

        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]))

    def test_inner_intervals(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(self.x[1], self.x[-2], 1000)

        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]))

    def test_leftmost_interval(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(self.x[0], self.x[1], 1000)

        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]))

    def test_rightmost_interval(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(self.x[-2], self.x[-1], 1000)

        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)
        results = np.zeros(test_x.shape)
        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)

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

        true = np.array([0.5, 0.5, 0, 0.125, -0.125, 0]).reshape((1,6))
        np.testing.assert_allclose(ws.struct_vec, true)

    def test_get_abcd_lhs_extrap(self):
        x = np.arange(3)
        r = -0.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(r)

        true = np.array([1, 0, 0, -0.5, 0, 0]).reshape((1,6))
        np.testing.assert_allclose(ws.struct_vec, true)

    def test_get_abcd_rhs_extrap(self):
        x = np.arange(3)
        r = 2.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(r)

        true = np.array([0, 0, 1, 0, 0, 0.5]).reshape((1,6))
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
        M = workerSplines.build_M(len(self.x), self.dx, bc_type=('natural',
                                                                'natural'))

        true = np.array([ 0.13719161, -0.47527463, -0.10830441, -0.33990513,
                          -1.55094231])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_natural_fixed(self):
        M = workerSplines.build_M(len(self.x), self.dx, bc_type=('natural',
                                                                'fixed'))

        true = np.array([0.15318071, -0.50725283, 0.00361927, -0.75562163, 0.])

        np.testing.assert_allclose(true, M@self.y, atol=1e-7)

    def test_build_M_fixed_natural(self):
        M = workerSplines.build_M(len(self.x), self.dx, bc_type=('fixed',
                                                                'natural'))

        true = np.array([0, -0.43850162, -0.11820483, -0.33707643, -1.55235667])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_fixed_fixed(self):
        M = workerSplines.build_M(len(self.x), self.dx, bc_type=('fixed',
                                                                'fixed'))

        true = np.array([0.00000000e+00, -4.66222277e-01, -7.32221143e-03,
                            -7.52886257e-01, -8.88178420e-16])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_bad_LHS(self):
        self.assertRaises(ValueError, workerSplines.build_M, len(self.x),
                          self.dx, bc_type=('bad', 'natural'))

    def test_build_M_bad_RHS(self):
        self.assertRaises(ValueError, workerSplines.build_M, len(self.x),
                          self.dx, bc_type=('natural', 'bad'))

    def test_zero_point_energy(self):
        # Should evaluate to # of evaluations (e.g. 4 fake atoms = result of 4)
        x = np.arange(10)

        y = np.arange(1, 13)
        d0 = dN = 1
        y[-2] = d0; y[-2] = dN

        ws = WorkerSpline(x, ('fixed', 'fixed'))

        ws.add_to_struct_vec(0)
        ws.add_to_struct_vec(0)
        ws.add_to_struct_vec(0)

        self.assertEqual(np.sum(ws(y)), 3)

class RhoSplineTests(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(10)
        self.y = np.arange(12)

        d0 = dN = 1
        self.y[-2] = d0; self.y[-1] = dN

        self.s = RhoSpline(self.x, ('fixed', 'fixed'), 5)
        self.s.y = self.y

    def test_update_dict(self):
        self.s.update_struct_vec_dict(0., 0)
        self.s.update_struct_vec_dict(0., 0)

        self.s.update_struct_vec_dict(0., 1)
        self.s.update_struct_vec_dict(0., 1)
        self.s.update_struct_vec_dict(0., 1)

        self.assertEquals(len(self.s.struct_vec_dict), 5)
        self.assertEqual(self.s.struct_vec_dict[0].shape[0], 2)
        self.assertEqual(self.s.struct_vec_dict[1].shape[0], 3)

    def test_compute_zeros(self):
        self.s.update_struct_vec_dict(0., 0)
        self.s.update_struct_vec_dict(0., 0)

        self.s.update_struct_vec_dict(0., 1)
        self.s.update_struct_vec_dict(0., 1)
        self.s.update_struct_vec_dict(0., 1)

        results = self.s.compute_for_all(self.y)

        np.testing.assert_allclose(np.sum(results[0]) , 0.)
        np.testing.assert_allclose(np.sum(results[1]), 0.)


    def test_compute_ones(self):
        self.s.update_struct_vec_dict(1., 0)
        self.s.update_struct_vec_dict(1., 0)

        self.s.update_struct_vec_dict(1., 1)
        self.s.update_struct_vec_dict(1., 1)
        self.s.update_struct_vec_dict(1., 1)

        results = self.s.compute_for_all(self.y)

        np.testing.assert_allclose(np.sum(results[0]), 2.)
        np.testing.assert_allclose(np.sum(results[1]), 3.)
