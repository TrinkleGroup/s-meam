import unittest
import numpy as np

import workerSplines

from spline import Spline
from workerSplines import WorkerSpline, RhoSpline

DIGITS = 15
EPS = 1e-15

class WorkerSplineTests(unittest.TestCase):

    def setUp(self):
        # self.x = np.linspace(-1, 1, 5)
        self.x = np.array([-1, -0.5, 0, 0.5, 1])
        self.dx = self.x[1] - self.x[0]

        self.y = np.array([1, .25, 0, .25, 1, -2, 2])

    def test_y_setter(self):

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))

        ws.y = self.y

        np.testing.assert_allclose(ws.y, self.y[:-2], atol=EPS, rtol=0)
        np.testing.assert_allclose(ws.end_derivs, self.y[-2:], atol=EPS, rtol=0)
        np.testing.assert_allclose(ws.y1, np.array([-2,-1,0,1,2]))

    def test_full_eval(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)),
                    end_derivs=(d0,dN))

        test_x = np.linspace(-10, 20, 1000)

        ws.add_to_struct_vec(test_x)

        results = ws(self.y)

        for i in range(len(test_x)):
            print(test_x[i])
            self.assertAlmostEqual(results[i], cs(test_x[i]), DIGITS)

    def test_rhs_extrap(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)),
                    end_derivs=(d0,dN))

        test_x = np.linspace(self.x[-1], self.x[-1]+2, 100)
        # test_x = np.array([1.5])

        ws.add_to_struct_vec(test_x)

        results = ws(self.y)

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]), DIGITS)

    def test_lhs_extrap(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)),
                    end_derivs=(d0,dN))

        test_x = np.linspace(self.x[0]-2, self.x[0], 1000)

        ws.add_to_struct_vec(test_x)

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

        test_x = np.linspace(self.x[0], self.x[1], 100)

        for i in range(len(test_x)):
            ws.add_to_struct_vec(test_x[i])

        results = ws(self.y)

        for i in range(len(test_x)):
            self.assertAlmostEqual(results[i], cs(test_x[i]))

    def test_rightmost_interval(self):
        d0, dN = self.y[-2:]

        ws = WorkerSpline(self.x, ('fixed', 'fixed'))
        cs = Spline(self.x, self.y[:-2], bc_type=((1,d0), (1,dN)))

        test_x = np.linspace(self.x[-2], self.x[-1]-0.01, 100)

        ws.add_to_struct_vec(test_x)

        results = ws(self.y)

        for i in range(len(test_x)):
            print(i)
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
        np.testing.assert_allclose(ws.struct_vecs[0], true, atol=EPS, rtol=0)

    def test_get_abcd_lhs_extrap(self):
        x = np.arange(3)
        r = -0.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(r)

        true = np.array([1, 0, 0, -0.5, 0, 0]).reshape((1,6))
        np.testing.assert_allclose(ws.struct_vecs[0], true, atol=EPS, rtol=0)

    def test_get_abcd_rhs_extrap(self):
        x = np.arange(3)
        r = 2.5

        ws = WorkerSpline(x, ('fixed', 'fixed'))
        ws.add_to_struct_vec(r)

        true = np.array([0, 0, 1, 0, 0, 0.5]).reshape((1,6))
        np.testing.assert_allclose(ws.struct_vecs[0], true, atol=EPS, rtol=0)

    def test_eval_flat(self):
        x = np.arange(10, dtype=float)
        y = np.zeros(12)

        ws = WorkerSpline(x, ('natural', 'natural'))
        ws.add_to_struct_vec(4)

        self.assertEqual(ws(y), 0.)

    def test_eval_internal_sloped(self):
        x = np.arange(10, dtype=float)
        y = np.arange(12, dtype=float); y[-2] = y[-1] = 1

        test_x = np.linspace(0, 9, 100)

        # TODO: should be same with 'nat'
        ws = WorkerSpline(x, ('fixed', 'fixed'))

        ws.add_to_struct_vec(test_x)

        for i in range(100):
            np.testing.assert_allclose(np.sum(ws(y)), np.sum(test_x), atol=EPS, rtol=0)

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
        ws.add_to_struct_vec(-2)

        np.testing.assert_allclose(ws(y), [-1., -2])

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

    def test_build_M_natural_natural(self):
        d0 = dN = 0
        x = np.array([-1, -0.5, 0, 0.5, 1])
        dx = x[1]-x[0]

        y = np.array([ 0.05138434,  0.01790244, -0.26065088, -0.19016379,
                            -0.76379542, d0, dN])

        M = workerSplines.build_M(len(x), dx, bc_type=('natural', 'natural'))

        true = np.array([ 0.13719161, -0.47527463, -0.10830441, -0.33990513,
                          -1.55094231])

        np.testing.assert_allclose(true, M@y, atol=1e-7, rtol=0)

    def test_build_M_natural_fixed(self):
        d0 = dN = 0
        x = np.linspace(-1,1,5)
        dx = x[1]-x[0]

        y = np.array([ 0.05138434,  0.01790244, -0.26065088, -0.19016379,
                            -0.76379542, d0, dN])

        M = workerSplines.build_M(len(x), dx, bc_type=('natural', 'fixed'))

        true = np.array([0.15318071, -0.50725283, 0.00361927, -0.75562163, 0.])

        np.set_printoptions(precision=15)
        np.testing.assert_allclose(true, M@y, atol=1e-8, rtol=0)

    def test_build_M_fixed_natural(self):
        d0 = dN = 0
        x = np.array([-1, -0.5, 0, 0.5, 1])
        dx = x[1]-x[0]

        y = np.array([ 0.05138434,  0.01790244, -0.26065088, -0.19016379,
                            -0.76379542, d0, dN])

        M = workerSplines.build_M(len(x), dx, bc_type=('fixed', 'natural'))

        true = np.array([0, -0.43850162, -0.11820483, -0.33707643, -1.55235667])

        np.testing.assert_allclose(true, M@y, atol=1e-8, rtol=0)

    def test_build_M_fixed_fixed(self):
        d0 = dN = 0
        x = np.array([-1, -0.5, 0, 0.5, 1])
        dx = x[1]-x[0]
        y = np.array([ 0.05138434,  0.01790244, -0.26065088, -0.19016379,
                            -0.76379542, d0, dN])

        M = workerSplines.build_M(len(x), dx, bc_type=('fixed',
                                                                'fixed'))

        true = np.array([0.00000000e+00, -4.66222277e-01, -7.32221143e-03,
                            -7.52886257e-01, -8.88178420e-16])

        np.testing.assert_allclose(true, M@y, atol=1e-8, rtol=0)

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

class WorkerSplineVsMEAMTests(unittest.TestCase):

    def setUp(self):

        import numpy as np
        import meam
        from worker import Worker
        from tests.testPotentials import get_random_pots

        np.random.seed(42)
        p = get_random_pots(1)['meams'][0]

        self.m_phis = p.phis
        self.m_rhos = p.rhos
        self.m_us = p.us
        self.m_fs = p.fs
        self.m_gs = p.gs

        x_pvec, y_pvec, x_indices = meam.splines_to_pvec(p.splines)

        y_indices = [x_indices[i]+2*i for i in range(len(x_indices))]



        params_split = np.split(y_pvec, y_indices[1:])

        nphi = 3
        ntypes = 2

        split_indices = [nphi, nphi+ntypes, nphi+2*ntypes, nphi+3*ntypes]

        self.phi_ys, self.rho_ys, self.u_ys, self.f_ys, self.g_ys = np.split(
            params_split, split_indices)

        ntypes = 2
        nphi = 3

        knots_split = np.split(x_pvec, x_indices[1:])

        splines = []

        for i in range(ntypes*(ntypes+4)):
            idx = x_indices[i]

            bc_type = ('fixed', 'fixed')

            s = WorkerSpline(knots_split[i], bc_type)

            s.index = idx

            splines.append(s)

        split_indices = np.array([nphi, nphi+ntypes, nphi+2*ntypes,
                                 nphi+3*ntypes])
        self.w_phis, self.w_rhos, self.w_us, self.w_fs, self.w_gs =\
            np.split(splines, split_indices)

        self.w_phis = list(self.w_phis)
        self.w_rhos = list(self.w_rhos)
        self.w_us = list(self.w_us)
        self.w_fs = list(self.w_fs)
        self.gs = list(self.w_gs)

    def test_phis(self):
        for i in range(len(self.w_phis)):
            w_phi = self.w_phis[i]
            m_phi = self.m_phis[i]

            test_vals = np.linspace(w_phi.x[0]-1, w_phi.x[-1]+1, 100)

            w_phi.add_to_struct_vec(test_vals)
            y_pvec = self.phi_ys[i]

            w_results = w_phi(y_pvec)

            for j in range(100):
                # print(test_vals[j])
                print(j)
                if j == 86:
                    self.assertAlmostEqual(w_results[j], m_phi(test_vals[j]), DIGITS)

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
        self.assertEqual(self.s.struct_vec_dict[0][0].shape[0], 2)
        self.assertEqual(self.s.struct_vec_dict[1][0].shape[0], 3)

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
