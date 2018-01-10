import unittest
import numpy as np

import meam
import workers

from meam import MEAM
from workers import Worker, WorkerSpline

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
    bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo

EPS = 1e-7

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

    # rzm: write worker. figure out tests

    def test_write_one_to_file(self):
        pass

    def test_write_many_to_file(self):
        pass

    def test_write_atoms_to_file(self):
        pass

class EvaluationTests(unittest.TestCase):

    def test_dimer_phionly(self):
        p = MEAM.from_file("../data/pot_files/HHe.meam.spline")

        x_pvec, y_pvec, indices = meam.splines_to_pvec(p.splines)

        w = Worker(dimers['aa'], x_pvec, indices, p.types)

        true = -1.25788918416789

        self.assertAlmostEqual(w.compute_energies(y_pvec), true)

    # Zero potential (single)
    def test_zero_phionly(self):
        pass

    def test_zero_nophi(self):
        pass

    def test_zero_rhophi(self):
        pass

    def test_zero_norhophi(self):
        pass

    def test_zero_rho(self):
        pass

    def test_zero_norho(self):
        pass

    # Constant potential (single)
    def test_zero_phionly(self):
        pass

    def test_zero_nophi(self):
        pass

    def test_zero_rhophi(self):
        pass

    def test_zero_norhophi(self):
        pass

    def test_zero_rho(self):
        pass

    def test_zero_norho(self):
        pass

    # Random potential (single)
    def test_zero_phionly(self):
        pass

    def test_zero_nophi(self):
        pass

    def test_zero_rhophi(self):
        pass

    def test_zero_norhophi(self):
        pass

    def test_zero_rho(self):
        pass

    # Zero potential (many)
    def test_zero_phionly(self):
        pass

    def test_zero_nophi(self):
        pass

    def test_zero_rhophi(self):
        pass

    def test_zero_norhophi(self):
        pass

    def test_zero_rho(self):
        pass

    def test_zero_norho(self):
        pass

    # Constant potential (many)
    def test_zero_phionly(self):
        pass

    def test_zero_nophi(self):
        pass

    def test_zero_rhophi(self):
        pass

    def test_zero_norhophi(self):
        pass

    def test_zero_rho(self):
        pass

    def test_zero_norho(self):
        pass

    # Random potential (many)
    def test_zero_phionly(self):
        pass

    def test_zero_nophi(self):
        pass

    def test_zero_rhophi(self):
        pass

    def test_zero_norhophi(self):
        pass

    def test_zero_rho(self):
        pass

class WorkerSplineTests(unittest.TestCase):


    def setUp(self):
        self.x = np.linspace(-1, 1, 5)
        self.dx = self.x[1] - self.x[0]

        d0 = dN = 0
        self.y = np.array([ 0.05138434,  0.01790244, -0.26065088, -0.19016379,
                       -0.76379542, d0, dN])

    def test_constructor(self):
        pass

    def test_constructor_bad_x(self):
        pass

    def test_constructor_bad_bc(self):
        pass

    def test_eval(self):
        #rzm: make trimer and test phionly; take from tests.structs
        pass

    def test_build_M_natural_natural(self):
        M = workers.build_M(len(self.x), self.dx, bc_type=('natural',
                                                                'natural'))

        true = np.array([ 0.13719161, -0.47527463, -0.10830441, -0.33990513,
                          -1.55094231])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_natural_fixed(self):
        M = workers.build_M(len(self.x), self.dx, bc_type=('natural',
                                                                'fixed'))

        true = np.array([0.15318071, -0.50725283, 0.00361927, -0.75562163, 0.])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_fixed_natural(self):
        M = workers.build_M(len(self.x), self.dx, bc_type=('fixed',
                                                                'natural'))

        true = np.array([0, -0.43850162, -0.11820483, -0.33707643, -1.55235667])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_fixed_fixed(self):
        M = workers.build_M(len(self.x), self.dx, bc_type=('fixed',
                                                                'fixed'))

        true = np.array([0.00000000e+00, -4.66222277e-01, -7.32221143e-03,
                            -7.52886257e-01, -8.88178420e-16])

        np.testing.assert_allclose(true, M@self.y, atol=EPS)

    def test_build_M_bad_LHS(self):
        self.assertRaises(ValueError, workers.build_M, len(self.x),
                          self.dx, bc_type=('bad', 'natural'))

    def test_build_M_bad_RHS(self):
        self.assertRaises(ValueError, workers.build_M, len(self.x),
                          self.dx, bc_type=('natural', 'bad'))
