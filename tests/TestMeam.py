import unittest
import numpy as np
import os

import meam

from meam import MEAM
from spline import Spline, ZeroSpline

from tests.testStructs import dimers, trimers, bulk_periodic_ortho,\
    bulk_vac_ortho, bulk_periodic_rhombo, bulk_vac_rhombo

DIGITS = 15
EPS = 1e-15

class ConstructorTests(unittest.TestCase):

    def setUp(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        self.types = ['H', 'He']

        self.splines = [Spline(x, y, end_derivs=(1.,1.))]*13

        self.x_pvec = np.array((list(x)+[1,1])*12)
        self.y_pvec = np.array(list(y)*12)
        self.x_indices = np.array([12*i for i in range(1,12)])

    def test_splines_good(self):
        p = MEAM(self.splines[:12], self.types)

        self.assertEqual(3, len(p.phis))
        self.assertEqual(2, len(p.rhos))
        self.assertEqual(2, len(p.us))
        self.assertEqual(2, len(p.fs))
        self.assertEqual(3, len(p.gs))

        self.assertEqual(3, p.nphi)
        self.assertEqual(['H', 'He'], p.types)
        self.assertEqual(2, p.ntypes)
        self.assertAlmostEqual(9., p.cutoff)
        self.assertAlmostEqual(1., p.phis[0](p.phis[0].x[0],1))
        self.assertAlmostEqual(1., p.phis[0](p.phis[0].x[-1],1))

        # TODO: these tests should check correct results from trimer, inp/outp

    def test_splines_unary_not_enough(self):
        self.assertRaises(ValueError, MEAM, self.splines[:4],
                          self.types[0])

    def test_splines_unary_too_many(self):
        self.assertRaises(ValueError, MEAM, self.splines[:6],
                          self.types[0])

    def test_splines_binary_not_enough(self):
        self.assertRaises(ValueError, MEAM, self.splines[:11],
                          self.types)

    def test_splines_binary_too_many(self):
        self.assertRaises(ValueError, MEAM, self.splines,
                          self.types)

    def test_splines_ternary_not_implemented(self):
        self.assertRaises(NotImplementedError, MEAM, self.splines,
                          self.types+['Li'])

    def test_pvec_good(self):
        p = MEAM.from_pvec(self.x_pvec, self.y_pvec, self.x_indices, self.types)

        self.assertEqual(len(p.phis), 3)
        self.assertEqual(len(p.rhos), 2)
        self.assertEqual(len(p.us), 2)
        self.assertEqual(len(p.fs), 2)
        self.assertEqual(len(p.gs), 3)

        self.assertEqual(p.nphi, 3)
        self.assertEqual(p.types, ['H', 'He'])
        self.assertEqual(p.ntypes, 2)
        self.assertAlmostEqual(p.cutoff, 9.)
        self.assertAlmostEqual(1., p.phis[0](p.phis[0].x[0],1))
        self.assertAlmostEqual(1., p.phis[0](p.phis[0].x[-1],1))

    def test_pvec_mismatch_xy(self):
        self.assertRaises(ValueError, MEAM.from_pvec, self.x_pvec,
                          self.y_pvec[:-1], self.x_indices, self.types)

    def test_from_file(self):
        p = MEAM.from_file('../data/pot_files/TiO.meam.spline')

        self.assertEqual(3, len(p.phis))
        self.assertEqual(2, len(p.rhos))
        self.assertEqual(2, len(p.us))
        self.assertEqual(2, len(p.fs))
        self.assertEqual(3, len(p.gs))

        self.assertEqual(3, p.nphi)
        self.assertEqual(['Ti', 'O'], p.types)
        self.assertEqual(2, p.ntypes)
        self.assertAlmostEqual(5.5, p.cutoff)
        self.assertAlmostEqual(-20., p.phis[0](p.phis[0].x[0],1))
        self.assertAlmostEqual(0., p.phis[0](p.phis[0].x[-1],1))

    def test_splines_matches_pvec(self):
        p_splines  = MEAM(self.splines[:12], self.types)
        p_pvec = MEAM.from_pvec(self.x_pvec, self.y_pvec, self.x_indices,
                            self.types)

        self.assertEqual(len(p_splines.phis), len(p_pvec.phis))
        self.assertEqual(len(p_splines.rhos), len(p_pvec.rhos))
        self.assertEqual(len(p_splines.us), len(p_pvec.us))
        self.assertEqual(len(p_splines.fs), len(p_pvec.fs))
        self.assertEqual(len(p_splines.gs), len(p_pvec.gs))

        self.assertEqual(p_splines.nphi, p_pvec.nphi)
        self.assertEqual(p_splines.types, p_pvec.types)
        self.assertEqual(p_splines.cutoff, p_pvec.cutoff)

        group_splines =  p_splines.phis + p_splines.rhos + p_splines.us + \
                         p_splines.fs + p_splines.gs
        group_pvec =  p_pvec.phis + p_pvec.rhos + p_pvec.us + p_pvec.fs + \
                      p_pvec.gs

        for i in range(12):
            s_splines = group_splines[i]
            s_pvec = group_pvec[i]

            self.assertTrue(s_splines == s_pvec)

class MethodTests(unittest.TestCase):

    def test_to_file(self):
        # note: this only really tests that nothing changes between read/write

        # override maximum string compare length
        self.maxDiff = None

        p = MEAM.from_file('../data/pot_files/TiO.meam.spline')
        p.write_to_file('test.write')

        true = open('test.write', 'r').readlines()[1:]

        p = MEAM.from_file('test.write')
        new = open('test.write', 'r').readlines()[1:]

        self.assertEqual(new, true)

        os.remove('test.write')

    def test_splines_to_pvec(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        splines = [Spline(x, y)]*12

        x_pvec = np.array((list(x))*12)
        y_pvec = np.array((list(y)+[1,1])*12)
        x_indices = np.array([10*i for i in range(12)])

        new_x_pvec, new_y_pvec, new_x_indices = meam.splines_to_pvec(splines)

        np.testing.assert_allclose(new_x_pvec, x_pvec, atol=1e-15)
        np.testing.assert_allclose(new_y_pvec, y_pvec, atol=1e-15)
        np.testing.assert_allclose(new_x_indices, x_indices, atol=1e-15)

    def test_splines_to_vec_file(self):
        p = MEAM.from_file('../data/pot_files/TiO.meam.spline')

        x_pvec, y_pvec, indices = meam.splines_to_pvec(p.splines)

        np.testing.assert_allclose(x_pvec[13:18], np.array([1.9, 2.8, 3.7,
                                                            4.6, 5.5]))

    def test_splines_from_pvec(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        d0 = dN = 1

        old_splines = [Spline(x, y, bc_type=((1,d0),(1,dN)), end_derivs=(
            d0,dN))]*12

        x_pvec = np.array((list(x)+[d0,dN])*12)
        y_pvec = np.array(list(y)*12)
        x_indices = np.array([12*i for i in range(1,12)])

        new_splines = meam.splines_from_pvec(x_pvec, y_pvec, x_indices)

        for i in range(len(old_splines)):
            self.assertTrue(new_splines[i] == old_splines[i])

    def test_i_to_potl_good(self):
        self.assertEqual(meam.i_to_potl(1), 0)
        self.assertEqual(meam.i_to_potl(1000), 999)

    def test_i_to_potl_bad(self):
        self.assertRaises(ValueError, meam.i_to_potl, 0)
        self.assertRaises(ValueError, meam.i_to_potl, -1)

    def test_ij_to_potl_good(self):
        self.assertEqual(0, meam.ij_to_potl(1,1,2))
        self.assertEqual(1, meam.ij_to_potl(1,2,2))
        self.assertEqual(1, meam.ij_to_potl(2,1,2))
        self.assertEqual(2, meam.ij_to_potl(2,2,2))

    def test_ij_to_potl_bad(self):
        self.assertRaises(ValueError, meam.ij_to_potl, 0, 1, 2)
        self.assertRaises(ValueError, meam.ij_to_potl, 1, 0, 2)

    def test_ij_to_potl_ternary_unimplemented(self):
        self.assertRaises(NotImplementedError, meam.ij_to_potl, 1, 1, 3)

    def test_ij_to_potl_bad(self):
        self.assertRaises(ValueError, meam.i_to_potl, 0)
        self.assertRaises(ValueError, meam.i_to_potl, -1)

    def test_plot(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).phionly_subtype()

        p.plot('test')

        for i in range(1,13):
            os.remove('test%i.png' % i)

    def test_phionly_subtype(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).phionly_subtype()

        all_splines = p.phis + p.rhos + p.us + p.fs + p.gs

        for i in range(12):
            if i < 3:
                self.assertTrue(isinstance(all_splines[i], Spline))
            else:
                self.assertTrue(isinstance(all_splines[i], ZeroSpline))

    def test_nophi_subtype(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).nophi_subtype()

        all_splines = p.phis + p.rhos + p.us + p.fs + p.gs

        for i in range(12):
            if i >= 3:
                self.assertTrue(isinstance(all_splines[i], Spline))
            else:
                self.assertTrue(isinstance(all_splines[i], ZeroSpline))

    def test_rhophi_subtype(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).rhophi_subtype()

        all_splines = p.phis + p.rhos + p.us + p.fs + p.gs

        for i in range(12):
            if i < 7:
                self.assertTrue(isinstance(all_splines[i], Spline))
            else:
                self.assertTrue(isinstance(all_splines[i], ZeroSpline))

    def test_norhophi_subtype(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).norhophi_subtype()

        all_splines = p.phis + p.rhos + p.us + p.fs + p.gs

        for i in range(12):
            if i >= 5:
                self.assertTrue(isinstance(all_splines[i], Spline))
            else:
                self.assertTrue(isinstance(all_splines[i], ZeroSpline))

    def test_norho_subtype(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).norho_subtype()

        all_splines = p.phis + p.rhos + p.us + p.fs + p.gs

        for i in range(12):
            if (i<3) or (i>=5):
                self.assertTrue(isinstance(all_splines[i], Spline))
            else:
                self.assertTrue(isinstance(all_splines[i], ZeroSpline))

    def test_rho_subtype(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).rho_subtype()

        all_splines = p.phis + p.rhos + p.us + p.fs + p.gs

        for i in range(12):
            if (i>2) and (i<7):
                self.assertTrue(isinstance(all_splines[i], Spline))
            else:
                self.assertTrue(isinstance(all_splines[i], ZeroSpline))

    def test_nog_subtype(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        types = ['H', 'He']

        splines = [Spline(x, y)]*12

        p = MEAM(splines, types).nog_subtype()

        all_splines = p.phis + p.rhos + p.us + p.fs + p.gs

        for i in range(12):
            if i <= 10:
                self.assertTrue(isinstance(all_splines[i], Spline))
            else:
                self.assertTrue(isinstance(all_splines[i], ZeroSpline))

class EnergyTests(unittest.TestCase):

    def setUp(self):
        self.p = MEAM.from_file("../data/pot_files/HHe.meam.spline")

    def test_energy_dimers(self):

        for name in dimers.keys():
            atoms = dimers[name]

            guess = self.p.compute_energy(atoms)
            true = self.p.get_lammps_results(atoms)['energy']

        self.assertAlmostEqual(guess, true, places=DIGITS)

    def test_energy_trimers(self):

        for name in trimers.keys():
            atoms = trimers[name]

            guess = self.p.compute_energy(atoms)
            true = self.p.get_lammps_results(atoms)['energy']

        self.assertAlmostEqual(guess, true, places=DIGITS)

    def test_energy_bulk_vac_rhombo(self):

        for name in bulk_vac_rhombo.keys():
            atoms = bulk_vac_rhombo[name]

            guess = self.p.compute_energy(atoms)
            true = self.p.get_lammps_results(atoms)['energy']

        self.assertAlmostEqual(guess, true, places=DIGITS)

    def test_energy_bulk_periodic_rhombo(self):

        for name in bulk_periodic_rhombo.keys():
            atoms = bulk_periodic_rhombo[name]

            guess = self.p.compute_energy(atoms)
            true = self.p.get_lammps_results(atoms)['energy']

        self.assertAlmostEqual(guess, true, places=DIGITS)

    def test_energy_bulk_periodic_ortho(self):

        for name in bulk_periodic_ortho.keys():
            atoms = bulk_periodic_ortho[name]

            guess = self.p.compute_energy(atoms)
            true = self.p.get_lammps_results(atoms)['energy']

        self.assertAlmostEqual(guess, true, places=DIGITS)

    def test_energy_bulk_vac_ortho(self):

        for name in bulk_vac_ortho.keys():
            atoms = bulk_vac_ortho[name]

            guess = self.p.compute_energy(atoms)
            true = self.p.get_lammps_results(atoms)['energy']

        self.assertAlmostEqual(guess, true, places=DIGITS)

class ForcesTests(unittest.TestCase):

    def setUp(self):
        self.p = MEAM.from_file("../data/pot_files/HHe.meam.spline")

    def test_forces_dimer(self):

        tmp_dimers = {'aa':dimers['aa']}

        for name in tmp_dimers.keys():
            atoms = tmp_dimers[name]

            guess = self.p.compute_forces(atoms)
            true = self.p.get_lammps_results(atoms)['forces']

            np.testing.assert_allclose(guess, true, atol=EPS)

    def test_forces_trimer(self):

        for name in trimers.keys():
            atoms = trimers[name]

            guess = self.p.compute_forces(atoms)
            true = self.p.get_lammps_results(atoms)['forces']

            np.testing.assert_allclose(guess, true, atol=EPS)

    def test_forces_bulk_vac_ortho(self):

        for name in bulk_vac_ortho.keys():
            atoms = bulk_vac_ortho[name]

            guess = self.p.compute_forces(atoms)
            true = self.p.get_lammps_results(atoms)['forces']

            np.testing.assert_allclose(guess, true, atol=EPS)

    def test_forces_bulk_vac_rhombo(self):

        for name in bulk_vac_rhombo.keys():
            atoms = bulk_vac_rhombo[name]

            guess = self.p.compute_forces(atoms)
            true = self.p.get_lammps_results(atoms)['forces']

            np.testing.assert_allclose(guess, true, atol=EPS)

    def test_forces_bulk_periodic_ortho(self):

        for name in bulk_periodic_ortho.keys():
            atoms = bulk_periodic_ortho[name]

            guess = self.p.compute_forces(atoms)
            true = self.p.get_lammps_results(atoms)['forces']

            np.testing.assert_allclose(guess, true, atol=EPS)

    def test_forces_bulk_periodic_rhombo(self):

        for name in bulk_periodic_rhombo.keys():
            atoms = bulk_periodic_rhombo[name]

            guess = self.p.compute_forces(atoms)
            true = self.p.get_lammps_results(atoms)['forces']

            np.testing.assert_allclose(guess, true, atol=EPS)

