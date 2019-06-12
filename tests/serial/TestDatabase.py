import sys
sys.path.append('/home/jvita/scripts/s-meam/project/')
sys.path.append('/home/jvita/scripts/s-meam/project/tests/')

import os
import glob
import pickle
import unittest
import numpy as np

from src.worker import Worker
from src.database import Database
from src.potential_templates import Template

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
            bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo, extra

points_per_spline = 7
DECIMALS = 8

class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inner_cutoff = 2.1
        outer_cutoff = 5.5

        cls.x_pvec = np.concatenate([
            np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 5),
            np.tile(np.linspace(-1, 1, points_per_spline), 2),
            np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 2),
            np.tile(np.linspace(-1, 1, points_per_spline), 3)]
        )

        cls.x_indices = list(
            range(0, points_per_spline * 12, points_per_spline)
        )

        cls.types = ['H', 'He']

        cls.template = build_template('full', inner_cutoff, outer_cutoff)

        cls.db = Database(
            'db_delete.hdf5', 'w', cls.template.pvec_len, cls.types, cls.x_pvec,
            cls.x_indices, [inner_cutoff, outer_cutoff]
        )

        cls.db.add_structure('aa', dimers['aa'], overwrite=True)
        cls.db.add_structure('ab', dimers['ab'], overwrite=True)
        cls.db.add_structure('aaa', trimers['aaa'], overwrite=True)
        cls.db.add_structure('aba', trimers['aba'], overwrite=True)
        cls.db.add_structure('bbb', trimers['bbb'], overwrite=True)

        cls.db.add_structure(
            'bulk_vac_ortho_type1', bulk_vac_ortho['bulk_vac_ortho_type1'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_vac_ortho_type2', bulk_vac_ortho['bulk_vac_ortho_type2'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_vac_ortho_mixed', bulk_vac_ortho['bulk_vac_ortho_mixed'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_periodic_ortho_type1',
            bulk_periodic_ortho['bulk_periodic_ortho_type1'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_periodic_ortho_type2',
            bulk_periodic_ortho['bulk_periodic_ortho_type2'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_periodic_ortho_mixed',
            bulk_periodic_ortho['bulk_periodic_ortho_mixed'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_vac_rhombo_type1', bulk_vac_rhombo['bulk_vac_rhombo_type1'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_vac_rhombo_type2', bulk_vac_rhombo['bulk_vac_rhombo_type2'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_vac_rhombo_mixed', bulk_vac_rhombo['bulk_vac_rhombo_mixed'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_periodic_rhombo_type1',
            bulk_periodic_rhombo['bulk_periodic_rhombo_type1'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_periodic_rhombo_type2',
            bulk_periodic_rhombo['bulk_periodic_rhombo_type2'],
            overwrite=True
        )

        cls.db.add_structure(
            'bulk_periodic_rhombo_mixed',
            bulk_periodic_rhombo['bulk_periodic_rhombo_mixed'],
            overwrite=True
        )

        cls.pvec = np.ones((5, cls.template.pvec_len))
        cls.pvec *= np.arange(1, 6)[:, np.newaxis]

    def test_energy_dimer_aa(self):
        db_eng, _ = self.db.compute_energy(
            'aa', self.pvec, self.template.u_ranges
        )

        worker = Worker(dimers['aa'], self.x_pvec, self.x_indices, self.types)

        wk_eng, _ = worker.compute_energy(self.pvec, self.template.u_ranges)

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_dimer_aa(self):
        db_fcs = self.db.compute_forces('aa', self.pvec, self.template.u_ranges)

        worker = Worker(dimers['aa'], self.x_pvec, self.x_indices, self.types)

        wk_fcs = worker.compute_forces(self.pvec, self.template.u_ranges)

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_dimer_aa(self):
        db_grad = self.db.compute_energy_grad(
            'aa', self.pvec, self.template.u_ranges
        )

        worker = Worker(dimers['aa'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_dimer_aa(self):
        db_grad = self.db.compute_forces_grad(
            'aa', self.pvec, self.template.u_ranges
        )

        worker = Worker(dimers['aa'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_dimer_ab(self):
        db_eng, _ = self.db.compute_energy(
            'ab', self.pvec, self.template.u_ranges
        )

        worker = Worker(dimers['ab'], self.x_pvec, self.x_indices, self.types)

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_dimer_ab(self):
        db_fcs = self.db.compute_forces('ab', self.pvec, self.template.u_ranges)

        worker = Worker(dimers['ab'], self.x_pvec, self.x_indices, self.types)

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_dimer_ab(self):
        db_grad = self.db.compute_energy_grad(
            'ab', self.pvec, self.template.u_ranges
        )

        worker = Worker(dimers['ab'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_dimer_ab(self):
        db_grad = self.db.compute_forces_grad(
            'ab', self.pvec, self.template.u_ranges
        )

        worker = Worker(dimers['ab'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_trimer_aaa(self):
        db_eng, _ = self.db.compute_energy(
            'aaa', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['aaa'], self.x_pvec, self.x_indices, self.types)

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_trimer_aaa(self):
        db_fcs = self.db.compute_forces('aaa', self.pvec, self.template.u_ranges)

        worker = Worker(trimers['aaa'], self.x_pvec, self.x_indices, self.types)

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_trimer_aaa(self):
        db_grad = self.db.compute_energy_grad(
            'aaa', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['aaa'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_trimer_aaa(self):
        db_grad = self.db.compute_forces_grad(
            'aaa', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['aaa'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_trimer_aba(self):
        db_eng, _ = self.db.compute_energy(
            'aba', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['aba'], self.x_pvec, self.x_indices, self.types)

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_trimer_aba(self):
        db_fcs = self.db.compute_forces('aba', self.pvec, self.template.u_ranges)

        worker = Worker(trimers['aba'], self.x_pvec, self.x_indices, self.types)

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_trimer_aba(self):
        db_grad = self.db.compute_energy_grad(
            'aba', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['aba'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_trimer_aba(self):
        db_grad = self.db.compute_forces_grad(
            'aba', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['aba'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_trimer_bbb(self):
        db_eng, _ = self.db.compute_energy(
            'bbb', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['bbb'], self.x_pvec, self.x_indices, self.types)

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_trimer_bbb(self):
        db_fcs = self.db.compute_forces('bbb', self.pvec, self.template.u_ranges)

        worker = Worker(trimers['bbb'], self.x_pvec, self.x_indices, self.types)

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_trimer_bbb(self):
        db_grad = self.db.compute_energy_grad(
            'bbb', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['bbb'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_trimer_bbb(self):
        db_grad = self.db.compute_forces_grad(
            'bbb', self.pvec, self.template.u_ranges
        )

        worker = Worker(trimers['bbb'], self.x_pvec, self.x_indices, self.types)

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bvo_t1(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_vac_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bvo_t1(self):
        db_fcs = self.db.compute_forces(
            'bulk_vac_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bvo_t1(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_vac_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bvo_t1(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_vac_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bvo_t2(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_vac_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bvo_t2(self):
        db_fcs = self.db.compute_forces(
            'bulk_vac_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bvo_t2(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_vac_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bvo_t2(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_vac_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bvo_mx(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_vac_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bvo_mx(self):
        db_fcs = self.db.compute_forces(
            'bulk_vac_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bvo_mx(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_vac_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bvo_mx(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_vac_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_ortho['bulk_vac_ortho_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bpo_t1(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_periodic_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bpo_t1(self):
        db_fcs = self.db.compute_forces(
            'bulk_periodic_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bpo_t1(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_periodic_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bpo_t1(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_periodic_ortho_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bpo_t2(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_periodic_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bpo_t2(self):
        db_fcs = self.db.compute_forces(
            'bulk_periodic_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bpo_t2(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_periodic_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bpo_t2(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_periodic_ortho_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bpo_mx(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_periodic_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bpo_mx(self):
        db_fcs = self.db.compute_forces(
            'bulk_periodic_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bpo_mx(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_periodic_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bpo_mx(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_periodic_ortho_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_ortho['bulk_periodic_ortho_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bvr_t1(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_vac_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bvr_t1(self):
        db_fcs = self.db.compute_forces(
            'bulk_vac_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bvr_t1(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_vac_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bvr_t1(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_vac_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type1'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bvr_t2(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_vac_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bvr_t2(self):
        db_fcs = self.db.compute_forces(
            'bulk_vac_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bvr_t2(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_vac_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bvr_t2(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_vac_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_type2'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bvr_mx(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_vac_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bvr_mx(self):
        db_fcs = self.db.compute_forces(
            'bulk_vac_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bvr_mx(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_vac_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bvr_mx(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_vac_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_vac_rhombo['bulk_vac_rhombo_mixed'], self.x_pvec, self.x_indices,
            self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bpr_t1(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_periodic_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bpr_t1(self):
        db_fcs = self.db.compute_forces(
            'bulk_periodic_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bpr_t1(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_periodic_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bpr_t1(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_periodic_rhombo_type1', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type1'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bpr_t2(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_periodic_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bpr_t2(self):
        db_fcs = self.db.compute_forces(
            'bulk_periodic_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bpr_t2(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_periodic_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bpr_t2(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_periodic_rhombo_type2', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_type2'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_energy_bpr_mx(self):
        db_eng, _ = self.db.compute_energy(
            'bulk_periodic_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_eng, _ = worker.compute_energy(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces_bpr_mx(self):
        db_fcs = self.db.compute_forces(
            'bulk_periodic_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_fcs = worker.compute_forces(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad_bpr_mx(self):
        db_grad = self.db.compute_energy_grad(
            'bulk_periodic_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.energy_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad_bpr_mx(self):
        db_grad = self.db.compute_forces_grad(
            'bulk_periodic_rhombo_mixed', self.pvec, self.template.u_ranges
        )

        worker = Worker(
            bulk_periodic_rhombo['bulk_periodic_rhombo_mixed'], self.x_pvec,
            self.x_indices, self.types
        )

        wk_grad = worker.forces_gradient_wrt_pvec(
            self.pvec, self.template.u_ranges
        )

        np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

class FromExisting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inner_cutoff = 2.1
        outer_cutoff = 5.5

        cls.x_pvec = np.concatenate([
            np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 5),
            np.tile(np.linspace(-1, 1, points_per_spline), 2),
            np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 2),
            np.tile(np.linspace(-1, 1, points_per_spline), 3)]
        )

        cls.x_indices = list(
            range(0, points_per_spline * 12, points_per_spline)
        )

        cls.types = ['H', 'He']

        cls.template = build_template('full', inner_cutoff, outer_cutoff)

        cls.db = Database(
            'db_delete_existing.hdf5', 'w', cls.template.pvec_len, cls.types,
            cls.x_pvec, cls.x_indices, [inner_cutoff, outer_cutoff],
            overwrite=True
        )

        cls.path_to_workers = \
            '/home/jvita/scripts/s-meam/data/fitting_databases/hyojung/structures'

        cls.db.add_from_existing_workers(cls.path_to_workers)

        cls.pvec = np.ones((1, cls.template.pvec_len))

    def test_energy(self):
        for struct_path in glob.glob(os.path.join(self.path_to_workers, '*')):
            worker = pickle.load(open(struct_path, 'rb'))

            struct_name = os.path.splitext(os.path.split(struct_path)[-1])[0]

            db_eng, _ = self.db.compute_energy(
                struct_name, self.pvec, self.template.u_ranges
            )

            wk_eng, _ = worker.compute_energy(self.pvec, self.template.u_ranges)

            np.testing.assert_almost_equal(wk_eng, db_eng, decimal=DECIMALS)

    def test_forces(self):
        for struct_path in glob.glob(os.path.join(self.path_to_workers, '*')):
            worker = pickle.load(open(struct_path, 'rb'))

            struct_name = os.path.splitext(os.path.split(struct_path)[-1])[0]

            db_fcs = self.db.compute_forces(
                struct_name, self.pvec, self.template.u_ranges
            )

            wk_fcs = worker.compute_forces(self.pvec, self.template.u_ranges)

            np.testing.assert_almost_equal(wk_fcs, db_fcs, decimal=DECIMALS)

    def test_energy_grad(self):
        for struct_path in glob.glob(os.path.join(self.path_to_workers, '*')):
            worker = pickle.load(open(struct_path, 'rb'))

            struct_name = os.path.splitext(os.path.split(struct_path)[-1])[0]

            db_grad = self.db.compute_energy_grad(
                struct_name, self.pvec, self.template.u_ranges
            )

            wk_grad = worker.energy_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

    def test_forces_grad(self):
        for struct_path in glob.glob(os.path.join(self.path_to_workers, '*')):
            worker = pickle.load(open(struct_path, 'rb'))

            struct_name = os.path.splitext(os.path.split(struct_path)[-1])[0]

            db_grad = self.db.compute_forces_grad(
                struct_name, self.pvec, self.template.u_ranges
            )

            wk_grad = worker.forces_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_almost_equal(wk_grad, db_grad, decimal=DECIMALS)

def build_template(version='full', inner_cutoff=1.5, outer_cutoff=5.5):

    potential_template = Template(
        pvec_len=108,
        u_ranges=[(-1, 1), (-1, 1)],
        # Ranges taken from Lou Ti-Mo (phis) or from old TiO (other)
        spline_ranges=[(-1, 1), (-1, 1), (-1, 1), (-10, 10), (-10, 10),
                       (-1, 1), (-1, 1), (-5, 5), (-5, 5),
                       (-10, 10), (-10, 10), (-10, 10)],
        spline_indices=[(0, 9), (9, 18), (18, 27), (27, 36), (36, 45),
                        (45, 54), (54, 63), (63, 72), (72, 81),
                        (81, 90), (90, 99), (99, 108)]
    )

    mask = np.ones(potential_template.pvec_len)

    if version == 'full':
        potential_template.pvec[6] = 0;
        mask[6] = 0  # lhs value phi_Ti
        potential_template.pvec[8] = 0;
        mask[8] = 0  # rhs deriv phi_Ti

        potential_template.pvec[15] = 0;
        mask[15] = 0  # rhs value phi_TiMo
        potential_template.pvec[17] = 0;
        mask[17] = 0  # rhs deriv phi_TiMo

        potential_template.pvec[24] = 0;
        mask[24] = 0  # rhs value phi_Mo
        potential_template.pvec[26] = 0;
        mask[26] = 0  # rhs deriv phi_Mo

        potential_template.pvec[33] = 0;
        mask[33] = 0  # rhs value rho_Ti
        potential_template.pvec[35] = 0;
        mask[35] = 0  # rhs deriv rho_Ti

        potential_template.pvec[42] = 0;
        mask[42] = 0  # rhs value rho_Mo
        potential_template.pvec[44] = 0;
        mask[44] = 0  # rhs deriv rho_Mo

        potential_template.pvec[69] = 0;
        mask[69] = 0  # rhs value f_Ti
        potential_template.pvec[71] = 0;
        mask[71] = 0  # rhs deriv f_Ti

        potential_template.pvec[78] = 0;
        mask[78] = 0  # rhs value f_Mo
        potential_template.pvec[80] = 0;
        mask[80] = 0  # rhs deriv f_Mo

    elif version == 'phi':
        mask[27:] = 0

        potential_template.pvec[6] = 0;
        mask[6] = 0  # lhs value phi_Ti
        potential_template.pvec[8] = 0;
        mask[8] = 0  # rhs deriv phi_Ti

        potential_template.pvec[15] = 0;
        mask[15] = 0  # rhs value phi_TiMo
        potential_template.pvec[17] = 0;
        mask[17] = 0  # rhs deriv phi_TiMo

        potential_template.pvec[24] = 0;
        mask[24] = 0  # rhs value phi_Mo
        potential_template.pvec[26] = 0;
        mask[26] = 0  # rhs deriv phi_Mo

        # accidental
        potential_template.pvec[34] = 1;
        mask[34] = -2. / 3  # rhs value f_Mo
        potential_template.pvec[87] = 1;
        mask[87] = -1  # rhs value f_Mo
        potential_template.pvec[96] = 1;
        mask[96] = -1  # rhs value f_Mo
        potential_template.pvec[105] = 1;
        mask[105] = -1  # rhs value f_Mo

    potential_template.active_mask = mask

    return potential_template

if __name__ == "__main__":
    unittest.main()
