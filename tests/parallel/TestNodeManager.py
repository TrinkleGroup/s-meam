import sys
sys.path.append('/home/jvita/scripts/s-meam/project/')
sys.path.append('/home/jvita/scripts/s-meam/project/tests/')

import unittest
import numpy as np
from src.worker import Worker
from src.database import Database
from src.nodemanager import NodeManager
from src.potential_templates import Template

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
            bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo, extra

points_per_spline = 7
DECIMALS = 8

class NodeManagerTests(unittest.TestCase):
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
            'db_delete.hdf5', 'w', cls.template.pvec_len, cls.types,
            cls.x_pvec, cls.x_indices, [inner_cutoff, outer_cutoff]
        )

        ow = False

        cls.db.add_structure('aa', dimers['aa'], overwrite=ow)
        cls.db.add_structure('ab', dimers['ab'], overwrite=ow)
        cls.db.add_structure('bb', dimers['bb'], overwrite=ow)
        cls.db.add_structure('aaa', trimers['aaa'], overwrite=ow)
        cls.db.add_structure('aba', trimers['aba'], overwrite=ow)
        cls.db.add_structure('bbb', trimers['bbb'], overwrite=ow)

        cls.db.add_structure('8_atoms', extra['8_atoms'], overwrite=ow)

        cls.db.add_structure(
            'bulk_vac_ortho_type1', bulk_vac_ortho['bulk_vac_ortho_type1'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_vac_ortho_type2', bulk_vac_ortho['bulk_vac_ortho_type2'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_vac_ortho_mixed', bulk_vac_ortho['bulk_vac_ortho_mixed'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_periodic_ortho_type1',
            bulk_periodic_ortho['bulk_periodic_ortho_type1'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_periodic_ortho_type2',
            bulk_periodic_ortho['bulk_periodic_ortho_type2'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_periodic_ortho_mixed',
            bulk_periodic_ortho['bulk_periodic_ortho_mixed'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_vac_rhombo_type1', bulk_vac_rhombo['bulk_vac_rhombo_type1'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_vac_rhombo_type2', bulk_vac_rhombo['bulk_vac_rhombo_type2'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_vac_rhombo_mixed', bulk_vac_rhombo['bulk_vac_rhombo_mixed'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_periodic_rhombo_type1',
            bulk_periodic_rhombo['bulk_periodic_rhombo_type1'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_periodic_rhombo_type2',
            bulk_periodic_rhombo['bulk_periodic_rhombo_type2'],
            overwrite=ow
        )

        cls.db.add_structure(
            'bulk_periodic_rhombo_mixed',
            bulk_periodic_rhombo['bulk_periodic_rhombo_mixed'],
            overwrite=ow
        )

        cls.pvec = np.ones((5, cls.template.pvec_len))
        cls.pvec *= np.arange(1, cls.pvec.shape[0] + 1)[:, np.newaxis]

        cls.struct_list = list(cls.db.keys())

        cls.node_manager = NodeManager(0, cls.template)
        cls.node_manager.load_structures(list(cls.struct_list), cls.db)
        # cls.node_manager.initialize_shared_memory()

        cls.node_manager.start_pool(1)

        cls.db.close()  # close to make sure NodeManager is using local data

    def test_forces_8_atoms(self):
        struct_list = ['8_atoms']

        nd_forces = self.node_manager.compute(
            'forces', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                extra[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.compute_forces(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_compute_type_error(self):
        self.assertRaises(
            ValueError, self.node_manager.compute, 'NaN', None, None, None
        )

    def test_energy_single(self):
        nd_energies = self.node_manager.compute(
            'energy', ['aa'], self.pvec, self.template.u_ranges
        )

        worker = Worker(dimers['aa'], self.x_pvec, self.x_indices, self.types)

        wk_eng, wk_ni = worker.compute_energy(self.pvec, self.template.u_ranges)

        np.testing.assert_allclose(
            wk_eng, nd_energies['aa'][0], rtol=1e-8
        )

        np.testing.assert_allclose(
            wk_ni, nd_energies['aa'][1], rtol=1e-8
        )

    def test_energy_dimers(self):
        struct_list = ['aa', 'ab', 'bb']

        nd_energies = self.node_manager.compute(
            'energy', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                dimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_eng, wk_ni = worker.compute_energy(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_eng, nd_energies[key][0], rtol=1e-8
            )

            np.testing.assert_allclose(
                wk_ni, nd_energies[key][1], rtol=1e-8
            )

    def test_forces_dimers(self):
        struct_list = ['aa', 'ab', 'bb']

        nd_forces = self.node_manager.compute(
            'forces', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                dimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.compute_forces(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_grad_dimers(self):
        struct_list = ['aa', 'ab', 'bb']

        nd_forces = self.node_manager.compute(
            'energy_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                dimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.energy_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_forces_grad_dimers(self):
        struct_list = ['aa', 'ab', 'bb']

        nd_forces = self.node_manager.compute(
            'forces_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                dimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.forces_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_trimers(self):
        struct_list = ['aaa', 'aba', 'bbb']

        nd_energies = self.node_manager.compute(
            'energy', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                trimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_eng, wk_ni = worker.compute_energy(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_eng, nd_energies[key][0], rtol=1e-8
            )

            np.testing.assert_allclose(
                wk_ni, nd_energies[key][1], rtol=1e-8
            )

    def test_forces_trimers(self):
        struct_list = ['aaa', 'aba', 'bbb']

        nd_forces = self.node_manager.compute(
            'forces', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                trimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.compute_forces(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_grad_trimers(self):
        struct_list = ['aaa', 'aba', 'bbb']

        nd_forces = self.node_manager.compute(
            'energy_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                trimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.energy_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_forces_grad_trimers(self):
        struct_list = ['aaa', 'aba', 'bbb']

        nd_forces = self.node_manager.compute(
            'forces_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                trimers[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.forces_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_bvo(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_energies = self.node_manager.compute(
            'energy', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )


            wk_eng, wk_ni = worker.compute_energy(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_eng, nd_energies[key][0], rtol=1e-8
            )

            # np.testing.assert_allclose(
            #     wk_ni, nd_energies[key][1], rtol=1e-8
            # )

    def test_forces_bvo(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.compute_forces(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_grad_bvo(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'energy_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:

            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.energy_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_forces_grad_bvo(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.forces_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_bpo(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_energies = self.node_manager.compute(
            'energy', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:

            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_eng, wk_ni = worker.compute_energy(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_eng, nd_energies[key][0], rtol=1e-8
            )

            np.testing.assert_allclose(
                wk_ni, nd_energies[key][1], rtol=1e-8
            )

    def test_forces_bpo(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.compute_forces(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_grad_bpo(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'energy_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:

            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.energy_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_forces_grad_bpo(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.forces_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_bvr(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_energies = self.node_manager.compute(
            'energy', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_eng, wk_ni = worker.compute_energy(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_eng, nd_energies[key][0], rtol=1e-8
            )

            np.testing.assert_allclose(
                wk_ni, nd_energies[key][1], rtol=1e-8
            )

    def test_forces_bvr(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.compute_forces(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_grad_bvr(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'energy_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.energy_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_forces_grad_bvr(self):
        struct_list = [
            'bulk_vac_ortho_type1',
            'bulk_vac_ortho_type2',
            'bulk_vac_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_vac_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.forces_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_bpr(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_energies = self.node_manager.compute(
            'energy', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_eng, wk_ni = worker.compute_energy(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_eng, nd_energies[key][0], rtol=1e-8
            )

            np.testing.assert_allclose(
                wk_ni, nd_energies[key][1], rtol=1e-8
            )

    def test_forces_bpr(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.compute_forces(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_energy_grad_bpr(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'energy_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.energy_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

    def test_forces_grad_bpr(self):
        struct_list = [
            'bulk_periodic_ortho_type1',
            'bulk_periodic_ortho_type2',
            'bulk_periodic_ortho_mixed'
        ]

        nd_forces = self.node_manager.compute(
            'forces_grad', struct_list, self.pvec, self.template.u_ranges
        )

        for key in struct_list:
            worker = Worker(
                bulk_periodic_ortho[key], self.x_pvec, self.x_indices, self.types
            )

            wk_fcs = worker.forces_gradient_wrt_pvec(
                self.pvec, self.template.u_ranges
            )

            np.testing.assert_allclose(
                wk_fcs, nd_forces[key], rtol=1e-8
            )

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
                        (81, 90), (90, 99), (99, 108)],
        types = ['H', 'He']
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
