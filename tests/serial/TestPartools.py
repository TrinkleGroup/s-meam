import unittest
import numpy as np
import src.partools as partools
from src.database import Database
from src.potential_templates import Template

class PartoolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE_PATH = "/home/jvita/scripts/s-meam/"
        LOAD_PATH = BASE_PATH + "data/fitting_databases/leno-redo/"
        DB_PATH = LOAD_PATH + 'structures'
        DB_INFO_FILE_NAME = LOAD_PATH + 'rhophi/info'

        cls.database = Database(DB_PATH, DB_INFO_FILE_NAME)
        cls.pot_template = initialize_potential_template()

        # TODO: these values will change once the ref energy is subtracted off
        cls.true_val_sum = 1986832.9815369176
        cls.true_grad_sum = 5762089.574448535

    def test_build_fxns(self):
        fxn, grad = partools.build_evaluation_functions(
            self.database, self.pot_template
        )

    def test_eval_fxns_one_pot(self):
        fxn, grad = partools.build_evaluation_functions(
            self.database, self.pot_template
        )

        eng, fcs = fxn(np.ones(len(np.where(self.pot_template.active_mask)[0])))

        val = np.concatenate([eng, fcs], axis=1)

        np.testing.assert_almost_equal(np.sum(val), self.true_val_sum)

    def test_eval_fxns_many_pots(self):
        fxn, grad = partools.build_evaluation_functions(
            self.database, self.pot_template
        )

        eng, fcs = fxn(
            np.ones((10, len(np.where(self.pot_template.active_mask)[0])))
        )

        val = np.concatenate([eng, fcs], axis=1)

        np.testing.assert_allclose(
            np.sum(val, axis=1), np.ones(10)*self.true_val_sum
        )

    def test_eval_grad_one_pot(self):
        fxn, grad = partools.build_evaluation_functions(
            self.database, self.pot_template
        )

        e_grad, f_grad = grad(
            np.ones(len(np.where(self.pot_template.active_mask)[0]))
        )

        grad = np.dstack([e_grad, f_grad])

        np.testing.assert_almost_equal(
            np.sum(np.sum(grad, axis=2),axis=1), self.true_grad_sum
        )

    def test_eval_grad_many_pots(self):
        fxn, grad = partools.build_evaluation_functions(
            self.database, self.pot_template
        )

        e_grad, f_grad = grad(
            np.ones((10, len(np.where(self.pot_template.active_mask)[0])))
        )

        grad = np.dstack([e_grad, f_grad])

        np.testing.assert_allclose(
            np.sum(np.sum(grad, axis=2), axis=1), self.true_grad_sum
        )

    def test_proc_assignment_one_per(self):
        procs_to_use = len(self.database.structures)

        assignments = partools.compute_procs_per_subset(
            self.database.structures.values(),
            total_num_procs=procs_to_use
        )

        self.assertEqual(np.concatenate(assignments).shape[0], procs_to_use)

    def test_proc_assignment_many_per(self):
        procs_to_use = len(self.database.structures) * 3 + np.random.randint(30)

        assignments = partools.compute_procs_per_subset(
            self.database.structures.values(),
            total_num_procs=procs_to_use
        )

        self.assertEqual(np.concatenate(assignments).shape[0], procs_to_use)

    def test_procs_per_subset(self):

        # subsets, work = partools.group_database_subsets(self.database, 336)
        ranks_per_subset = partools.compute_procs_per_subset(
            self.database.structures.values(), 336
        )

        # print(ranks_per_subset)
        np.testing.assert_allclose(
            ranks_per_subset, np.ones(len(self.database.structures))
        )

    def test_procs_invalid_method(self):
        self.assertRaises(
            ValueError, partools.compute_procs_per_subset(
                self.database.structures.values(), 'bad_arg'
            )
        )

def initialize_potential_template():

    potential_template = Template(
        pvec_len=137,
        u_ranges = [(-1, 1), (-1, 1)],
        spline_ranges=[(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3),
                       (-0.5, 1), (-0.5, 1), (-2, 3), (-2, 3), (-7, 2),
                       (-7, 2), (-7, 2)],
        spline_indices=[(0, 15), (15, 30), (30, 45), (45, 58), (58, 71),
                         (71, 77), (77, 83), (83, 95), (95, 107),
                         (107, 117), (117, 127), (127, 137)]
    )

    mask = np.ones(potential_template.pvec.shape)

    potential_template.pvec[12] = 0; mask[12] = 0 # rhs phi_A knot
    potential_template.pvec[14] = 0; mask[14] = 0 # rhs phi_A deriv

    potential_template.pvec[27] = 0; mask[27] = 0 # rhs phi_B knot
    potential_template.pvec[29] = 0; mask[29] = 0 # rhs phi_B deriv

    potential_template.pvec[42] = 0; mask[42] = 0 # rhs phi_B knot
    potential_template.pvec[44] = 0; mask[44] = 0 # rhs phi_B deriv

    potential_template.pvec[55] = 0; mask[55] = 0 # rhs rho_A knot
    potential_template.pvec[57] = 0; mask[57] = 0 # rhs rho_A deriv

    potential_template.pvec[68] = 0; mask[68] = 0 # rhs rho_B knot
    potential_template.pvec[70] = 0; mask[70] = 0 # rhs rho_B deriv

    potential_template.pvec[92] = 0; mask[92] = 0 # rhs f_A knot
    potential_template.pvec[94] = 0; mask[94] = 0 # rhs f_A deriv

    potential_template.pvec[104] = 0; mask[104] = 0 # rhs f_B knot
    potential_template.pvec[106] = 0; mask[106] = 0 # rhs f_B deriv

    potential_template.pvec[83:] = 0; mask[83:] = 0 # EAM params only
    # potential_template.pvec[45:] = 0; mask[45:] = 0 # pair params only

    potential_template.active_mask = mask

    return potential_template
