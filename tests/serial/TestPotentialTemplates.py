import unittest
import numpy as np
from src.potential_templates import Template


class TemplateTests(unittest.TestCase):

    def setUp(self):
        self.pvec_len = 137
        self.seed = np.arange(self.pvec_len)

        self.active_mask = np.ones(self.pvec_len)
        self.active_mask[:45] = 0
        self.active_mask[71:83] = 0

        self.spline_ranges = [(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3),
                              (-0.5, 1), (-0.5, 1), (-2, 3), (-2, 3), (-7, 2),
                              (-7, 2), (-7, 2)]
        self.spline_indices = [(0, 15), (15, 30), (30, 45), (45, 58), (58, 71),
                               (71, 77), (77, 83), (83, 95), (95, 107),
                               (107, 117), (117, 127), (127, 137)]
        self.boundary_conditions = [()]

    def test_blank_init(self):
        template = Template(
            self.pvec_len,
            u_ranges=[(-1, 1), (-1, 1)],
            active_mask=self.active_mask,
            spline_ranges=self.spline_ranges,
            spline_indices=self.spline_indices,
            seed=self.seed
        )

        print("active_mask_size:", len(np.where(template.active_mask)[0]))

        np.testing.assert_allclose(self.seed, template.pvec)
        np.testing.assert_allclose(
            np.concatenate([self.seed[45:71], self.seed[83:]]),
            template.get_active_params())

    def test_insert_pvec(self):
        template = Template(
            self.pvec_len,
            u_ranges=[(-1, 1), (-1, 1)],
            active_mask=self.active_mask,
            spline_ranges=self.spline_ranges,
            spline_indices=self.spline_indices,
            seed=self.seed
        )

        test_pvec = template.insert_active_splines(np.ones((1,80))*-1)

        should_be = np.arange(137)
        should_be[45:71] = -1
        should_be[83:] = -1

        np.testing.assert_allclose(np.atleast_2d(should_be), test_pvec)

    def test_random_doesnt_break(self):
        template = Template(
            self.pvec_len,
            u_ranges=[(-1, 1), (-1, 1)],
            active_mask=self.active_mask,
            spline_ranges=self.spline_ranges,
            spline_indices=self.spline_indices,
            seed=self.seed
        )

        template.generate_random_instance()
