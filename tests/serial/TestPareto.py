import unittest
import numpy as np
import src.pareto as pareto

class ParetoTests(unittest.TestCase):

    def setUp(self):
        self.dummy_objectives = np.array([
                [0.1, 0.2, 0.0, 0.3, 0.4],
                [0.1, 0.1, 0.0, 0.3, 0.4],  # 1 dominates 0
                [0.2, 0.1, 0.0, 0.3, 0.4],  # 2 does NOT dominate 0
                [0.1, 0.2, 0.0, 0.3, 0.4],  # 3 does NOT dominate 0
                [0.2, 0.2, 0.0, 0.3, 0.4],  # 0 dominates 4
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.4, 0.4, 0.4, 0.4],
            ])

        self.num_objectives = 5
        self.divs = 5
        self.grid_widths = np.array([0.096]*5)
        self.lower_bounds = np.array([-0.04]*5)

        self.grid_coords = np.array([
                [1, 2, 0, 3, 4],
                [1, 1, 0, 3, 4],
                [2, 1, 0, 3, 4],
                [1, 2, 0, 3, 4],
                [2, 2, 0, 3, 4],
                [0, 0, 0, 0, 0],
                [4, 4, 4, 4, 4],
            ])

        self.grid_ranks = np.array([10, 9, 10, 10, 11, 0, 20])

        self.grid_difference = np.array([
                [ 0,  1,  2,  0,  1,  10, 10],
                [ 1,  0,  1,  1,  2,   9, 11],
                [ 2,  1,  0,  2,  1,  10, 10],
                [ 0,  1,  2,  0,  1,  10, 10],
                [ 1,  2,  1,  1,  0,  11,  9],
                [10,  9, 10, 10, 11,   0, 20],
                [10, 11, 10, 10,  9,  20,  0],
            ])

        box_w = self.lower_bounds + self.grid_coords*self.grid_widths
        diff = self.dummy_objectives - box_w
        err = (diff/self.grid_widths)
        self.gcpd = np.sqrt(np.sum(err**2, axis=1))

        self.grid_crowding = np.array([16, 15, 14, 16, 15, 0, 0])

        self.fronts = [[5], [1], [0, 2, 3], [4], [6]]

    def test_grid_settings(self):
        widths, lowers = pareto.grid_settings(self.dummy_objectives, self.divs)

        np.testing.assert_equal(self.grid_widths, widths)
        np.testing.assert_equal(self.lower_bounds, lowers)

    def test_coordinates(self):
        coords = pareto.grid_coordinates(
                self.dummy_objectives, self.grid_widths, self.lower_bounds
            )

        np.testing.assert_equal(self.grid_coords, coords)

    def test_difference(self):
        diff = pareto.grid_difference(self.grid_coords)
        np.testing.assert_equal(self.grid_difference, diff)

    def test_ranks(self):
        ranks = pareto.grid_ranks(self.grid_coords)
        np.testing.assert_equal(self.grid_ranks, ranks)

    def test_crowding(self):
        crowding = pareto.grid_crowding_distance(
                self.grid_difference, self.num_objectives
            )

        np.testing.assert_equal(self.grid_crowding, crowding)

    def test_gcpd(self):
        gcpd = pareto.grid_gcpd(
                self.dummy_objectives, self.grid_coords, self.lower_bounds,
                self.grid_widths
            )

        np.testing.assert_allclose(self.gcpd, gcpd)

    def test_pareto_dominance(self):
        self.assertTrue(pareto.dominates(
                self.dummy_objectives[1], self.dummy_objectives[0]
            ))


        self.assertFalse(pareto.dominates(
                self.dummy_objectives[0], self.dummy_objectives[1]
            ))

        self.assertFalse(pareto.dominates(
                self.dummy_objectives[2], self.dummy_objectives[0]
            ))

        self.assertFalse(pareto.dominates(
                self.dummy_objectives[3], self.dummy_objectives[0]
            ))

        self.assertFalse(pareto.dominates(
                self.dummy_objectives[0], self.dummy_objectives[3]
            ))

        self.assertTrue(pareto.dominates(
                self.dummy_objectives[0], self.dummy_objectives[4]
            ))


        self.assertFalse(pareto.dominates(
                self.dummy_objectives[4], self.dummy_objectives[0]
            ))

    def test_grid_dominance(self):
        self.grid_coords = np.array([
                [1, 2, 0, 3, 4],
                [1, 1, 0, 3, 4],
                [2, 1, 0, 3, 4],
                [1, 2, 0, 3, 4],  # q
                [2, 2, 0, 3, 4],
                [0, 0, 0, 0, 0],
                [4, 4, 4, 4, 4],
            ])

        self.assertTrue(pareto.dominates(
                self.grid_coords[1], self.grid_coords[0]
            ))


        self.assertFalse(pareto.dominates(
                self.grid_coords[0], self.grid_coords[1]
            ))

        self.assertFalse(pareto.dominates(
                self.grid_coords[2], self.grid_coords[0]
            ))
        self.assertFalse(pareto.dominates(
                self.grid_coords[3], self.grid_coords[0]
            ))

        self.assertFalse(pareto.dominates(
                self.grid_coords[0], self.grid_coords[3]
            ))

        self.assertTrue(pareto.dominates(
                self.grid_coords[0], self.grid_coords[4]
            ))


        self.assertFalse(pareto.dominates(
                self.grid_coords[4], self.grid_coords[0]
            ))


    def test_tournament_doesnt_crash(self):
        pareto.tournament_selection(
                0, 1,
                self.dummy_objectives, self.grid_coords, self.grid_crowding
            )

    def test_rank_adjustments(self):

        expected = self.grid_ranks.copy()
        expected[0] += 7
        expected[1] += 4
        expected[2] += 4
        expected[4] += 5
        expected[6] += 5

        modified_ranks = pareto.gr_adjustment(
                3, self.grid_coords, self.grid_ranks, self.grid_difference,
                self.num_objectives
            )

        np.testing.assert_equal(expected, modified_ranks)

    def test_fronts(self):
        fronts = pareto.fast_non_dominated_sort(self.dummy_objectives)

        for f, t_f in zip(fronts, self.fronts):
            np.testing.assert_equal(f, t_f)

    def test_environment_selection(self):
        population = np.arange(35).reshape((7, 5))

        environment = pareto.environmental_selection(
            4, population, self.dummy_objectives, self.divs
        )

        expected = [
            [25, 26, 27, 28, 29],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ]

        np.testing.assert_equal(expected, environment)

    def test_breed_doesnt_crash(self):
        population = pareto.breed(
            np.arange(35).reshape((7, 5)), self.dummy_objectives,
            self.grid_coords, self.grid_crowding, 7
        )

        np.testing.assert_equal((7, 5), population.shape)

    def test_mutation_does_something(self):
        population = np.arange(35).reshape((7, 5)).astype(np.float)

        pareto.mutate(population, 0, 1, 0.5)
