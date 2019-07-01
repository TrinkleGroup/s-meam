import unittest
import numpy as np

# import src.optimization

DIGITS_ACC = 12

class ForceMatching(unittest.TestCase):
    pass

    # def setUp(self):
    #     self.comp_f = [np.arange(12).reshape((4,3)) for _ in range(3)]
    #     self.true_f = [force_mat - i for i,force_mat in enumerate(self.comp_f)]
    # 
    #     self.comp_a = np.arange(10).tolist()
    #     self.true_a = (np.arange(10) - 1).tolist()
    # 
    #     self.weights = np.ones(10).tolist()
    # 
    # def test_same(self):
    #     Z = src.optimization.force_matching(self.comp_f[1:2], self.comp_f[1:2])
    #     np.testing.assert_equal(Z, 0)
    # 
    # def test_different(self):
    #     Z = src.optimization.force_matching(self.comp_f[1:2], self.true_f[1:2])
    #     np.testing.assert_equal(Z, 1)
    # 
    # def test_multi_same(self):
    #     Z = src.optimization.force_matching(self.comp_f, self.comp_f)
    # 
    #     np.testing.assert_equal(Z, 0)
    # 
    # def test_multi_different(self):
    #     Z = src.optimization.force_matching(self.comp_f, self.true_f)
    # 
    #     np.testing.assert_equal(Z, 5/3.)
    # 
    # def test_multi_different_sized_forces(self):
    #     new_comp_f = list(self.comp_f) + [np.arange(15).reshape((5,3))]
    #     new_true_f = list(self.true_f) + [np.arange(15).reshape((5,3)) - 1]
    #     # new_comp_f = self.comp_f[:1] + [np.arange(15).reshape((5,3))]
    #     # new_true_f = self.true_f[:1] + [np.arange(15).reshape((5,3)) - 1]
    # 
    #     Z = src.optimization.force_matching(new_comp_f, new_true_f,)
    # 
    #     np.testing.assert_almost_equal(Z, 1.4705882352941178, decimal=12)
    # 
    # def test_same_with_same_constraints(self):
    #     Z = src.optimization.force_matching(self.comp_f, self.comp_f,
    #         self.comp_a, self.comp_a, self.weights)
    # 
    #     np.testing.assert_equal(Z, 0)
    # 
    # def test_same_with_different_constraints(self):
    #     Z = src.optimization.force_matching(self.comp_f, self.comp_f,
    #         self.comp_a, self.true_a, self.weights)
    # 
    #     np.testing.assert_equal(Z, 10)
    # 
    # def test_different_with_different_constraints(self):
    #     Z = src.optimization.force_matching(self.comp_f, self.true_f,
    #         self.comp_a, self.true_a, self.weights)
    # 
    #     np.testing.assert_almost_equal(Z, 5/3. + 10, decimal=12)
    # 
    # def test_no_true_constraints(self):
    #     self.assertRaises(ValueError, src.optimization.force_matching,
    #         self.comp_f, self.true_f, self.comp_a)
    # 
    # def test_no_weights(self):
    #     self.assertRaises(ValueError, src.optimization.force_matching,
    #         self.comp_f, self.true_f, self.comp_a, self.true_a)
