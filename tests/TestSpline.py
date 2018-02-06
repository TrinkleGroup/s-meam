import unittest
import numpy as np
import os

from spline import Spline, ZeroSpline

DIGITS = 10


class ConstructorTests(unittest.TestCase):
    """Verifies appropriate constructor behavior"""

    # Note: most error handling goes through scipy.CubicSpline
    def test_missing_data(self):
        self.assertRaises(TypeError, Spline)

    def test_too_many_end_derivs(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        self.assertRaises(ValueError, Spline, x, y, (1, 2, 3))

    def test_uneven_xy(self):
        x = np.arange(10)
        y = np.arange(9)

        self.assertRaises(ValueError, Spline, x, y)

    def test_overlapping_knots(self):
        x = np.array([0, 1, 1])
        y = np.array([0, 2, 2])

        self.assertRaises(ValueError, Spline, x, y)

    def test_invalid_bc(self):
        x = np.arange(10)
        y = np.arange(10)

        self.assertRaises(ValueError, Spline, x, y, ('bad', 'bc'))

    def test_basic_data(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        s = Spline(x, y, end_derivs=(1., 1.))

        np.testing.assert_allclose(s.x, x)
        self.assertEquals(s.cutoff, (0., 9.))
        self.assertEquals(s(x[0], 1), 1.0)
        self.assertEquals(s(x[-1], 1), 1.0)
        self.assertAlmostEqual(s.h, 1.0, DIGITS)

    def test_natural_bc(self):
        d0 = 0.0077693494604539
        dN = 0.105197706160344
        y2_expected = np.array([0.00152870603877451, 0.00038933722543571,
                                0.00038124926922248, 0.0156639264890892])

        x = np.array([-55.1423316, -44.7409899033333, -34.3396481566667,
                      -23.9383064])
        y = np.array([-0.2974556800, -0.15449458722, 0.05098657168,
                      0.57342694704])

        s = Spline(x, y, bc_type=((1, d0), (1, dN)), end_derivs=(d0, dN))

        self.assertAlmostEqual(s(x[0], 1), d0, DIGITS)
        self.assertAlmostEqual(s(x[-1], 1), dN, DIGITS)

        for i in range(len(x)):
            self.assertAlmostEqual(s(x[i], 2), y2_expected[i], DIGITS)

    def test_one_set_derivative(self):
        d0 = 0.15500135578733
        dN = 0.
        y2_expected = np.array([0, -1.60311717015859, 1.19940299483249,
                                1.47909794595154, -2.49521499855605])

        x = np.array([1.9, 2.8, 3.7, 4.6, 5.5])
        y = np.array([0.533321679606674, 0.456402081843862, -0.324281383502201,
                      -0.474029826906675, 0])

        s = Spline(x, y, bc_type=('natural', (1, dN)), end_derivs=(
            d0, dN))

        self.assertAlmostEqual(s(x[0], 1), d0, DIGITS)
        self.assertAlmostEqual(s(x[-1], 1), dN, DIGITS)

        for i in range(len(x)):
            self.assertAlmostEqual(s(x[i], 2), y2_expected[i], DIGITS)

    def test_zero_spline(self):
        x = np.arange(10)

        s = ZeroSpline(x)

        self.assertEqual(s(x[0], 1), 0.)
        self.assertEqual(s(x[-1], 1), 0.)
        self.assertEqual(s.h, 1)
        self.assertEqual(s.cutoff, (0., 9.))

        for i in range(len(x)):
            self.assertAlmostEqual(s(x[i]), 0, DIGITS)
            self.assertAlmostEqual(s(x[i], 1), 0, DIGITS)
            self.assertAlmostEqual(s(x[i], 1), 0, DIGITS)


class EvaluationTests(unittest.TestCase):
    """Test cases for Spline class"""

    def test_equality(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        s1 = Spline(x, y)
        s2 = Spline(x, y)

        self.assertTrue(s1 == s2)

    def test_inequality(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        s1 = Spline(x, y)
        s2 = Spline(x + 2, y)

        self.assertTrue(s1 != s2)

    def test_eval_knot_single(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0, dN))

        self.assertAlmostEqual(s(x[4]), 4., DIGITS)

    def test_eval_knot_all(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0, dN))

        for i in range(len(x)):
            self.assertAlmostEqual(s(x[i]), y[i], DIGITS)

    def test_eval_extrap_single_LHS(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0, dN))

        np.testing.assert_allclose(s(-1), -1.)

    def test_eval_extrap_single_RHS(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0, dN))

        np.testing.assert_allclose(s(11), 11.)

    def test_eval_extrap_many(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0, dN))

        x = np.concatenate(([-1], y, [11]))
        y = np.concatenate(([-1], y, [11]))

        for i in range(len(x)):
            self.assertAlmostEqual(s(x[i]), y[i], DIGITS)


class MethodTests(unittest.TestCase):

    def test_in_range(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        s = Spline(x, y)

        self.assertTrue(s.in_range(4))
        self.assertFalse(s.in_range(-1))
        self.assertFalse(s.in_range(11))

    def test_plot(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        s = Spline(x, y)

        s.plot(saveName='test.png')
        os.remove('test.png')


if __name__ == "__main__":
    unittest.main()
