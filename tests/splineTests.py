import unittest
import numpy as np

from spline import Spline

class ConstructorTests(unittest.TestCase):
    """Verifies appropriate constructor behavior"""

    # Note: mosterror handling goes through scipy.CubicSpline
    def test_missing_data(self):
        self.assertRaises(ValueError, Spline)

    def test_too_many_derivs(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        self.assertRaises(ValueError, Spline, x, y, (1,2,3))

    def test_uneven_xy(self):
        x = np.arange(10)
        y = np.arange(9)

        self.assertRaises(ValueError, Spline, x, y)

    def test_overlapping_knots(self):
        x = np.array([0,1,1])
        y = np.array([0,2,2])

        self.assertRaises(ValueError, Spline, x, y)

    def test_basic_data(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        s = Spline(x, y)

        np.testing.assert_allclose(s.x, x)
        np.testing.assert_allclose(s(s.x), y)
        self.assertEquals(s.cutoff, (0., 9.))
        self.assertEquals(s.d0, 1.0)
        self.assertEquals(s.dN, 1.0)
        self.assertAlmostEqual(s.h, 1.0, 15)

    def test_natural_bc(self):
        d0_expected = 0.0077693494604539
        dN_expected = 0.105197706160344
        y2_expected = np.array([0.00152870603877451, 0.00038933722543571,
                       0.00038124926922248, 0.0156639264890892])

        x = np.array([-55.1423316, -44.7409899033333, -34.3396481566667,
                  -23.9383064])
        y = np.array([-0.2974556800, -0.15449458722, 0.05098657168,
                     0.57342694704])

        s = Spline(x, y, (d0_expected, dN_expected))

        self.assertAlmostEqual(s.d0, d0_expected)
        self.assertAlmostEqual(s.dN, dN_expected)
        np.testing.assert_allclose(s(s.x, 2), y2_expected)

    def test_one_set_derivative(self):
        d0_expected = 0.15500135578733
        dN_expected = 0.
        y2_expected = np.array([0, -1.60311717015859, 1.19940299483249,
                                1.47909794595154, -2.49521499855605])

        x = np.array([1.9, 2.8, 3.7, 4.6, 5.5])
        y = np.array([0.533321679606674, 0.456402081843862, -0.324281383502201,
                      -0.474029826906675, 0])

        s = Spline(x, y, end_derivs=(d0_expected, dN_expected))

        self.assertAlmostEqual(s.d0, d0_expected)
        self.assertAlmostEqual(s.dN, dN_expected)
        np.testing.assert_allclose(s(s.x, 2), y2_expected, atol=1e-15)

# rzm: test evaluations; internal, @ knots, EXTRAPOLATION
class EvaluationTests(unittest.TestCase):
    """Test cases for Spline class"""

    def test_eval_knot_single(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0,dN))

        self.assertAlmostEqual(s(x[4]), 4.)

    def test_eval_knot_all(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0,dN))

        np.testing.assert_allclose(s(x), y)

    def test_eval_extrap_single_LHS(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0,dN))

        np.testing.assert_allclose(s(-1), -1.)

    def test_eval_extrap_single_RHS(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0,dN))

        np.testing.assert_allclose(s(11), 11.)

    def test_eval_extrap_many(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        d0 = dN = 1.

        s = Spline(x, y, end_derivs=(d0,dN))

        x = np.concatenate(([-1], y, [11]))
        y = np.concatenate(([-1], y, [11]))

        np.testing.assert_allclose(s(x), y)

    def test_sin_fxn(self):
        xi = np.linspace(0,2*np.pi,10)
        yi = np.sin(xi)

        s = Spline(xi, yi)

        # Check knot values
        for i in range(len(xi)):
            self.assertAlmostEqual(s(xi[i]), yi[i])

        # Check easy points
        self.assertAlmostEqual(s(0), 0.0, places=3)
        self.assertAlmostEqual(s(np.pi/2.), 1.0, places=3)
        self.assertAlmostEqual(s(np.pi), 0.0, places=3)
        self.assertAlmostEqual(s(3*np.pi/2.), -1.0, places=3)
        self.assertAlmostEqual(s(2*np.pi), 0.0, places=3)

class MethodTests(unittest.TestCase):

    def test_in_range(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        s = Spline(x, y)

        self.assertTrue(s.in_range(4))
        self.assertFalse(s.in_range(-1))
        self.assertFalse(s.in_range(11))

if __name__ == "__main__":
    unittest.main()
