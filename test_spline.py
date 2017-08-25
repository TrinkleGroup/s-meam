import unittest
import numpy as np
from spline import Spline
from scipy.interpolate import CubicSpline

class SplineTests(unittest.TestCase):
    """Test cases for Spline class"""

    def test_setters(self):
        xi = np.arange(1,10)
        yi = np.sin(xi)

        s = Spline(xi,yi)
        
        s.d0 = 100
        self.assertEquals(s.d0, 100)

        s.dN = 7
        self.assertEquals(s.dN, 7)

    def test_constructor(self):
        xi = np.arange(1,10)
        yi = np.sin(xi)

        s = Spline(xi,yi)

suite = unittest.TestLoader().loadTestsFromTestCase(SplineTests)
unittest.TextTestRunner(verbosity=2).run(suite)
