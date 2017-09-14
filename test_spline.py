import unittest
import numpy as np
from spline import Spline

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

    def test_basic(self):
        xi = np.linspace(0,2*np.pi,10)
        yi = np.sin(xi)

        s = Spline(xi,yi)

        # Check knot values
        for i in xrange(len(xi)):
            self.assertAlmostEqual(s(xi[i]), yi[i])

        # Check easy points
        self.assertAlmostEqual(s(0), 0.0, places=3)
        self.assertAlmostEqual(s(np.pi/2.), 1.0, places=3)
        self.assertAlmostEqual(s(np.pi), 0.0, places=3)
        self.assertAlmostEqual(s(3*np.pi/2.), -1.0, places=3)
        self.assertAlmostEqual(s(2*np.pi), 0.0, places=3)

if __name__ == "__main__":
    unittest.main()
