import unittest
from potential import Potential

class PotentialTests(unittest.TestCase):
    """Test cases for Potential class"""

    def test_constructor(self):
        p = Potential()
        self.assertEquals(0.0, p.cutoff)

    def test_cutoff(self):
        p = Potential()

        p.cutoff = 1.0
        self.assertEquals(1.0, p.cutoff)

    def test_cutoff_error(self):
        p = Potential()

        with self.assertRaises(ValueError):
            p.cutoff = -1.0

    def test_unimplemented_compute_energies(self):
        p = Potential()

        with self.assertRaises(NotImplementedError):
            p.compute_energies()

if __name__ == "__main__":
    unittest.main()
