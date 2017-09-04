import unittest
import meam
import lammpsTools
from meam import MEAM

class MEAMTests(unittest.TestCase):
    """Test cases for MEAM class"""

    def test_spline_derivatives(self):
        p = MEAM("TiO.meam.spline")

        self.assertAlmostEqual(p.phis[0].d0, -20, 1)
        self.assertAlmostEqual(p.phis[0].dN, 0.0, 1)

        self.assertAlmostEqual(p.rhos[0].d0, -1.0, 1)
        self.assertAlmostEqual(p.rhos[0].dN, 0.0, 1)

        self.assertAlmostEqual(p.fs[0].d0, 2.7733, 1)
        self.assertAlmostEqual(p.fs[0].dN, 0.0, 1)

        self.assertAlmostEqual(p.us[0].d0, 0.0078, 1)
        self.assertAlmostEqual(p.us[0].dN, 0.1052, 1)

        self.assertAlmostEqual(p.gs[0].d0, 8.3364, 1)
        self.assertAlmostEqual(p.gs[0].dN, -60.4025, 1)

    def test_type_setter(self):
        p = MEAM()
        self.assertEquals(0, p.ntypes)

        with self.assertRaises(AttributeError):
            p.ntypes = 2

        p.types = ['Ti','O']
        self.assertEquals(2, p.ntypes)

    def test_constructor(self):
        p = MEAM()
        self.assertEquals(0.0, p.cutoff)
        self.assertEquals(0, p.ntypes)
        self.assertEquals(0, len(p.phis))
        self.assertEquals(0, len(p.rhos))
        self.assertEquals(0, len(p.us))
        self.assertEquals(0, len(p.fs))
        self.assertEquals(0, len(p.gs))

        p = MEAM(types='H')
        self.assertEquals(0.0, p.cutoff)
        self.assertEquals(1, len(p.types))
        self.assertEquals(1, len(p.phis))
        self.assertEquals(1, len(p.rhos))
        self.assertEquals(1, len(p.us))
        self.assertEquals(1, len(p.fs))
        self.assertEquals(1, len(p.gs))

        p = MEAM(types=['Ti', 'O'])
        self.assertEquals(0.0, p.cutoff)
        self.assertEquals(2, len(p.types))
        self.assertEquals(3, len(p.phis))
        self.assertEquals(2, len(p.rhos))
        self.assertEquals(2, len(p.us))
        self.assertEquals(2, len(p.fs))
        self.assertEquals(3, len(p.gs))

        p = MEAM("TiO.meam.spline")
        self.assertGreater(p.cutoff, 0.0)   # a > b
        self.assertEquals(2, len(p.types))
        self.assertEquals(3, len(p.phis))
        self.assertEquals(2, len(p.rhos))
        self.assertEquals(2, len(p.us))
        self.assertEquals(2, len(p.fs))
        self.assertEquals(3, len(p.gs))

    def test_read_file(self):
        with self.assertRaises(IOError):
            p = MEAM()
            rets = p.read_from_file("does_not_exist.txt")

    def test_crowd_phionly(self):
        p = MEAM("TiO.phionly.spline")

        atoms = lammpsTools.atoms_from_lammps_data("data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -153.6551, delta=1.0)

    def test_crowd_meam(self):
        p = MEAM("TiO.meam.spline")

        atoms = lammpsTools.atoms_from_lammps_data("data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -463.71469, delta=1.0)

    def test_crowd_rho(self):
        p = MEAM("TiO.rho.spline")

        atoms = lammpsTools.atoms_from_lammps_data("data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -316.97704, delta=1.0)

    def test_crowd_rhophi(self):
        p = MEAM("TiO.rhophi.spline")

        atoms = lammpsTools.atoms_from_lammps_data("data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -470.63215, delta=1.0)


suite = unittest.TestLoader().loadTestsFromTestCase(MEAMTests)
unittest.TextTestRunner(verbosity=2).run(suite)
