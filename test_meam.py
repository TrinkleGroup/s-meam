import os
import unittest
import meam
import lammpsTools
from meam import MEAM

class MEAMTests(unittest.TestCase):
    """Test cases for MEAM class"""

    def test_uc_meam(self):
        p = MEAM("TiO.meam.spline")
        atoms = lammpsTools.atoms_from_lammps_data("data.uc.Ti", ['Ti'])

        expected = -9.6613101
        val = p.eval(atoms)
        diff = (val-expected)/expected

        self.assertAlmostEqual(val, expected, delta=0.1*expected)

        print(("(py) %f (lmps) %f || diff = %f %%" % (val,e0,diff)))

    def test_uc_nophi(self):
        p = MEAM("TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_lammps_data("data.uc.Ti", ['Ti'])

        expected = -6,361927
        val = p.eval(atoms)
        diff = (val-expected)/expected

        self.assertAlmostEqual(val, expected, delta=0.1*expected)

        print(("(py) %f (lmps) %f || diff = %f %%" % (val,e0,diff)))

    def test_uc_phionly(self):
        p = MEAM("TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_lammps_data("data.uc.Ti", ['Ti'])

        expected = -3.1626209
        val = p.eval(atoms)
        diff = (val-expected)/expected

        self.assertAlmostEqual(val, expected, delta=0.1*expected)

        print(("(py) %f (lmps) %f || diff = %f %%" % (val,e0,diff)))

    def test_uc_rhophi(self):
        p = MEAM("TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_lammps_data("data.uc.Ti", ['Ti'])

        expected = -9.809552
        val = p.eval(atoms)
        diff = (val-expected)/expected

        self.assertAlmostEqual(val, expected, delta=0.1*expected)

        print(("(py) %f (lmps) %f || diff = %f %%" % (val,e0,diff)))

    def test_uc_rho(self):
        p = MEAM("TiO.rho.spline")
        atoms = lammpsTools.atoms_from_lammps_data("data.uc.Ti", ['Ti'])

        expected = -6.646931
        val = p.eval(atoms)
        diff = (val-expected)/expected

        self.assertAlmostEqual(val, expected, delta=0.1*expected)

        print(("(py) %f (lmps) %f || diff = %f %%" % (val,e0,diff)))


#    def test_all_structs(self):
#        p = MEAM("TiO.meam.spline")
#        
#        with open('lammps_results.dat', 'r') as f:
#            line = f.readline()
#
#            while line:
#                atoms = lammpsTools.atoms_from_lammps_data(line.strip(), ['Ti','O'])
#                e0 = float(f.readline().split()[0])
#
#                delta = abs(0.1*e0)
#
#                val = p.eval(atoms)
#                diff = (val-e0)/e0
#                self.assertAlmostEqual(e0, val, delta=delta)
#                line = f.readline()
#
#                print(line.strip()),
#                print(" ... ok: (py) %f (lmps) %f || diff = %f %%" % (val,e0,diff))


    def test_spline_derivatives(self):
        p = MEAM("TiO.meam.spline")

        # Assert endpoint values and derivatives, then perform extrapolation
        phi = p.phis[0]
        self.assertAlmostEqual(phi.d0, -20, 1)
        self.assertAlmostEqual(phi.dN, 0.0, 1)
        self.assertAlmostEqual(phi(phi.x[0]), 3.7443, 1)
        self.assertAlmostEqual(phi(phi.x[-1]), 0.0, 1)
        self.assertAlmostEqual(phi(phi.x[0]-1), 3.7443+20, 1)
        self.assertAlmostEqual(phi(phi.x[-1]+1), 0.0, 1)

        rho = p.rhos[0]
        self.assertAlmostEqual(rho.d0, -1.0, 1)
        self.assertAlmostEqual(rho.dN, 0.0, 1)
        self.assertAlmostEqual(rho(rho.x[0]), 1.7475, 1)
        self.assertAlmostEqual(rho(rho.x[-1]), 0.0, 1)
        self.assertAlmostEqual(rho(rho.x[0]-1), 1.7475+1, 1)
        self.assertAlmostEqual(rho(rho.x[-1]+1), 0.0, 1)

        f = p.fs[0]
        self.assertAlmostEqual(f.d0, 2.7733, 1)
        self.assertAlmostEqual(f.dN, 0.0, 1)
        self.assertAlmostEqual(f(f.x[0]), -0.1485, 1)
        self.assertAlmostEqual(f(f.x[-1]), 0.0, 1)
        self.assertAlmostEqual(f(f.x[0]-1), -0.1485-2.7733, 1)
        self.assertAlmostEqual(f(f.x[-1]+1), 0.0, 1)

        u = p.us[0]
        self.assertAlmostEqual(u.d0, 0.0078, 1)
        self.assertAlmostEqual(u.dN, 0.1052, 1)
        self.assertAlmostEqual(u(u.x[0]), -0.29746, 1)
        self.assertAlmostEqual(u(u.x[-1]), 0.57343, 1)
        self.assertAlmostEqual(u(u.x[0]-1), -0.29746-0.0078, 1)
        self.assertAlmostEqual(u(u.x[-1]+1), 0.57343+0.1052, 1)

        g = p.gs[0]
        self.assertAlmostEqual(g.d0, 8.3364, 1)
        self.assertAlmostEqual(g.dN, -60.4025, 1)
        self.assertAlmostEqual(g(g.x[0]), 0.0765, 1)
        self.assertAlmostEqual(g(g.x[-1]), -6.0091, 1)
        self.assertAlmostEqual(g(g.x[0]-1), 0.0765-8.3364, 1)
        self.assertAlmostEqual(g(g.x[-1]+1), -6.0091-60.4025, 1)

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
#suite = unittest.TestSuite()
#suite.addTests(MEAMTests.test_read_file)
unittest.TextTestRunner(verbosity=2).run(suite)
#if "__name__" == "__main__":
#    unittest.main()
