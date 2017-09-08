import os
import unittest
import meam
import lammpsTools
from meam import MEAM

class bulk_vac_tio(unittest.TestCase):
    """Test cases for a 2-Ti 1-O trimer in a vacuum"""

    def test_bulk_vac_tio_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.bulk_vac.TiO", ['Ti','O'])

        expected = -401.650326979
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

    def test_bulk_vac_tio_nophi(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.bulk_vac.TiO", ['Ti','O'])

        expected = -28.6621507852
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 10)

    def test_bulk_vac_tio_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.bulk_vac.TiO", ['Ti','O'])

        expected = -372.988176194
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

    def test_bulk_vac_tio_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.bulk_vac.TiO", ['Ti','O'])

        expected = -452.364422389
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

class trimer_2ti_1o(unittest.TestCase):
    """Test cases for a 2-Ti 1-O trimer in a vacuum"""

    def test_trimer_2ti_1o_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer2Ti1O.TiO", ['Ti','O'])

        expected = -1.58813051581
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_trimer_2ti_1o_meam(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer2Ti1O.TiO", ['Ti','O'])

        expected = -0.43729940725
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_trimer_2ti_1o_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer2Ti1O.TiO", ['Ti','O'])

        expected = -1.15083110856
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_trimer_2ti_1o_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer2Ti1O.TiO", ['Ti','O'])

        expected = -1.55680370894
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

class trimer_1ti_2o(unittest.TestCase):
    """Test cases for a 1-Ti 2-O trimer in a vacuum"""

    def test_trimer_1ti_2o_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer1Ti2O.TiO", ['Ti','O'])

        expected = -1.43566537282
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_trimer_1ti_2o_meam(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer1Ti2O.TiO", ['Ti','O'])

        expected = -0.351830715448
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 12)

    def test_trimer_1ti_2o_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer1Ti2O.TiO", ['Ti','O'])

        expected = -1.08383465737
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_trimer_1ti_2o_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file(\
                "./test-files/data.trimer1Ti2O.TiO", ['Ti','O'])

        expected = -1.43756083666
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

class dimer_tio(unittest.TestCase):
    """Test cases for a single Ti-O dimer in a vacuum"""

    def test_dimer_tio_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.TiO",\
                ['Ti','O'])

        expected = -0.731698702164
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 12)

    def test_dimer_tio_nophi(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.TiO",\
                ['Ti','O'])

        expected = -0.18978073834
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_dimer_tio_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.TiO",\
                ['Ti','O'])

        expected = -0.541917963824
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 12)

    def test_dimer_tio_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.TiO",\
                ['Ti','O'])

        expected = -0.731698702164
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 12)

class bulk_vac_ti(unittest.TestCase):
    """Test cases for bulk material in a vacuum"""

    def test_bulk_vac_ti_nophi(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.bulk_vac.Ti", ['Ti'])

        expected = -19.522211
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 6)

    def test_bulk_vac_ti_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.bulk_vac.Ti", ['Ti'])

        expected = -440.02272
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 5)

    def test_bulk_vac_ti_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.bulk_vac.Ti", ['Ti'])

        expected = -369.72476
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 5)

    def test_bulk_vac_ti_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.bulk_vac.Ti", ['Ti'])

        expected = -389.24697
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 5)

class ti_only_trimer(unittest.TestCase):
    """Test cases for a single trimer of ONLY Ti in a vacuum"""

    def test_trimer_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.trimer.Ti", ['Ti'])

        expected = -0.28895358
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 8)

    def test_trimer_nophi(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.trimer.Ti", ['Ti'])

        expected = -0.087967085
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

    def test_trimer_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.trimer.Ti", ['Ti'])

        expected = -0.20098649
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 8)

    def test_trimer_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.trimer.Ti", ['Ti'])

        expected = -0.28013609
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 8)

    def test_trimer_rho(self):
        p = MEAM("./test-files/TiO.rho.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.trimer.Ti", ['Ti'])

        expected = -0.0791495949193
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 13)

class dimer_ti_only(unittest.TestCase):
    """Test cases for a single trimer of ONLY Ti in a vacuum"""

    def test_dimer_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.Ti", ['Ti'])

        expected = -0.093377425
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

    def test_dimer_nophi(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.Ti", ['Ti'])

        expected = -0.026382567
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

    def test_dimer_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.Ti", ['Ti'])

        expected = -0.066994858
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

    def test_dimer_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.Ti", ['Ti'])

        expected = -0.093377425
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

    def test_dimer_rho(self):
        p = MEAM("./test-files/TiO.rho.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.dimer.Ti", ['Ti'])

        expected = -0.026382567
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 9)

#class uc_ti_only(unittest.TestCase):
#    """Test cases for a small unit cell of ONLY Ti atoms"""
#
#    def test_uc_meam(self):
#        p = MEAM("./test-files/TiO.meam.spline")
#        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])
#
#        expected = -9.6613101
#        val = p.eval(atoms)
#        diff = (val-expected)/expected
#
#        self.assertAlmostEqual(val, expected, 7)
#
#    def test_uc_nophi(self):
#        p = MEAM("./test-files/TiO.nophi.spline")
#        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])
#
#        expected = -6,361927
#        val = p.eval(atoms)
#        diff = (val-expected)/expected
#
#        self.assertAlmostEqual(val, expected, 6)
#
#    def test_uc_phionly(self):
#        p = MEAM("./test-files/TiO.phionly.spline")
#        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])
#
#        expected = -3.1626209
#        val = p.eval(atoms)
#        diff = (val-expected)/expected
#
#        self.assertAlmostEqual(val, expected, 7)
#
#    def test_uc_rhophi(self):
#        p = MEAM("./test-files/TiO.rhophi.spline")
#        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])
#
#        expected = -9.809552
#        val = p.eval(atoms)
#        diff = (val-expected)/expected
#
#        self.assertAlmostEqual(val, expected, 6)
#
#    def test_uc_rho(self):
#        p = MEAM("./test-files/TiO.rho.spline")
#        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])
#
#        expected = -6.646931
#        val = p.eval(atoms)
#        diff = (val-expected)/expected
#
#        self.assertAlmostEqual(val, expected, 6)

class crowd_ti_only(unittest.TestCase):
    """Test cases for bulk Ti in hexagonal structure"""

    # TODO: add _nophi

    def test_crowd_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")

        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -153.6551, 4)

    def test_crowd_rho(self):
        p = MEAM("./test-files/TiO.rho.spline")

        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -316.97704, 5)

    def test_crowd_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")

        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -470.63215, 5)

    def test_crowd_meam(self):
        p = MEAM("./test-files/TiO.rhophi.spline")

        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])
        self.assertAlmostEqual(p.eval(atoms), -470.63215, 5)

#class all_structs(unittest.TestCase):
#    """Test cases for all structures in the Zhang/Trinkle Ti-O database"""
#
#    def test_all_structs(self):
#        p = MEAM("./test-files/TiO.meam.spline")
#        
#        print("")
#        with open('lammps_results.dat', 'r') as f:
#            line = f.readline()
#
#            while line:
#                print(line.strip()),
#                atoms = lammpsTools.atoms_from_file(line.strip(), ['Ti','O'])
#                e0 = float(f.readline().split()[0])
#
#                delta = abs(0.1*e0)
#
#                val = p.eval(atoms)
#                diff = (val-e0)/e0
#                self.assertAlmostEqual(e0, val, 6)
#
#                print(" ... ok: (py) %f (lmps) %f || diff = %f %%" % (val,e0,diff))
#
#                line = f.readline()

class MEAMTests(unittest.TestCase):
    """Test cases for general MEAM class functions"""

    def test_spline_derivatives(self):
        p = MEAM("./test-files/TiO.meam.spline")

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

        p = MEAM("./test-files/TiO.meam.spline")
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

#suite = unittest.TestLoader().loadTestsFromTestCase(bulk_vac)
#suite = unittest.TestLoader().loadTestsFromTestCase(ti_only_trimer)
#suite = unittest.TestLoader().loadTestsFromTestCase(dimer_ti_only)
#suite = unittest.TestLoader().loadTestsFromTestCase(uc_ti_only)
#suite = unittest.TestLoader().loadTestsFromTestCase(crowd_ti_only)
#suite = unittest.TestLoader().loadTestsFromTestCase(all_structs)
#suite = unittest.TestLoader().loadTestsFromTestCase(MEAMTests)
#suite = unittest.TestSuite()
#suite.addTests(MEAMTests.test_read_file)
#unittest.TextTestRunner(verbosity=2).run(suite)
#if "__name__" == "__main__":
#    unittest.main()
