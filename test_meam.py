import os
import unittest
import meam
import lammpsTools
from meam import MEAM

all_structs = False

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

class uc(unittest.TestCase):
    """Test cases for a small unit cell of ONLY Ti atoms"""

    def test_uc_meam(self):
        p = MEAM("./test-files/TiO.meam.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])

        expected = -9.66131009181
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_uc_nophi(self):
        p = MEAM("./test-files/TiO.nophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])

        expected = -6.49868915564
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_uc_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])

        expected = -3.16262093617
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_uc_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])

        expected = -9.80955197681
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

    def test_uc_rho(self):
        p = MEAM("./test-files/TiO.rho.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.uc.Ti", ['Ti'])

        expected = -6.64693104065
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 11)

class crowd_ti_only(unittest.TestCase):
    """Test cases for bulk Ti in hexagonal structure"""

    # TODO: add _nophi

    def test_crowd_phionly(self):
        p = MEAM("./test-files/TiO.phionly.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])

        expected = -153.6551
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 4)

    def test_crowd_rho(self):
        p = MEAM("./test-files/TiO.rho.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])

        expected = -316.97704
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 5)

    def test_crowd_rhophi(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])

        expected = -470.63215
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 5)

    def test_crowd_meam(self):
        p = MEAM("./test-files/TiO.rhophi.spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data.post_min_crowd.Ti",\
                ['Ti'])

        expected = -470.63215
        val = p.eval(atoms)

        self.assertAlmostEqual(val, expected, 5)

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

if all_structs:
    class all_structs(unittest.TestCase):
            """Test cases for all structures in the Zhang/Trinkle Ti-O database"""

            def test_crowd_hc10_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc10.Ti", ['Ti', 'O'])

                    expected = -464.602736288
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc1_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc1.Ti", ['Ti', 'O'])

                    expected = -470.055954431
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc2_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc2.Ti", ['Ti', 'O'])

                    expected = -470.006918479
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc3_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc3.Ti", ['Ti', 'O'])

                    expected = -469.804066269
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc4_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc4.Ti", ['Ti', 'O'])

                    expected = -469.627890751
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc5_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc5.Ti", ['Ti', 'O'])

                    expected = -469.44634577
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc6_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc6.Ti", ['Ti', 'O'])

                    expected = -469.445810124
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc7_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc7.Ti", ['Ti', 'O'])

                    expected = -467.9037012
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc8_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc8.Ti", ['Ti', 'O'])

                    expected = -465.865004263
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_hc9_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_hc9.Ti", ['Ti', 'O'])

                    expected = -465.004149191
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_oc1_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_oc1.Ti", ['Ti', 'O'])

                    expected = -470.080444578
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_oc2_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_oc2.Ti", ['Ti', 'O'])

                    expected = -469.966057831
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_oc3_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_oc3.Ti", ['Ti', 'O'])

                    expected = -469.718800189
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_oc4_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_oc4.Ti", ['Ti', 'O'])

                    expected = -469.493085728
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_oc5_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_oc5.Ti", ['Ti', 'O'])

                    expected = -469.058838354
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_relax_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd.relax.Ti", ['Ti', 'O'])

                    expected = -470.518447014
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd11_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd11.Ti", ['Ti', 'O'])

                    expected = -470.075461084
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd12_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd12.Ti", ['Ti', 'O'])

                    expected = -470.073554813
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd13_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd13.Ti", ['Ti', 'O'])

                    expected = -470.065753641
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd15_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd15.Ti", ['Ti', 'O'])

                    expected = -470.072833727
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd31_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd31.Ti", ['Ti', 'O'])

                    expected = -470.032252022
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd32_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd32.Ti", ['Ti', 'O'])

                    expected = -469.948489711
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd33_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd33.Ti", ['Ti', 'O'])

                    expected = -469.911419684
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd34_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd34.Ti", ['Ti', 'O'])

                    expected = -470.005913916
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_rnd35_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd_rnd35.Ti", ['Ti', 'O'])

                    expected = -469.953440496
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_crowd_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/crowd.Ti", ['Ti', 'O'])

                    expected = -470.108181472
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_face_c_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/face_c.Ti", ['Ti', 'O'])

                    expected = -467.056756839
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hc_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hc.Ti", ['Ti', 'O'])

                    expected = -470.065001169
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_1_7TiO_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_1.7TiO.Ti", ['Ti', 'O'])

                    expected = -469.348991405
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_1_8TiO_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_1.8TiO.Ti", ['Ti', 'O'])

                    expected = -470.166607215
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_1_9TiO_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_1.9TiO.Ti", ['Ti', 'O'])

                    expected = -470.619952418
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_2TiO_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_2TiO.Ti", ['Ti', 'O'])

                    expected = -470.737407541
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_minus1_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_minus1.Ti", ['Ti', 'O'])

                    expected = -470.969702686
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_minus2_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_minus2.Ti", ['Ti', 'O'])

                    expected = -470.927019146
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_plus1_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_plus1.Ti", ['Ti', 'O'])

                    expected = -470.975523809
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_plus2_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_plus2.Ti", ['Ti', 'O'])

                    expected = -470.941127583
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_relax_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex.relax.Ti", ['Ti', 'O'])

                    expected = -471.170628496
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd11_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd11.Ti", ['Ti', 'O'])

                    expected = -470.926590974
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd12_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd12.Ti", ['Ti', 'O'])

                    expected = -470.928174941
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd13_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd13.Ti", ['Ti', 'O'])

                    expected = -470.920791861
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd14_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd14.Ti", ['Ti', 'O'])

                    expected = -470.92438698
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd15_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd15.Ti", ['Ti', 'O'])

                    expected = -470.922637264
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd31_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd31.Ti", ['Ti', 'O'])

                    expected = -470.851516147
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd32_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd32.Ti", ['Ti', 'O'])

                    expected = -470.781029621
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd33_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd33.Ti", ['Ti', 'O'])

                    expected = -470.91291818
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd34_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd34.Ti", ['Ti', 'O'])

                    expected = -470.875303992
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_rnd35_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex_rnd35.Ti", ['Ti', 'O'])

                    expected = -470.845886662
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_hex_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/hex.Ti", ['Ti', 'O'])

                    expected = -470.984273603
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oc_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oc.Ti", ['Ti', 'O'])

                    expected = -470.04383262
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_relax_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct.relax.Ti", ['Ti', 'O'])

                    expected = -472.292756566
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd11_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd11.Ti", ['Ti', 'O'])

                    expected = -472.004500135
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd12_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd12.Ti", ['Ti', 'O'])

                    expected = -471.979396891
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd15_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd15.Ti", ['Ti', 'O'])

                    expected = -471.990693602
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd31_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd31.Ti", ['Ti', 'O'])

                    expected = -471.824367566
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd32_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd32.Ti", ['Ti', 'O'])

                    expected = -471.780799723
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd33_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd33.Ti", ['Ti', 'O'])

                    expected = -472.01120623
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd34_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd34.Ti", ['Ti', 'O'])

                    expected = -471.827652984
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_rnd35_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct_rnd35.Ti", ['Ti', 'O'])

                    expected = -471.659302614
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oct_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oct.Ti", ['Ti', 'O'])

                    expected = -472.132978022
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oh_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oh.Ti", ['Ti', 'O'])

                    expected = -469.859138679
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_oo_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/oo.Ti", ['Ti', 'O'])

                    expected = -469.149035955
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_stk40TiO0_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/stk40TiO0.Ti", ['Ti', 'O'])

                    expected = -189.138398412
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_stk40TiO_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/stk40TiO.Ti", ['Ti', 'O'])

                    expected = -171.684189366
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_stk80TiO0_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/stk80TiO0.Ti", ['Ti', 'O'])

                    expected = -389.515246359
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

            def test_stk80TiO_Ti(self):
                    p = MEAM("./test-files/TiO.meam.spline")
                    atoms = lammpsTools.atoms_from_file("./test-files/stk80TiO.Ti", ['Ti', 'O'])

                    expected = -333.392072506
                    val = p.eval(atoms)

                    self.assertAlmostEqual(val, expected, 6)

if "__name__" == "__main__":
    unittest.main()
