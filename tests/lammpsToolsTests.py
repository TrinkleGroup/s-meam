import unittest
import os
import lammpsTools
import numpy as np

class ReadWriteTests(unittest.TestCase):

    def test_read_meam(self):
        fname = '../data/pot_files/TiO.meam.spline'

        knot_x, knot_y, all_y2, indices, all_d0, all_dN = \
            lammpsTools.read_spline_meam(fname)

        splines_x = np.split(knot_x, indices)
        lens = [len(splines_x[i]) for i in range(len(splines_x))]

        self.assertEqual(13, lens[0])
        self.assertEqual(5, lens[1])
        self.assertEqual(13, lens[2])
        self.assertEqual(11, lens[3])
        self.assertEqual(5, lens[4])
        self.assertEqual(4, lens[5])
        self.assertEqual(5, lens[6])
        self.assertEqual(10, lens[7])
        self.assertEqual(5, lens[8])
        self.assertEqual(8, lens[9])
        self.assertEqual(5, lens[10])
        self.assertEqual(8, lens[11])

    def test_read_write_meam(self):
        fname = '../data/pot_files/TiO.meam.spline'

        knots_x, knots_y, all_y2, indices, all_d0, all_dN = \
            lammpsTools.read_spline_meam(fname)

        splines_x = np.split(knots_x, indices)
        lens = [len(splines_x[i]) for i in range(len(splines_x))]

        lammpsTools.write_spline_meam('test.spline', knots_x, knots_y,
                                      all_d0, all_dN, indices, ['Ti', 'O'])

        knots_x_post, knots_y_post, all_y2_post, indices_post, all_d0_post,\
            all_dN_post = lammpsTools.read_spline_meam(fname)

        np.testing.assert_allclose(knots_x_post, knots_x)
        np.testing.assert_allclose(knots_y_post, knots_y)
        np.testing.assert_allclose(all_y2_post, all_y2)
        np.testing.assert_allclose(indices_post, indices)
        np.testing.assert_allclose(all_d0_post, all_d0)
        np.testing.assert_allclose(all_dN_post, all_dN)

        os.remove('test.spline')
