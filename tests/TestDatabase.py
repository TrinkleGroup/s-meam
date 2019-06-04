import sys
sys.path.append('/home/jvita/scripts/s-meam/project/')

import unittest
import numpy as np

from src.potential_templates import Template
from src.database import Database

from tests.testStructs import dimers, trimers, bulk_vac_ortho, \
            bulk_periodic_ortho, bulk_vac_rhombo, bulk_periodic_rhombo, extra

points_per_spline = 7

class DatabaseTests(unittest.TestCase):
    def setUp(self):
        inner_cutoff = 2.1
        outer_cutoff = 5.5

        x_pvec = np.concatenate([
            np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 5),
            np.tile(np.linspace(-1, 1, points_per_spline), 2),
            np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 2),
            np.tile(np.linspace(-1, 1, points_per_spline), 3)]
        )

        x_indices = list(range(0, points_per_spline * 12, points_per_spline))
        types = ["Ti", "Mo"]

        template  = build_template()

        self.db = Database(
            'db_delete.hdf5', template.pvec_len, types, x_pvec, x_indices,
            [inner_cutoff, outer_cutoff]
        )

        self.db.add_structure('aa', dimers['aa'])

    def test_matches_worker_dimer(self):
        pass

def build_template(version='full', inner_cutoff=1.5, outer_cutoff=5.5):
    x_pvec = np.concatenate([
        np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 5),
        np.tile(np.linspace(-1, 1, points_per_spline), 2),
        np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 2),
        np.tile(np.linspace(-1, 1, points_per_spline), 3)]
    )

    x_indices = range(0, points_per_spline * 12, points_per_spline)
    types = ["Ti", "Mo"]

    potential_template = Template(
        pvec_len=108,
        u_ranges=[(-1, 1), (-1, 1)],
        # Ranges taken from Lou Ti-Mo (phis) or from old TiO (other)
        spline_ranges=[(-1, 1), (-1, 1), (-1, 1), (-10, 10), (-10, 10),
                       (-1, 1), (-1, 1), (-5, 5), (-5, 5),
                       (-10, 10), (-10, 10), (-10, 10)],
        spline_indices=[(0, 9), (9, 18), (18, 27), (27, 36), (36, 45),
                        (45, 54), (54, 63), (63, 72), (72, 81),
                        (81, 90), (90, 99), (99, 108)]
    )

    mask = np.ones(potential_template.pvec_len)

    if version == 'full':
        potential_template.pvec[6] = 0;
        mask[6] = 0  # lhs value phi_Ti
        potential_template.pvec[8] = 0;
        mask[8] = 0  # rhs deriv phi_Ti

        potential_template.pvec[15] = 0;
        mask[15] = 0  # rhs value phi_TiMo
        potential_template.pvec[17] = 0;
        mask[17] = 0  # rhs deriv phi_TiMo

        potential_template.pvec[24] = 0;
        mask[24] = 0  # rhs value phi_Mo
        potential_template.pvec[26] = 0;
        mask[26] = 0  # rhs deriv phi_Mo

        potential_template.pvec[33] = 0;
        mask[33] = 0  # rhs value rho_Ti
        potential_template.pvec[35] = 0;
        mask[35] = 0  # rhs deriv rho_Ti

        potential_template.pvec[42] = 0;
        mask[42] = 0  # rhs value rho_Mo
        potential_template.pvec[44] = 0;
        mask[44] = 0  # rhs deriv rho_Mo

        potential_template.pvec[69] = 0;
        mask[69] = 0  # rhs value f_Ti
        potential_template.pvec[71] = 0;
        mask[71] = 0  # rhs deriv f_Ti

        potential_template.pvec[78] = 0;
        mask[78] = 0  # rhs value f_Mo
        potential_template.pvec[80] = 0;
        mask[80] = 0  # rhs deriv f_Mo

    elif version == 'phi':
        mask[27:] = 0

        potential_template.pvec[6] = 0;
        mask[6] = 0  # lhs value phi_Ti
        potential_template.pvec[8] = 0;
        mask[8] = 0  # rhs deriv phi_Ti

        potential_template.pvec[15] = 0;
        mask[15] = 0  # rhs value phi_TiMo
        potential_template.pvec[17] = 0;
        mask[17] = 0  # rhs deriv phi_TiMo

        potential_template.pvec[24] = 0;
        mask[24] = 0  # rhs value phi_Mo
        potential_template.pvec[26] = 0;
        mask[26] = 0  # rhs deriv phi_Mo

        # accidental
        potential_template.pvec[34] = 1;
        mask[34] = -2. / 3  # rhs value f_Mo
        potential_template.pvec[87] = 1;
        mask[87] = -1  # rhs value f_Mo
        potential_template.pvec[96] = 1;
        mask[96] = -1  # rhs value f_Mo
        potential_template.pvec[105] = 1;
        mask[105] = -1  # rhs value f_Mo

    potential_template.active_mask = mask

    return potential_template

if __name__ == "__main__":
    unittest.main()
