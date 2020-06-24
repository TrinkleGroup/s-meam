import sys
sys.path.append('./')
import numpy as np
np.random.seed(42)

import pickle

# import src.lammpsTools
# from tests.testStructs import allstructs
# from tests.testPotentials import get_random_pots
from src.worker import Worker

pot = get_random_pots(1)['meams'][0]

pot.write_to_file('data/fitting_databases/seed_42/seed_42.meam')

x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)
print(x_pvec, indices)

for name in allstructs.keys():
#for name in ['4_atoms']:
    print(name)
    struct = allstructs[name]

    outfile = open('data/fitting_databases/seed_42/info.'+name, 'wb')
    w_outfile = open('data/fitting_databases/seed_42/evaluator.'+name, 'wb')

    src.lammpsTools.atoms_to_LAMMPS_file(
            'data/fitting_databases/seed_42/structs/data.'+name, struct)

    w = Worker(struct, x_pvec, indices, pot.types,)

    lmp_res = pot.get_lammps_results(struct)

    w_eng = w.compute_energy(y_pvec) / len(struct)
    w_fcs = w.compute_forces(y_pvec) / len(struct)

    np.testing.assert_almost_equal(
            np.abs(w_eng - lmp_res['energy']/len(struct)), 0.0, decimal=12)

    np.testing.assert_almost_equal(
            np.abs(w_fcs - lmp_res['forces']/len(struct)), 0.0, decimal=12)

    # np.savetxt(outfile, np.array([len(struct)]), fmt="%d")
    np.savetxt(outfile, np.array(w_eng), fmt="%.16f")
    np.savetxt(outfile, np.array(w_fcs[0]), fmt="%.16f")

    pickle.dump(w, w_outfile)

    outfile.close()
    w_outfile.close()
