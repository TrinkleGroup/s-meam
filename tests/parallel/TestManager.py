import sys
sys.path.append('./')

import unittest
import numpy as np
from mpi4py import MPI
# from mpinoseutils import *
from src.manager import Manager
from src.potential_templates import Template

# @mpitest(4)
class ManagerTests(unittest.TestCase):
    def test_main(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            print("Loading manager ...", flush=True)

        pot_template = initialize_potential_template()

        manager = Manager(1, comm, pot_template)
        manager.struct_name = 'hcp_2.95_4.68_ab0'
        manager.struct = manager.load_structure(
            manager.struct_name,
            '/home/jvita/scripts/s-meam/data/fitting_databases/leno-redo/structures/'
        )

        if rank == 0:
            print("Evaluating energy ...", flush=True)

        num_pots = 100

        energies = manager.compute_energy(
            np.ones((num_pots, len(np.where(pot_template.active_mask)[0]))),
        )

        if rank == 0:
            print(energies, flush=True)
            print("Evaluating forces ...", flush=True)

            self.assertEqual(energies.shape, (num_pots,))

        forces = manager.compute_forces(
                np.ones((num_pots, len(np.where(pot_template.active_mask)[0])))
            )

        if rank == 0:
            print(forces.shape, flush=True)

            self.assertEqual(
                forces.shape, (num_pots, manager.struct.natoms, 3)
            )

def initialize_potential_template():

    potential_template = Template(
        pvec_len=137,
        u_ranges = [(-1, 1), (-1, 1)],
        spline_ranges=[(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3),
                       (-0.5, 1), (-0.5, 1), (-2, 3), (-2, 3), (-7, 2),
                       (-7, 2), (-7, 2)],
        spline_indices=[(0, 15), (15, 30), (30, 45), (45, 58), (58, 71),
                         (71, 77), (77, 83), (83, 95), (95, 107),
                         (107, 117), (117, 127), (127, 137)]
    )

    mask = np.ones(potential_template.pvec.shape)

    potential_template.pvec[12] = 0; mask[12] = 0 # rhs phi_A knot
    potential_template.pvec[14] = 0; mask[14] = 0 # rhs phi_A deriv

    potential_template.pvec[27] = 0; mask[27] = 0 # rhs phi_B knot
    potential_template.pvec[29] = 0; mask[29] = 0 # rhs phi_B deriv

    potential_template.pvec[42] = 0; mask[42] = 0 # rhs phi_B knot
    potential_template.pvec[44] = 0; mask[44] = 0 # rhs phi_B deriv

    potential_template.pvec[55] = 0; mask[55] = 0 # rhs rho_A knot
    potential_template.pvec[57] = 0; mask[57] = 0 # rhs rho_A deriv

    potential_template.pvec[68] = 0; mask[68] = 0 # rhs rho_B knot
    potential_template.pvec[70] = 0; mask[70] = 0 # rhs rho_B deriv

    potential_template.pvec[92] = 0; mask[92] = 0 # rhs f_A knot
    potential_template.pvec[94] = 0; mask[94] = 0 # rhs f_A deriv

    potential_template.pvec[104] = 0; mask[104] = 0 # rhs f_B knot
    potential_template.pvec[106] = 0; mask[106] = 0 # rhs f_B deriv

    # potential_template.pvec[83:] = 0; mask[83:] = 0 # EAM params only
    potential_template.pvec[45:] = 0; mask[45:] = 0 # EAM params only

    potential_template.active_mask = mask

    return potential_template

# if __name__ == "__main__":
#     main()
