import unittest
import numpy as np
from ase import Atoms
from src.node import Node
from src.database import Database

class NodeTests(unittest.TestCase):

    def setUp(self):

        atoms = Atoms(
            [1, 2, 1],
            positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
        )

        atoms.set_pbc(True)
        atoms.center(vacuum=5.0)

        structures = {"example%d" % i: atoms.copy() for i in range(10)}
        energies = {name: np.random.random() for name in structures.keys()}
        forces = {name: np.random.random((3, 3)) for name in structures.keys()}
        weights = {name: 1 for name in structures.keys()}
        ref_struct = "example0"
        ref_energy = energies[ref_struct]


        database = Database.manual_init(
            structures, energies, forces, weights, ref_struct, ref_energy
        )

        template = build_template()
        procs_per_mpi = 2

        self.node = Node(database, template, procs_per_mpi)

    def test_job_assignment(self):
        pass

    def test_evaluation(self):
        pass

    def test_gradients(self):
        pass

    # rzm: finish making the stuff for Node; then complete the Hybird stuff

def build_template():

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


