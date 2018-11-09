import sys
sys.path.append('./')
import unittest
import numpy as np
from ase import Atoms
from src.node import Node
from src.database import Database
from src.potential_templates import Template
from src.worker import Worker
import logging
logging.basicConfig(filename='node.log',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

class NodeTests(unittest.TestCase):

    def setUp(self):

        r0 = 2.5

        atoms = Atoms(
            [1, 2, 1],
            positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
        )

        atoms.set_pbc(True)
        atoms.center(vacuum=5.0)

        structures = {"example%d" % i: atoms.copy() for i in range(10)}
        # energies = {name: np.random.random() for name in structures.keys()}
        # forces = {name: np.random.random((3, 3)) for name in structures.keys()}

        name_list = list(structures.keys())

        energies = {name: 1 for name in name_list}
        forces = {name: np.ones((3, 3)) for name in name_list}
        weights = {name: 1 for name in name_list}

        ref_struct = "example0"
        ref_energy = energies[ref_struct]

        x_pvec, indices  = example_pot()

        structures = {name:
            Worker(atoms, x_pvec, indices, ['H', 'He'])
            for name,atoms in structures.items()
        }

        database = Database.manual_init(
            structures, energies, forces, weights, ref_struct, ref_energy
        )

        template = build_template()
        procs_per_mpi = 4

        self.node = Node(database, template, procs_per_mpi)

    def test_weighting_basic(self):
        np.testing.assert_equal(
            np.ones(len(self.node.database.structures)),
            self.node.compute_relative_weights()[0]
        )

    def test_job_assignment(self):
        assignments, distributed_work, work_per_proc = \
            self.node.partition_work()

        min_work = np.min(distributed_work)

        for work_block in distributed_work:
            self.assertTrue(
                np.abs(work_block - work_per_proc) < work_per_proc
                )

    def test_eval_f_doesnt_crash(self):
        logging.info(self.node.evaluate_f(np.random.random(123)))
        self.node.evaluate_f(np.random.random(123))

    def test_eval_grad_doesnt_crash(self):
        logging.info(self.node.evaluate_grad(np.random.random(123))[:,:3])
        logging.info(self.node.evaluate_grad(np.random.random(123)).shape)
        self.node.evaluate_grad(np.random.random(123))

    def test_gradients(self):
        pass

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

    potential_template.active_mask = mask

    return potential_template

def example_pot():
    x_pvec = [1.7426928371636325, 2.0558017673999966, 2.3689106976363608, 2.6820196278727244, 2.9951285581090885, 3.3082374883454526, 3.6213464185818163, 3.9344553488181804, 4.2475642790545445, 4.560673209290909, 4.873782139527272, 5.186891069763636, 5.5, 1.7426928371636325, 2.0558017673999966, 2.3689106976363608, 2.6820196278727244, 2.9951285581090885, 3.3082374883454526, 3.6213464185818163, 3.9344553488181804, 4.2475642790545445, 4.560673209290909, 4.873782139527272, 5.186891069763636, 5.5, 1.7426928371636325, 2.0558017673999966, 2.3689106976363608, 2.6820196278727244, 2.9951285581090885, 3.3082374883454526, 3.6213464185818163, 3.9344553488181804, 4.2475642790545445, 4.560673209290909, 4.873782139527272, 5.186891069763636, 5.5, 2.0558017673999966, 2.291221590659997, 2.5266414139199975, 2.7620612371799975, 2.997481060439998, 3.2329008836999984, 3.468320706959999, 3.7037405302199993, 3.9391603534799993, 4.17458017674, 4.41, 2.0558017673999966, 2.291221590659997, 2.5266414139199975, 2.7620612371799975, 2.997481060439998, 3.2329008836999984, 3.468320706959999, 3.7037405302199993, 3.9391603534799993, 4.17458017674, 4.41, -55.142331649275434, -44.740989903709206, -34.33964815814298, -23.938306412576747, -55.142331649275434, -44.740989903709206, -34.33964815814298, -23.938306412576747, 2.0558017673999966, 2.317379348799997, 2.5789569301999973, 2.840534511599998, 3.1021120929999983, 3.3636896743999984, 3.625267255799999, 3.8868448371999995, 4.1484224186, 4.41, 2.0558017673999966, 2.317379348799997, 2.5789569301999973, 2.840534511599998, 3.1021120929999983, 3.3636896743999984, 3.625267255799999, 3.8868448371999995, 4.1484224186, 4.41, -1.000000000000002, -0.7245090543770937, -0.4490181087541855, -0.1735271631312772, 0.1019637824916311, 0.3774547281145394, 0.6529456737374476, 0.9284366193603559, -1.000000000000002, -0.7245090543770937, -0.4490181087541855, -0.1735271631312772, 0.1019637824916311, 0.3774547281145394, 0.6529456737374476, 0.9284366193603559, -1.000000000000002, -0.7245090543770937, -0.4490181087541855, -0.1735271631312772, 0.1019637824916311, 0.3774547281145394, 0.6529456737374476, 0.9284366193603559]

    indices = [0, 13, 26, 39, 50, 61, 65, 69, 79, 89, 97, 105]

    return x_pvec, indices

if __name__ == '__main__':
    unittest.main()
