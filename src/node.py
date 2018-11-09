"""Library for hybrid parallelization of database evaluation"""

import numpy as np
import logging
from multiprocessing import Pool
from src.database import Database

logging.basicConfig(filename='node.log',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

class Node:

    def __init__(self, database, potential_template, procs_per_mpi):
        """
        Args:
            procs_per_mpi: number of processors available to the Node
        """

        # TODO: structures in shared memory?? or is that why I separated...

        self.pool = Pool(processes=procs_per_mpi)

        self.database = database
        self.potential_template = potential_template
        self.procs_per_mpi = procs_per_mpi

        self.assignments, _ , _ = self.partition_work()

    def evaluate_f(self, x):
        # TODO: sort according to some given order
        return np.concatenate(
            self.pool.starmap(
                fxn,
                [(db, x, self.potential_template) for db in self.assignments]
            )
        )

    def evaluate_grad(self, x):
        return np.vstack(
            self.pool.starmap(
                grad,
                [(db, x, self.potential_template) for db in self.assignments]
            )
        )

    # def build_grouped_eval_fxns(self, work_weights, name_list):
    #     grouped_functions = []
    #
    #     for weight_block, name_block in zip(work_weights, name_list):
    #         function_block = []
    #
    #         mini_database = {self.database[s_name] for s_name in name_block}
    #
    #         for weight, name in zip(weight_block, name_block):
    #             function_block.append(
    #                 build_eval_fxns(
    #                     mini_database, self.potential_template
    #                 )
    #             )
    #
    #         grouped_functions.append(function_block)
    #
    #     return grouped_functions

    def compute_relative_weights(self):
        work_weights = []
        name_list = list(self.database.structures.keys())

        for name in name_list:
            work_weights.append(self.database.structures[name].natoms)

        work_weights = np.array(work_weights)
        work_weights = work_weights / np.min(work_weights)
        work_weights = work_weights*work_weights # cost assumed to scale as N^2

        return work_weights, name_list

    def partition_work(self):
        """Groups workers based on evaluation time to help with load balancing.

        Returns:
            distributed_work (list): the partitioned work
            work_per_proc (float): approximate work done by each processor
        """
        # TODO: record scaling on first run, then redistribute to load balance

        unassigned_structs = list(self.database.structures.keys())

        work_weights, name_list = self.compute_relative_weights()

        work_per_proc = np.sum(work_weights) / self.procs_per_mpi

        # work_cumsum = np.cumsum(work_weights).tolist()

        work_weights = work_weights.tolist()

        assignments = []
        grouped_work = []
        grouped_databases = []

        for _ in range(self.procs_per_mpi):
            cumulated_work = 0

            names = []
            work_for_one_proc = []

            while unassigned_structs and (cumulated_work < work_per_proc):
                names.append(unassigned_structs.pop())
                cumulated_work += work_weights.pop()

            mini_database = Database.manual_init(
                {name:self.database.structures[name] for name in names},
                {name:self.database.true_energies[name] for name in names},
                {name:self.database.true_forces[name] for name in names},
                {name:self.database.weights[name] for name in names},
                self.database.reference_struct,
                self.database.reference_energy
            )

            assignments.append(mini_database)
            grouped_work.append(cumulated_work)

        return assignments, grouped_work, work_per_proc

def fxn(database, pot, potential_template):
    full = potential_template.insert_active_splines(pot)

    w_energies = np.zeros(len(database.structures))
    t_energies = np.zeros(len(database.structures))

    fcs_fitnesses = np.zeros(len(database.structures))

    ref_energy = 0

    for j,name in enumerate(database.structures.keys()):

        w = database.structures[name]

        w_energies[j] = w.compute_energy(
            full, potential_template.u_ranges
        )

        t_energies[j] = database.true_energies[name]

        if name == database.reference_struct:
            ref_energy = w_energies[j]

        w_fcs = w.compute_forces(full, potential_template.u_ranges)
        true_fcs = database.true_forces[name]

        fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
        fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

        fcs_fitnesses[j] = fcs_err

    w_energies -= ref_energy
    t_energies -= database.reference_energy

    eng_fitnesses = np.zeros(len(database.structures))

    for j, (w_eng, t_eng) in enumerate(zip(w_energies, t_energies)):
        eng_fitnesses[j] = (w_eng - t_eng) ** 2

    fitnesses = np.concatenate([eng_fitnesses, fcs_fitnesses])

    return fitnesses

def grad(database, pot, potential_template):
    full = potential_template.insert_active_splines(pot)

    fcs_grad_vec = np.zeros((len(database.structures), 137))

    w_energies = np.zeros(len(database.structures))
    t_energies = np.zeros(len(database.structures))

    ref_energy = 0

    for j,name in enumerate(database.structures.keys()):
        w = database.structures[name]

        w_energies[j] = w.compute_energy(
            full, potential_template.u_ranges
        )

        t_energies[j] = database.true_energies[name]

        if name == database.reference_struct:
            ref_energy = w_energies[j]

        w_fcs = w.compute_forces(full, potential_template.u_ranges)
        true_fcs = database.true_forces[name]

        fcs_err = ((w_fcs - true_fcs) / np.sqrt(10))

        fcs_grad = w.forces_gradient_wrt_pvec(
            full, potential_template.u_ranges
        )

        scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
        summed = scaled.sum(axis=1).sum(axis=1)

        fcs_grad_vec[j] += (2 * summed / 10).ravel()

    w_energies -= ref_energy
    t_energies -= database.reference_energy

    eng_grad_vec = np.zeros((len(database.structures), 137))
    for j, (w_eng, t_eng) in enumerate(zip(w_energies, t_energies)):
        eng_err = (w_eng - t_eng)
        eng_grad = w.energy_gradient_wrt_pvec(
            full, potential_template.u_ranges
        )

        eng_grad_vec[j] += (eng_err * eng_grad * 2).ravel()

    grad_vec = np.vstack([eng_grad_vec, fcs_grad_vec])
    return grad_vec[:, np.where(potential_template.active_mask)[0]]
