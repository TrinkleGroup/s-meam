"""Library for hybrid parallelization of database evaluation"""

import numpy as np
import logging
from scipy.optimize import least_squares
from src.database import Database

logging.basicConfig(filename='node.log',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

class Node:

    def __init__(self, database, potential_template, comm):
        """
        An object for organizing jobs on a single compute node (e.g. a
        32-core node on Blue Waters)

        Args:
            database (Database): subset of full database stored on node
            potential_template (Template): stores potential form information
            comm (MPI.Communicator): communicator for all procs on node
        """

        # TODO: how do you ensure all node procs are physically ON the node?

        self.comm = comm
        self.mpi_size = comm.Get_size()
        self.database = self.group_database_subset(database, mpi_size)

        self.potential_template = potential_template
        self.num_structs = len(self.database.structures)

        self.fxn, self.grad = self.build_functions()

    def evaluate_f(self, x):
        pass

    def evaluate_grad(self, x):
        pass

    def local_minimization(self, x, max_nsteps):
        pass

    def compute_relative_weights(self, database):
        work_weights = []
        # name_list = list(self.database.structures.keys())
        name_list = list(database.structures.keys())

        for name in name_list:
            # work_weights.append(self.database.structures[name].natoms)
            work_weights.append(database.structures[name].natoms)

        work_weights = np.array(work_weights)
        work_weights = work_weights / np.min(work_weights)
        work_weights = work_weights*work_weights # cost assumed to scale as N^2

        return work_weights, name_list

    def partition_work(self, database, mpi_size):
        """Groups workers based on evaluation time to help with load balancing.

        Returns:
            distributed_work (list): the partitioned work
            work_per_proc (float): approximate work done by each processor
        """
        # TODO: record scaling on first run, then redistribute to load balance

        # unassigned_structs = list(self.database.structures.keys())
        unassigned_structs = list(database.structures.keys())

        work_weights, name_list = self.compute_relative_weights(database)

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
                {name:database.structures[name] for name in names},
                {name:database.true_energies[name] for name in names},
                {name:database.true_forces[name] for name in names},
                {name:database.weights[name] for name in names},
                database.reference_struct,
                database.reference_energy
            )

            assignments.append(mini_database)
            # assignments.append(names)
            grouped_work.append(cumulated_work)

        return assignments, grouped_work, work_per_proc

    # def build_evaluation_functions(self, database):

    def build_functions(self):

        def fxn(pot):

            pot = np.atleast_2d(pot)

            full = self.potential_template.insert_active_splines(pot)

            w_energies = np.zeros((self.num_structs,full.shape[0]))
            t_energies = np.zeros(self.num_structs)

            fcs_fitnesses = np.zeros((self.num_structs, full.shape[0]))

            ref_energy = np.zeros(full.shape[0])

            for j, name in enumerate(self.database.structures.keys()):

                w = self.database.structures[name]

                w_energies[j, :] = w.compute_energy(
                    full, self.potential_template.u_ranges
                )

                t_energies[j] = self.database.true_energies[name]

                if name == self.database.reference_struct:
                    ref_energy = w_energies[j, :]

                w_fcs = w.compute_forces(full, self.potential_template.u_ranges)
                true_fcs = self.database.true_forces[name]

                fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
                fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

                fcs_fitnesses[j, :] = fcs_err

            w_energies = (w_energies - ref_energy)
            t_energies -= self.database.reference_energy

            eng_fitnesses = np.zeros(
                (self.num_structs, pot.shape[0])
            )

            for j, (w_eng, t_eng) in enumerate(zip(w_energies, t_energies)):
                eng_fitnesses[j, :] = (w_eng - t_eng) ** 2

            fitnesses = np.concatenate([eng_fitnesses, fcs_fitnesses])

            return fitnesses

        def grad(pot):
            pot = np.atleast_2d(pot)

            full = self.potential_template.insert_active_splines(pot)

            fcs_grad_vec = np.zeros(
                (self.num_structs, full.shape[0], full.shape[1])
            )

            w_energies = np.zeros((self.num_structs, full.shape[0]))
            t_energies = np.zeros(self.num_structs)

            ref_energy = 0

            names = self.database.structures.keys()

            for j,name in enumerate(names):
                w = self.database.structures[name]

                w_energies[j, :] = w.compute_energy(
                    full, self.potential_template.u_ranges
                )

                t_energies[j] = self.database.true_energies[name]

                if name == self.database.reference_struct:
                    ref_energy = w_energies[j, :]

                w_fcs = w.compute_forces(full, self.potential_template.u_ranges)
                true_fcs = self.database.true_forces[name]

                fcs_err = ((w_fcs - true_fcs) / np.sqrt(10))

                fcs_grad = w.forces_gradient_wrt_pvec(
                    full, self.potential_template.u_ranges
                )

                scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
                summed = scaled.sum(axis=1).sum(axis=1)

                fcs_grad_vec[j, :] += (2 * summed / 10)

            w_energies = (w_energies - ref_energy)
            t_energies -= self.database.reference_energy

            eng_grad_vec = np.zeros(
                (self.num_structs, full.shape[0], full.shape[1])
            )

            for j, (name, w_eng, t_eng) in enumerate(
                    zip(names, w_energies, t_energies)):

                w = self.database.structures[name]

                eng_err = (w_eng - t_eng)
                eng_grad = w.energy_gradient_wrt_pvec(
                    full, self.potential_template.u_ranges
                )

                eng_grad_vec[j, :] += (eng_err * eng_grad.T * 2).T

            grad_vec = np.vstack([eng_grad_vec, fcs_grad_vec])

            return grad_vec[:, :, np.where(
                self.potential_template.active_mask)[0]]

        return fxn, grad
