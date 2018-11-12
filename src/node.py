"""Library for hybrid parallelization of database evaluation"""

import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool, Process, Array
from multiprocessing.managers import BaseManager
from scipy.optimize import least_squares
from src.database import Database

logging.basicConfig(filename='node.log',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

class EvaluationManager(BaseManager):
    pass

class Node:

    def __init__(self, database, potential_template, procs_per_mpi):
    # def __init__(self, manager, potential_template, procs_per_mpi):
        """
        Args:
            procs_per_mpi: number of processors available to the Node
            """

        # self.database = manager.get_database().get_database()

        # manager = manager.get_database()
        #
        # self.database = Database.manual_init(
        #     manager.get_structures(), manager.get_energies(),
        #     manager.get_forces(), manager.get_weights(),
        #     manager.get_ref_struct(), manager.get_ref_energy()
        # )
        #
        # rzm: what the heck does an AutoProxy object look like...
        # print("Typecheck", type(self.database), flush=True)
        self.potential_template = potential_template
        self.procs_per_mpi = procs_per_mpi

        self.assignments, _ , _ = self.partition_work(database)
        self.num_structs = len(database.structures)

        # functions_list = []
        #
        # for db in assignments:
        #     functions_list.append(self.build_evaluation_functions(db))
        #
        # self.manager = EvaluationManager()
        #
        # def f_wrapper(idx):
        #     return functions_list[idx][0]()
        #
        # def g_wrapper(idx):
        #     return functions_list[idx][1]()
        #
        # self.manager.register('eval_f', callable=f_wrapper)
        # self.manager.register('eval_g', callable=g_wrapper)
        #
        # self.assignments = manager.list(assignments)

        # self.pool = Pool(processes=procs_per_mpi, args=self.database)

    def evaluate_f(self, x):

        # results = Array('f', np.zeros((2*self.num_structs, x.shape[0])))
        results = Array('f', 2*self.num_structs*x.shape[0])

        f_pool = [
            Process(
                target=fxn,
                args=(task, x, self.potential_template, results)
            ) for task in self.assignments
        ]

        for p in f_pool:
            p.start()

        # [p.start() for p in f_pool]
        # results = [p.join() for p in f_pool]

        # results = []
        for p in f_pool:
            p.join()
            # print(p.join())
            # results.append(p.join())

        print('here', results)
        return np.array(results)

        # TODO: sort according to some given order
        # TODO: should only pass names, not full database
        # return np.concatenate(
        #     self.pool.starmap(
        #         # self.manager.eval_f,
        #         # range(self.procs_per_mpi)
        #         fxn,
        #         [(db, x, self.potential_template) for db in self.assignments]
        #     )
        # )

    def evaluate_grad(self, x):
        return np.vstack(
            self.pool.starmap(
                grad,
                [(db, x, self.potential_template) for db in self.assignments]
            )
        )

    def local_minimization(self, x, max_nsteps):

        # Need to reshape returns for use in LM
        def wrap_f(x):
            return self.evaluate_f(x)[:, 0]

        def wrap_g(x):
            return self.evaluate_grad(x)[:, 0]

        opt_results = least_squares(
            wrap_f, x, wrap_g, method='lm', max_nfev=max_nsteps
        )

        return opt_results

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

    def partition_work(self, database):
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

def fxn(database, pot, potential_template, fitnesses):

    pot = np.atleast_2d(pot)

    full = potential_template.insert_active_splines(pot)

    w_energies = np.zeros((len(database.structures), full.shape[0]))
    t_energies = np.zeros(len(database.structures))
    # w_energies = np.zeros((len(database), full.shape[0]))
    # t_energies = np.zeros(len(database))

    fcs_fitnesses = np.zeros((len(database.structures), full.shape[0]))
    # fcs_fitnesses = np.zeros((len(database), full.shape[0]))

    ref_energy = np.zeros(full.shape[0])

    for j,name in enumerate(database.structures.keys()):
    # for j,name in enumerate(database):

        w = database.structures[name]
        # w = self.database.structures[name]

        w_energies[j, :] = w.compute_energy(
            full, potential_template.u_ranges
        )

        t_energies[j] = database.true_energies[name]
        # t_energies[j] = self.database.true_energies[name]

        if name == database.reference_struct:
        # if name == self.database.reference_struct:
            ref_energy = w_energies[j, :]

        w_fcs = w.compute_forces(full, potential_template.u_ranges)
        true_fcs = database.true_forces[name]
        # true_fcs = self.database.true_forces[name]

        fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
        fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

        fcs_fitnesses[j, :] = fcs_err

    w_energies = (w_energies - ref_energy)
    t_energies -= database.reference_energy
    # t_energies -= self.database.reference_energy

    eng_fitnesses = np.zeros((len(database.structures), pot.shape[0]))
    # eng_fitnesses = np.zeros((len(self.database.structures), pot.shape[0]))

    for j, (w_eng, t_eng) in enumerate(zip(w_energies, t_energies)):
        eng_fitnesses[j, :] = (w_eng - t_eng) ** 2

    fitnesses = np.frombuffer(fitnesses)
    fitnesses = fitnesses.reshape((2*len(database.structures), pot.shape[0]))
    fitnesses += np.concatenate([eng_fitnesses, fcs_fitnesses])

    return fitnesses

def grad(database, pot, potential_template):
    pot = np.atleast_2d(pot)

    full = potential_template.insert_active_splines(pot)

    fcs_grad_vec = np.zeros(
        (len(database.structures), full.shape[0], full.shape[1])
        # (len(database), full.shape[0], full.shape[1])
    )

    w_energies = np.zeros((len(database.structures), full.shape[0]))
    t_energies = np.zeros(len(database.structures))
    # w_energies = np.zeros((len(database), full.shape[0]))
    # t_energies = np.zeros(len(database))

    ref_energy = 0

    for j,name in enumerate(database.structures.keys()):
    # for j,name in enumerate(database):
        w = database.structures[name]
        # w = self.database.structures[name]

        w_energies[j, :] = w.compute_energy(
            full, potential_template.u_ranges
        )

        t_energies[j] = database.true_energies[name]
        # t_energies[j] = self.database.true_energies[name]

        if name == database.reference_struct:
        # if name == self.database.reference_struct:
            ref_energy = w_energies[j, :]

        w_fcs = w.compute_forces(full, potential_template.u_ranges)
        true_fcs = database.true_forces[name]
        # true_fcs = self.database.true_forces[name]

        fcs_err = ((w_fcs - true_fcs) / np.sqrt(10))

        fcs_grad = w.forces_gradient_wrt_pvec(
            full, potential_template.u_ranges
        )

        scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
        summed = scaled.sum(axis=1).sum(axis=1)

        fcs_grad_vec[j, :] += (2 * summed / 10)

    w_energies = (w_energies - ref_energy)
    t_energies -= database.reference_energy
    # t_energies -= self.database.reference_energy

    eng_grad_vec = np.zeros(
        (len(database.structures), full.shape[0], full.shape[1])
        # (len(database), full.shape[0], full.shape[1])
    )

    for j, (w_eng, t_eng) in enumerate(zip(w_energies, t_energies)):
        eng_err = (w_eng - t_eng)
        eng_grad = w.energy_gradient_wrt_pvec(
            full, potential_template.u_ranges
        )

        eng_grad_vec[j, :] += (eng_err * eng_grad.T * 2).T

    grad_vec = np.vstack([eng_grad_vec, fcs_grad_vec])
    return grad_vec[:, :, np.where(potential_template.active_mask)[0]]
