"""Library for hybrid parallelization of database evaluation"""

import numpy as np
from multiprocessing import Pool


class Node:

    def __init__(self, database, potential_template, procs_per_mpi):
        """
        Args:
            database (Database): collection of Workers and energies/forces
            potential_template (Template): for
            procs_per_mpi: number of processors available to the Node
        """

        self.database = database
        self.potential_template = potential_template
        self.pool = Pool(processes=procs_per_mpi)

        # TODO: each processor should work on a subset of the structures
        # TODO: build_eval_fxns for each subset; pools will receive the fxns

    def build_grouped_eval_fxns(self):
        pass

    def distribute_work(self, database, mpi_size):
        """Groups workers based on evaluation time to help with load balancing"""

        # TODO: record scaling on first run, then redistribute to load balance

        unassigned_structs = list(database.structures.keys())
        work_weights = []

        for name in unassigned_structs:
            work_weights.append(database.structures[name].natoms)

        work_weights = np.array(work_weights)
        work_weights /= np.min(work_weights)
        work_weights = work_weights*work_weights # cost assumed to scale as N^2

        work_per_proc = np.sum(work_weights) / mpi_size

        work_cumsum = np.cumsum(work_weights).tolist()
        structure_assignments = []

        for _ in range(mpi_size):
            cumulated_work = 0

            structs_for_one_proc = []

            while unassigned_structs and (cumulated_work < work_per_proc):
                structs_for_one_proc.append(unassigned_structs.pop())
                cumulated_work += work_cumsum.pop()

            structure_assignments.append(structs_for_one_proc)

        return struct_work_weights

