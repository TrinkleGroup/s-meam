import pickle
import logging
import numpy as np
import src.partools as partools
from src.meam import MEAM
import src.meam
from mpi4py import MPI
# from memory_profiler import profile

LOGGING = False

if LOGGING:
    output = print
else:
    output = logging.info

class Manager:

    def __init__(self, manager_id, comm, potential_template):
        """
        Manages the evaluation of a given structure using a set of processors

        Args:
            manager_id (int): for tracking which manager is doing what
            struct_name (Worker): the name of the structure being evaluated
            comm (MPI.Comm): communicator linked to available processors
            potential_template (Template): holds all MEAM potential information
        """

        self.id = manager_id
        self.comm = comm
        self.num_procs = comm.Get_size()
        self.proc_rank = comm.Get_rank()

        self.pot_template = potential_template

        self.struct_name = None
        self.struct = None

    # @profile
    def load_structure(self, struct_name, db_path):

        # TODO: should load into shared memory with all procs on node

        if self.proc_rank == 0:
            struct = pickle.load(open(db_path + struct_name + '.pkl', 'rb'))
        else:
            struct = None

        return struct

    # @profile
    def broadcast_struct(self, struct):
        return self.comm.bcast(struct, root=0)

    def compute_energy(self, master_pop):
        """Evaluates the structure energy for the whole population"""

        # TODO: handle evaluating one potential at a time
        # TODO: insert_active should happen in ga instead of in each fxn

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            # full = self.pot_template.insert_active_splines(full)

            only_one_pot = (full.shape[0] == 1)

            if only_one_pot:
                full = [full] * self.num_procs
            else:
                full = np.array_split(full, self.num_procs)

        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        # eng, max_ni, min_ni = self.struct.compute_energy(pop, self.pot_template.u_ranges)
        # eng, ni = self.struct.compute_energy(pop)
        eng, ni = self.struct.compute_energy(pop, self.pot_template.u_ranges)

        all_eng = self.comm.gather(eng, root=0)
        all_ni = self.comm.gather(ni, root=0)

        min_ni = None
        max_ni = None
        avg_ni = None
        ni_var = None
        frac_in = None

        if self.proc_rank == 0:
            all_ni = np.vstack(all_ni)

            per_type_ni = []
            frac_in = []

            for i in range(self.struct.ntypes):
                type_ni = all_ni[:, self.struct.type_of_each_atom - 1 == i]

                per_type_ni.append(
                    type_ni
                )

                # num_in = np.where(np.logical_and(ni <= 1, ni >= -1))
                num_in = np.logical_and(type_ni >= -1.2, type_ni <= 1.2).sum(axis=1)

                frac_in.append(num_in / type_ni.shape[1])

            min_ni = [np.min(ni, axis=1) for ni in per_type_ni]
            max_ni = [np.max(ni, axis=1) for ni in per_type_ni]
            avg_ni = [np.average(ni, axis=1) for ni in per_type_ni]
            ni_var = [np.std(ni, axis=1)**2 for ni in per_type_ni]

            if only_one_pot:
                all_eng = all_eng[0]
            else:
                all_eng = np.concatenate(all_eng)

        return all_eng, min_ni, max_ni, avg_ni, ni_var, frac_in
        # return all_eng, min_ni, max_ni, avg_ni, ni_var

    def compute_forces(self, master_pop):
        """Evaluates the structure forces for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            # full = self.pot_template.insert_active_splines(full)

            only_one_pot = (full.shape[0] == 1)

            if only_one_pot:
                full = [full] * self.num_procs
            else:
                full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        fcs = self.struct.compute_forces(pop, self.pot_template.u_ranges)

        all_fcs = self.comm.gather(fcs, root=0)

        if self.proc_rank == 0:
            if only_one_pot:
                all_fcs = all_fcs[0]
            else:
                all_fcs = np.vstack(all_fcs)

        return all_fcs


    # @profile
    def compute_energy_grad(self, master_pop):
        """Evaluates the structure energy gradient for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            # full = self.pot_template.insert_active_splines(full)

            only_one_pot = (full.shape[0] == 1)

            if only_one_pot:
                full = [full] * self.num_procs
            else:
                full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        eng_grad = self.struct.energy_gradient_wrt_pvec(pop, self.pot_template.u_ranges)

        all_eng_grad = self.comm.gather(eng_grad, root=0)

        if self.proc_rank == 0:
            if only_one_pot:
                all_eng_grad = all_eng_grad[0]
            else:
                all_eng_grad = np.vstack(all_eng_grad)

        return all_eng_grad

    # @profile
    def compute_forces_grad(self, master_pop):
        """Evaluates the structure for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            # full = self.pot_template.insert_active_splines(full)

            only_one_pot = (full.shape[0] == 1)

            if only_one_pot:
                full = [full] * self.num_procs
            else:
                full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        fcs_grad = self.struct.forces_gradient_wrt_pvec(pop, self.pot_template.u_ranges)

        all_fcs_grad = self.comm.gather(fcs_grad, root=0)

        if self.proc_rank == 0:
            if only_one_pot:
                all_fcs_grad = all_fcs_grad[0]
            else:
                all_fcs_grad = np.vstack(all_fcs_grad)

        return all_fcs_grad
