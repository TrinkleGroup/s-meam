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

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            full = self.pot_template.insert_active_splines(full)
            full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        # eng, ni = self.struct.compute_energy(pop, self.pot_template.u_ranges)
        eng = self.struct.compute_energy(pop, self.pot_template.u_ranges)

        all_eng = self.comm.gather(eng, root=0)
        # all_ni = self.comm.gather(ni, root=0)

        if self.proc_rank == 0:
            all_eng = np.concatenate(all_eng)
            # all_ni = np.vstack(all_ni)

        return all_eng#, all_ni

    def compute_forces(self, master_pop):
        """Evaluates the structure forces for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            full = self.pot_template.insert_active_splines(full)
            full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        potential = MEAM.from_file('/projects/sciteam/baot/pz-unfx-cln/TiO.meam.spline')

        x_pvec, true_pvec, indices = src.meam.splines_to_pvec(potential.splines)

        ti_only_pvec = true_pvec.copy()

        mask = np.zeros(self.pot_template.pvec_len, dtype=int)

        mask[15:22] = 1 # phi_TiO
        mask[22:37] = 1 # phi_O
        mask[50:57] = 1 # rho_O
        mask[63:70] = 1 # U_O
        mask[82:89] = 1 # f_O
        mask[99:116] = 1 # g_TiO, g_O

        ti_only_pvec[np.where(mask)[0]] = 0
        ti_only_pvec = np.atleast_2d(ti_only_pvec)

        ti_fcs = self.struct.compute_forces(
            ti_only_pvec, self.pot_template.u_ranges
        )

        fcs = self.struct.compute_forces(pop, self.pot_template.u_ranges)
        fcs -= ti_fcs

        all_fcs = self.comm.gather(fcs, root=0)

        if self.proc_rank == 0:
            all_fcs = np.vstack(all_fcs)

        return all_fcs


    def compute_energy_grad(self, master_pop):
        """Evaluates the structure energy gradient for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            full = self.pot_template.insert_active_splines(full)
            full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        eng_grad = self.struct.energy_gradient_wrt_pvec(
            pop, self.pot_template.u_ranges
        )

        all_eng_grad = self.comm.gather(eng_grad, root=0)

        if self.proc_rank == 0:
            all_eng_grad = np.vstack(all_eng_grad)

        return all_eng_grad

    def compute_forces_grad(self, master_pop):
        """Evaluates the structure for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            full = self.pot_template.insert_active_splines(full)
            full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        # TODO: subtract off Ti-only forces from gradient?

        fcs_grad = self.struct.forces_gradient_wrt_pvec(
            pop, self.pot_template.u_ranges
        )

        all_fcs_grad = self.comm.gather(fcs_grad, root=0)

        if self.proc_rank == 0:
            all_fcs_grad = np.vstack(all_fcs_grad)

        return all_fcs_grad
