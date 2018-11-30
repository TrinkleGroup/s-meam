import pickle
import logging
import numpy as np
from mpi4py import MPI

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

    def load_structure(self, struct_name, db_path):

        if self.proc_rank == 0:
            if self.struct is None:
                print(
                    "Manager " + str(self.id) + " is loading " + struct_name,
                    end=""
                )
            else:
                print(
                    "Manager " + str(self.id) + " is overwriting " + \
                    self.struct_name + " with " + struct_name,
                    end=""
                )

            print(" ... ", end="", flush=True)
            struct = pickle.load(open(db_path + struct_name + '.pkl', 'rb'))
            print("done", flush=True)
        else:
            struct = None

        struct = self.comm.bcast(struct, root=0)

        return struct

    def compute_energy(self, master_pop):
        """Evaluates the structure for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            full = self.pot_template.insert_active_splines(full)
            full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        eng = self.struct.compute_energy(pop, self.pot_template.u_ranges)

        all_eng = self.comm.gather(eng, root=0)

        if self.proc_rank == 0:
            all_eng = np.concatenate(all_eng)

        return all_eng

    def compute_forces(self, master_pop):
        """Evaluates the structure for the whole population"""

        if self.proc_rank == 0:
            full = np.atleast_2d(master_pop)
            full = self.pot_template.insert_active_splines(full)
            full = np.array_split(full, self.num_procs)
        else:
            full = None

        pop = self.comm.scatter(full, root=0)

        fcs = self.struct.compute_forces(pop, self.pot_template.u_ranges)

        all_fcs = self.comm.gather(fcs, root=0)

        if self.proc_rank == 0:
            all_fcs = np.vstack(all_fcs)

        return all_fcs
