import sys
sys.path.insert(0, './')

import numpy as np
np.random.seed(42)

import os
import time
import glob
from mpi4py import MPI

import src.lammpsTools
from src.worker import Worker

################################################################################
"""Assumes exactly 1 structure per processor"""

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # initialize potential form
        knot_positions, spline_start_indices, atom_types = \
                get_potential_information()

        num_splines = len(spline_start_indices)

        # get desired structure names
        all_struct_file_names = get_structure_list()

        print("Harry has created", len(all_struct_file_names), "tasks", flush=True)
        print(flush=True)

        if len(all_struct_file_names) > comm.size-1:
            raise ValueError("number of structures != number of workers")

        for slave_num in range(1, comm.size):
            comm.send(
                all_struct_file_names[slave_num-1],
                dest=slave_num,
                tag=1,
                )
    else:
        knot_positions = None
        spline_start_indices = None
        atom_types = None

    # broadcast potential information
    knot_pos = comm.bcast(knot_positions, root=0)
    indices = comm.bcast(spline_start_indices, root=0)
    types = comm.bcast(atom_types, root=0)

    # workers build structures
    if rank > 0:
        struct_name = comm.recv(source=0, tag=1)
        print("Dobby number", rank, "is building:", struct_name, flush=True)

        atoms = src.lammpsTools.atoms_from_file(struct_name, types)
        w = Worker(atoms, knot_pos, indices, types)


    # create starting spline parameters
    if rank == 0:
        spline_parameters = np.random.random(
                knot_positions.shape[0] + 2*num_splines)

        spline_parameters = np.atleast_2d(spline_parameters)

    else:
        spline_parameters = None

    y = comm.bcast(spline_parameters, root=0)

    val = None

    if rank > 0:
        val = compute_energy(w, y)

    all_eng = comm.gather(val, root=0)
    
    if rank == 0:
        all_eng = np.concatenate(all_eng[1:])
        print(all_eng)

def init_slave(comm):
    return w

def eval_slave():
    return compute_energy(w, y)

################################################################################

def get_structure_list():
    from tests.testStructs import allstructs

    #test_name = '8_atoms'                                                        

    num_copies = int(sys.argv[1]) if (len(sys.argv) > 1) else MPI.COMM_WORLD.size-1
    #tmp = [allstructs[test_name]]*num_copies
    #dct = {test_name+'_v{0}'.format(i+1):tmp[i] for i in range(len(tmp))} 

    #dct = allstructs

    #return list(dct.keys())#, list(dct.values())
    val = ["data/test_db/data.bulk_periodic_rhombo_type2"]*num_copies
    val = glob.glob("data/test_db/data.*")[:MPI.COMM_WORLD.size-1]
    val = ["data/test_db/data.aaa"]*num_copies

    return val

def get_potential_information():
    import src.meam
    from src.meam import MEAM
    from tests.testPotentials import get_random_pots

    potential = get_random_pots(1)['meams'][0]
    potential = MEAM.from_file("data/pot_files/HHe.meam.spline")

    x_pvec, _, indices = src.meam.splines_to_pvec(potential.splines)

    return x_pvec, indices, potential.types

def compute_energy(w, y):
    return w.compute_energy(y)

def compute_forces(w, y):
    return w.compute_forces(y)

################################################################################

if __name__ == "__main__":
    main()
