import sys
sys.path.insert(0, './')

import numpy as np
np.random.seed(42)

import os
import time
import glob
import pickle
from prettytable import PrettyTable
from pprint import pprint
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
        all_struct_file_names = get_structure_list()[:comm.size-1]

        atom_names = []

        for evaluator_name in all_struct_file_names:
            fname = os.path.split(evaluator_name)[-1]
            short_name = os.path.splitext(fname)[-1]

            atom_names.append(short_name[1:])

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

    # workers load structures
    if rank > 0:
        struct_name = comm.recv(source=0, tag=1)
        print("Dobby number", rank, "is loading:", struct_name, flush=True)

        #atoms = src.lammpsTools.atoms_from_file(struct_name, types)
        #w = Worker(atoms, knot_pos, indices, types)
        w = pickle.load(open(struct_name, 'rb'))

    # create starting spline parameters
    if rank == 0:
        spline_parameters = np.random.random(
                knot_pos.shape[0] + 2*num_splines)

        spline_parameters = np.atleast_2d(spline_parameters)

    else:
        spline_parameters = None

    y = comm.bcast(spline_parameters, root=0)

    val = None

    if rank > 0:
        val = compute_energy(w, y) / len(w.atoms)

    all_eng = comm.gather(val, root=0)

    if rank == 0:
        all_eng = {atom_names[i]:all_eng[i+1] for i in range(len(atom_names))}

        correct_values = {}
        info_file_names = glob.glob("data/fitting_databases/seed_42/info.*")

        for i,name in enumerate(info_file_names):
            with open(name, 'r') as info_file:
                natoms = int(info_file.readline())
                eng = float(info_file.readline())

                fname = os.path.split(name)[-1]
                short_name = os.path.splitext(fname)[-1][1:]
                correct_values[short_name] = eng

        table = PrettyTable()
        table.field_names = ["name", "expected", "actual"]
        table.float_format = ".12"
        
        for field_name in table.field_names:
                table.align[field_name] = 'r'

        for name in atom_names:
            calc = all_eng[name][0]
            true_val = correct_values[name]

            table.add_row([name, calc, true_val])

            diff = np.abs(calc - true_val)
            np.testing.assert_almost_equal(diff, 0.0, decimal=12)

        print()
        print(table)

def init_slave(comm):
    return w

def eval_slave():
    return compute_energy(w, y)

################################################################################

def get_structure_list():
    from tests.testStructs import allstructs

    num_copies = int(sys.argv[1]) if (len(sys.argv) > 1) else MPI.COMM_WORLD.size-1

    val = ["data/test_db/data.bulk_periodic_rhombo_type2"]*num_copies
    val = ["data/fitting_databases/seed_42/evaluator.aaa"]*num_copies
    val = glob.glob("data/fitting_databases/seed_42/evaluator.*")

    return val

def get_potential_information():
    import src.meam
    from src.meam import MEAM
    from tests.testPotentials import get_random_pots

    #potential = get_random_pots(1)['meams'][0]
    potential = MEAM.from_file('data/fitting_databases/seed_42/seed_42.meam')

    x_pvec, _, indices = src.meam.splines_to_pvec(potential.splines)

    return x_pvec, indices, potential.types

def compute_energy(w, y):
    return w.compute_energy(y)

def compute_forces(w, y):
    return w.compute_forces(y)

################################################################################

if __name__ == "__main__":
    main()
