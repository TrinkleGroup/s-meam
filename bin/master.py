import os
import sys
sys.path.insert(0, './')

import time
import numpy as np

seed = np.random.randint(0, high=(2**32 - 1))
#seed = int(sys.argv[1])
#seed = 42
np.random.seed(seed)
print("Seed value: {0}\n".format(seed))

from prettytable import PrettyTable
from pprint import pprint

import parsl
from parsl import *
import multiprocessing as mp

parsl.set_file_logger('parsl.log')
################################################################################

def main():
    # initialize structures and splines
    structures = get_structure_dict()

    print("Structures:")
    pprint(list(structures.keys()))
    print()

    knot_positions,spline_start_indices,atom_types = get_potential_information()
    num_splines = len(spline_start_indices)

    # construct workers given spline information
    worker_futures = dict_of_workers(
            structures, knot_positions, spline_start_indices, atom_types
            )

    # evaluate energies/forces using a random vector of spline parameters
    spline_parameters = np.random.random(
            knot_positions.shape[0] + 2*num_splines)

    energy_futures = dict_of_energy_tasks(worker_futures, spline_parameters)
    forces_futures = dict_of_forces_tasks(worker_futures, spline_parameters)

    eng_res = {key: job.result() for key, job in energy_futures.items()}
    fcs_res = {key: job.result() for key, job in forces_futures.items()}

    print("{0}".format(np.vstack(list(eng_res.values()))))
    #print("{0}".format(np.vstack(list(fcs_res.values()))))

################################################################################

num_threads = mp.cpu_count() - 1
print("Requesting {0}/{1} threads".format(num_threads, mp.cpu_count()))
print()

config = {
    "sites": [
        {"site": "Local_IPP",
         "auth": {
             "channel": None,
         },
         "execution": {
             "executor": 'ipp',
             "provider": "local", # Run locally
             "block": {  # Definition of a block
                 "minBlocks" : 1, # }
                 "maxBlocks" : 1, # }<---- Shape of the blocks
                 "initBlocks": 1, # }
                 "taskBlocks": num_threads, # <--- No. of workers in a block
                 "parallelism" : 1 # <-- Parallelism
             }
         }
        }]
}

dfk = DataFlowKernel(config=config, lazy_fail=False)

@App('python', dfk)
def build_worker(atoms, atoms_name, x_pvec, indices, types):
    import os
    import pickle

    from src.worker import Worker

    WORKER_SAVE_PATH = "/home/jvita/scripts/s-meam/project/data/workers/"
    allow_reload = False

    file_name = WORKER_SAVE_PATH + atoms_name + '.pkl'

    if allow_reload and os.path.isfile(file_name):
        w = pickle.load(open(file_name, 'rb'))
    else:
        w = Worker(atoms, x_pvec, indices, types)
        #pickle.dump(w, open(file_name, 'wb'))

    return w

@App('python', dfk)
def compute_energy(w, parameter_vector):
    return w.compute_energies(parameter_vector)

@App('python', dfk)
def compute_forces(w, parameter_vector):
    return w.compute_forces(parameter_vector)

################################################################################

def get_structure_dict():
    from tests.testStructs import allstructs

    test_name = '8_atoms'                                                        
    tmp = [allstructs[test_name]]*20
    allstructs = {test_name+'_v{0}'.format(i+1):tmp[i] for i in range(len(tmp))} 

    return allstructs

def get_potential_information():
    import src.meam
    from tests.testPotentials import get_random_pots

    potential = get_random_pots(1)['meams'][0]
    x_pvec, _, indices = src.meam.splines_to_pvec(potential.splines)

    return x_pvec, indices, potential.types

def dict_of_workers(struct_dict, knot_positions, spline_delimiters, atom_types):
    structure_futures = {}
    for struct_name, struct in struct_dict.items():

        structure_futures[struct_name] = build_worker(
                struct, struct_name, knot_positions, spline_delimiters,
                atom_types
                )

    return structure_futures

def dict_of_energy_tasks(structure_futures, y):
    return {struct_name: compute_energy(structure_futures[struct_name], y)
            for (struct_name, worker) in structure_futures.items()}

def dict_of_forces_tasks(structure_futures, y):
    return {struct_name: compute_forces(structure_futures[struct_name], y)
            for (struct_name, worker) in structure_futures.items()}

################################################################################

if __name__ == "__main__":

    print()
    print("WAIT - check the following before crying:")
    print("\t1) ASE lammpsrun.py has Prism(digits=16)")
    print("\t2) ASE lammpsrun.py using custom_printing")
    print("\t3) Comparing PER-ATOM values")
    print()

    start = time.time()
    main()

    print()
    print("Total runtime: {0}".format(time.time() - start))
