import sys
sys.path.insert(0, './')

import numpy as np
seed = 42
np.random.seed(seed)
print("Seed value: {0}\n".format(seed))

from pympler import muppy, summary

import os
import psutil
import time
import glob
import ctypes
import pickle
import multiprocessing as multi
from itertools import repeat
from pprint import pprint

import src.lammpsTools
from src.worker import Worker

################################################################################

def nodes_should_get_initialized_with_this_potential_info():
    tmp_tup = get_potential_information()
    knot_positions, spline_start_indices, atom_types = tmp_tup

    return knot_positions, spline_start_indices, atom_types

def nodes_should_get_initialized_with_this_structure_info():
    return get_structure_list()

def nodes_should_get_PASSED_this_info():
    return spline_parameters

def evaluate_struct(i, evaluators, params, results):
    results[i] = evaluators[i].compute_energy(params)

################################################################################

def get_structure_list():
    from tests.testStructs import allstructs

    #test_name = '8_atoms'                                                        

    num_copies = int(sys.argv[1]) if (len(sys.argv) > 1) else 1
    #tmp = [allstructs[test_name]]*num_copies
    #dct = {test_name+'_v{0}'.format(i+1):tmp[i] for i in range(len(tmp))} 

    #dct = allstructs

    #return list(dct.keys())#, list(dct.values())
    val = ["data/fitting_databases/seed_42/evaluator.aaa"]*num_copies
    val = ["data/fitting_databases/seed_42/evaluator.bulk_periodic_rhombo_type2"]*num_copies
    val = glob.glob("data/fitting_databases/seed_42/evaluator.*")[:4]

    return val

def get_potential_information():
    import src.meam
    from tests.testPotentials import get_random_pots

    potential = get_random_pots(1)['meams'][0]
    x_pvec, _, indices = src.meam.splines_to_pvec(potential.splines)

    return x_pvec, indices, potential.types

################################################################################

if __name__ == "__main__":
    start = time.time()

    print()
    print("WAIT - check the following before crying:")
    print("\t1) ASE lammpsrun.py has Prism(digits=16)")
    print("\t2) ASE lammpsrun.py using custom_printing")
    print("\t3) Comparing PER-ATOM values")
    print()

    # load potential info (should be passed in)
    knots, knot_idxs, types = nodes_should_get_initialized_with_this_potential_info()
    num_splines = len(knot_idxs)

    # load structure info (should also be passed in)
    struct_names = nodes_should_get_initialized_with_this_structure_info()

    print("Structures:")
    pprint(struct_names)
    print()

    # load parameter vector info (should ALSO get passed in)
    parameter_array = np.random.random(knots.shape[0] + 2*num_splines)
    parameter_array = np.vstack([parameter_array]*1000)

    # build evaluators
    workers = [pickle.load(open(path, 'rb')) for path in struct_names]

    num_avail_procs = multi.cpu_count()

    if len(workers) > num_avail_procs:
        raise ValueError("num_workers should not exceed num_avail_processors")

    #spawner = multi.get_context('spawn')

    manager = multi.Manager()

    # create shared versions of necessary variables TODO: del old refs?
    s_parameters = manager.list(parameter_array)
    s_workers = manager.list(workers)

    del knots
    del knot_idxs
    del parameter_array
    del struct_names
    del types
    del workers

    s_results = manager.list([None for i in range(len(s_workers))])

    num_procs = 4

    print("Requesting {0}/{1} processors".format(num_procs, multi.cpu_count()))
    print()

    pool = [multi.Process(target=evaluate_struct, args=(i, s_workers,
    s_parameters, s_results))
        for i in range(len(s_workers))]

    for p in pool: p.start()

    computed_energies = [p.join() for p in pool]

    #np.set_printoptions(precision=10, suppress=True)
    #print()
    #print("Computed energies:\n{}".format(np.vstack(s_results)))

    print()
    print("Total runtime: {0}".format(time.time() - start))
