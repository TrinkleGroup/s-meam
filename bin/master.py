import sys
sys.path.insert(0, './')

import os
import psutil
import time
import glob
import ctypes
import numpy as np
import multiprocessing as multi
from itertools import repeat
from pprint import pprint

import src.lammpsTools
from src.worker import Worker

################################################################################

def nodes_should_get_initialized_with_this_potential_info():
    # initialize settings for interatomic potential
    tmp_tup = get_potential_information()
    knot_positions, spline_start_indices, atom_types = tmp_tup

    return knot_positions, spline_start_indices, atom_types

def nodes_should_get_initialized_with_this_structure_info():
    # get desired structure names (file paths)
    all_names = get_structure_list()

    return all_names

def nodes_should_get_PASSED_this_info():
    # evaluate energies/forces using a random vector of spline parameters

    
    return spline_parameters

################################################################################

def get_structure_list():
    from tests.testStructs import allstructs

    #test_name = '8_atoms'                                                        

    num_copies = int(sys.argv[1]) if (len(sys.argv) > 1) else 1
    #tmp = [allstructs[test_name]]*num_copies
    #dct = {test_name+'_v{0}'.format(i+1):tmp[i] for i in range(len(tmp))} 

    #dct = allstructs

    #return list(dct.keys())#, list(dct.values())
    val = ["data/test_db/data.bulk_periodic_rhombo_type2"]*num_copies
    val = ["data/test_db/data.aaa"]*num_copies
    val = glob.glob("data/test_db/data.*")

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

    seed = 42
    np.random.seed(seed)
    print("Seed value: {0}\n".format(seed))

    # load potential info (should be passed in)
    knots, knot_idxs, types = nodes_should_get_initialized_with_this_potential_info()
    #num_params = 2*(len(knots) + 2)
    num_splines = len(knot_idxs)

    # load structure info (should also be passed in)
    struct_names = nodes_should_get_initialized_with_this_structure_info()

    print("Structures:")
    pprint(struct_names)
    print()

    # load parameter vector info (should ALSO get passed in)
    parameter_array = np.random.random(knots.shape[0] + 2*num_splines)
    parameter_array = np.atleast_2d(parameter_array)

    # prepare parallel stuff
    num_procs = 7
    print("Requesting {0}/{1} processors".format(num_procs, multi.cpu_count()))
    print()

    #spawner = multi.get_context('spawn')

    manager = multi.Manager()

    # create shared versions of necessary variables TODO: del old refs?
    #s_knots = multi.Array(ctypes.c_double, knots)
    #s_idxs = multi.Array(ctypes.c_double, knot_idxs)
    #s_parameters = multi.Array(ctypes.c_double, parameter_array.ravel())
    #s_names = manager.list(struct_names)
    #s_types = manager.list(types)

    #del knots
    #del knot_idxs
    #del parameter_array
    #del struct_names
    #del types

    # build evaluators

    initialized_structures = manager.list()

    def build_worker(name):
        atoms = src.lammpsTools.atoms_from_file(name, types)

        return Worker(atoms, knots, knot_idxs, types)

    def evaluate_structs(evaluator):
        return evaluator.compute_energy(parameter_array)

    pool = multi.Pool(num_procs)

    initialized_evaluators = pool.map(build_worker, struct_names)
    s_evaluators = manager.list(initialized_evaluators)

    del initialized_evaluators

    for i in range(3):
        computed_energies = pool.map(evaluate_structs, s_evaluators)
        print(np.concatenate(computed_energies))

        parameter_array = np.random.random(knots.shape[0] + 2*num_splines)
        parameter_array = np.atleast_2d(parameter_array)

    print()
    print("Total runtime: {0}".format(time.time() - start))
