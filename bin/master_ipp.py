import sys
sys.path.insert(0, './')

import os
import psutil
import time
import numpy as np
import multiprocessing as multi
import ipyparallel as ipp
from itertools import repeat
from pprint import pprint

from src.worker import Worker

seed = np.random.randint(0, high=(2**32 - 1))
#seed = int(sys.argv[1])
#seed = 42

np.random.seed(seed)
print("Seed value: {0}\n".format(seed))

print("Memory info: {0}".format(psutil.virtual_memory()))

################################################################################

def main():

    num_procs = 2
    #os.system("ipcluster start -n {0} &> /dev/null".format(num_procs))

    print("Requesting {0}/{1} processors".format(num_procs, multi.cpu_count()))
    print()

    client = ipp.Client()
    lbv = client.load_balanced_view()

    struct_names, structs = get_structure_list()

    print("Structures:")
    pprint(struct_names)
    print()

    knot_positions,spline_start_indices,atom_types = get_potential_information()
    num_splines = len(spline_start_indices)

    print("Adding jobs")
    print(sys.path)
    from src.worker import Worker
    initialized_structures = lbv.map_sync(
        lambda a, b, c, d: Worker(a, b, c, d),
        structs, knot_positions, spline_start_indices, atom_types,
        )

    print(initialized_structures)

    # evaluate energies/forces using a random vector of spline parameters
    #spline_parameters = np.random.random(
    #        knot_positions.shape[0] + 2*num_splines)

    #calculated_energies = pool.starmap(
    #        compute_energy,
    #        zip(initialized_structures, repeat(spline_parameters))
    #        )

    #calculated_forces = pool.starmap(
    #        compute_forces,
    #        zip(initialized_structures, repeat(spline_parameters))
    #        )


    #print(calculated_energies)
    #print(calculated_forces)

################################################################################

def get_structure_list():
    from tests.testStructs import allstructs

    test_name = '8_atoms'                                                        
    tmp = [allstructs[test_name]]*int(sys.argv[1])
    dct = {test_name+'_v{0}'.format(i+1):tmp[i] for i in range(len(tmp))} 

    #dct = allstructs

    return list(dct.keys()), list(dct.values())

def get_potential_information():
    import src.meam
    from tests.testPotentials import get_random_pots

    potential = get_random_pots(1)['meams'][0]
    x_pvec, _, indices = src.meam.splines_to_pvec(potential.splines)

    return x_pvec, indices, potential.types

def compute_energy(w, y):
    return w.compute_energy(y)

def compute_forces(w, y):
    return w.compute_forces(y)

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
