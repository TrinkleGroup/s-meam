import os
import sys
sys.path.insert(0, './')

import numpy as np

np.set_printoptions(precision=16, suppress=True)

seed = np.random.randint(0, high=(2**32 - 1))
#seed = int(sys.argv[1])
seed = 42
np.random.seed(seed)
print("Seed value: {0}\n".format(seed))

import time
import glob
import pickle
from prettytable import PrettyTable
from pprint import pprint

import parsl
from parsl import *
import multiprocessing as mp

import src.meam
from src.meam import MEAM
from src.optimization import force_matching

#parsl.set_file_logger('parsl.log')

################################################################################

def main():

    true_energies, true_forces = load_true_energies_and_forces()
    energy_weights = np.ones(len(true_energies)).tolist()

    structure_names = get_structure_list()

    sorted_true_energies = [true_energies[key] for key in structure_names]
    sorted_true_forces = [true_forces[key] for key in structure_names]

    print("True energies")
    print(np.vstack(sorted_true_energies))

    workers = load_workers(structure_names)

    pot = MEAM.from_file('data/fitting_databases/seed_42/seed_42.meam')
    _, y_pvec, _ = src.meam.splines_to_pvec(pot.splines)

    # eval_tasks = {key: compute_energy(w, y_pvec)
    #         for key,w in workers.items()}
    eng_tasks = [compute_energy(w, y_pvec) for w in workers]
    fcs_tasks = [compute_forces(w, y_pvec) for w in workers]

    eng_res = [job.result() for job in eng_tasks]
    fcs_res = [job.result() for job in fcs_tasks]

    eng_res = [energy[0] for energy in eng_res]
    fcs_res = [forces[0] for forces in fcs_res]

    print()
    print("Computed energies")
    print(np.vstack(eng_res))

    cost_function_evaluation = force_matching(fcs_res, sorted_true_forces,
            eng_res, sorted_true_energies, energy_weights)

    print()
    print("Cost function value:", cost_function_evaluation)

################################################################################

num_threads = mp.cpu_count() - 1
num_threads = 1
print("Requesting {0}/{1} threads".format(num_threads, mp.cpu_count()))
print()

config = {
    "sites": [
        {"site": "Local_IPP",
         "auth": {
             "channel": 'local',
         },
         "execution": {
             "executor": 'ipp',
             "provider": "local", # Run locally
             "block": {  # Definition of a block
                 "minBlocks" : 0, # }
                 "maxBlocks" : 1, # }<---- Shape of the blocks
                 "initBlocks": 1, # }
                 "taskBlocks": num_threads, # <--- No. of workers in a block
                 "parallelism" : 1 # <-- Parallelism
             }
         }
        }],
    "controller": {
        "publicIp": '128.174.228.50'  # <--- SPECIFY PUBLIC IP HERE
        }
}

dfk = DataFlowKernel(config=config, lazy_fail=False)

@App('python', dfk)
def compute_energy(w, parameter_vector):
    import sys
    return w.compute_energy(parameter_vector) / len(w.atoms)

@App('python', dfk)
def compute_forces(w, parameter_vector):
    import sys
    return w.compute_forces(parameter_vector) / len(w.atoms)

@App('python', dfk)
def do_force_matching(computed_forces, true_forces, computed_others=[],
        true_others=[], weights=[]):

    return force_matching(computed_forces, true_forces, computed_others,
            true_others, weights,)

################################################################################

def get_structure_list():
    full_paths = glob.glob("data/fitting_databases/seed_42/evaluator.*")
    file_names = [os.path.split(path)[-1] for path in full_paths]

    return [os.path.splitext(f_name)[-1][1:] for f_name in file_names]

def load_workers(all_names):
    path = "data/fitting_databases/seed_42/evaluator."

    return [pickle.load(open(path + name, 'rb')) for name in all_names]

def load_true_energies_and_forces():
    path = "data/fitting_databases/seed_42/info."

    true_energies = {}
    true_forces = {}

    for struct_name in glob.glob(path + '*'):
        f_name = os.path.split(struct_name)[-1]
        atoms_name = os.path.splitext(f_name)[-1][1:]

        # with open(struct_name, 'rb') as f:
        eng = np.genfromtxt(open(struct_name, 'rb'), max_rows=1)
        fcs = np.genfromtxt(open(struct_name, 'rb'), skip_header=1)

        true_energies[atoms_name] = eng
        true_forces[atoms_name] = fcs

    return true_energies, true_forces

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
