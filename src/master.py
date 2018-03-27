import os
import sys
import time
import numpy as np
from collections import namedtuple

np.random.seed(42)

import parsl
from parsl import *
from prettytable import PrettyTable

import src.meam
from tests.testStructs import allstructs
from tests.testPotentials import get_random_pots

# test_name1 = 'bulk_periodic_rhombo_mixed'
# test_name1 = 'bulk_vac_ortho_type1'
# test_name2 = '8_atoms'
# allstructs = {test_name1: allstructs[test_name1],
#              test_name2+'_v2': allstructs[test_name2],
#              test_name1+'_v3': allstructs[test_name1],
#              test_name1+'_v4': allstructs[test_name1],
#              test_name1+'_v5': allstructs[test_name1],}
#               test_name+'_v6': allstructs[test_name],
#               }

NUM_THREADS = 6

C_str = 'C'
R_str = '*'
P_str = '-'

config = {
    "sites": [
        {"site": "Local_IPP",
         "auth": {
             "channel": None,
         },
         "execution": {
             "executor": "ipp",
             "provider": "local", # Run locally
             "block": {  # Definition of a block
                 "minBlocks" : 1, # }
                 "maxBlocks" : 1, # }<---- Shape of the blocks
                 "initBlocks": 1, # }
                 "taskBlocks": NUM_THREADS, # <----- No. of workers in a block
                 "parallelism" : 1 # <-- Parallelism
             }
         }
        }]
}

################################################################################

#workers = ThreadPoolExecutor(max_workers=NUM_THREADS)
#dfk = DataFlowKernel(executors=[workers])

dfk = DataFlowKernel(config=config)

@App('python', dfk)
def build_worker(atoms, atoms_name, x_pvec, indices, types):
    import os
    import pickle

    from src.worker import Worker

    WORKER_SAVE_PATH = "../data/workers/"
    allow_reload = False

    file_name = WORKER_SAVE_PATH + atoms_name + '.pkl'

    if allow_reload and os.path.isfile(file_name):
        w = pickle.load(open(file_name, 'rb'))
    else:
        w = Worker(atoms, x_pvec, indices, types)
        #pickle.dump(w, open(file_name, 'wb'))

    return w

@App('python', dfk)
def compute_energy(w, y_pvec_):
    eng = w.compute_energies(y_pvec_)

    return eng

@App('python', dfk)
def compute_forces(w, y_pvec_):
    forces = w.compute_forces(y_pvec_)

    return forces

def get_status(all_jobs, dfk):

    data_c = []
    for row in all_jobs:
        row_results = []

        for task in row:
            state = P_str

            if task.parent:
                # print(task.parent._state)
                if task.parent.started:
                    if task.parent.completed: state = C_str
                    else: state = R_str

                # for v in running:
                #     if task.parent.engine_uuid == v: state = R_str
                # elif task.parent.engine_uuid in running: state = R_str
                # elif task.parent._state == 'RUNNING': state = R_str
            # print(task.running())
            # if task.running(): state = R_str
            # elif task.done(): state = C_str

            row_results.append(state)
        data_c.append(row_results)

    return np.array(data_c).T

def print_table(data_c):
    keys = list(allstructs.keys())

    table = PrettyTable(['struct', 'built', 'energy', 'forces'])
    for i in range(data_c.shape[0]):
        table.add_row([keys[i]] + data_c[i, :].tolist())

    for field_name in table.field_names:
        table.align[field_name] = 'r'

    print(table, end='\r\n')
    print()

def reset_cursor(n):

    for i in range(n):
        sys.stdout.write("\033[F")

def count_state(status, state):
    val, counts = np.unique(status, return_counts=True)

    if state in val:
        return int(counts[np.where(val == state)])
    else:
        return 0

@App('python', dfk)
def print_complete(all_calcs):
    print(flush=True)
    print("Total build time: {:.5f} (s)".format(time.time() - start),
          flush=True)

################################################################################

print("\nBeginning calculations ...\n", flush=True)
print(C_str + " := complete")
print(R_str + " := running")
print(P_str + " := in queue")
print()

potential = get_random_pots(1)['meams'][0]
x_pvec, y_pvec, indices = src.meam.splines_to_pvec(potential.splines)

start = time.time()

all_jobs = [[], [], []]

for name in allstructs.keys():
    atoms = allstructs[name]
    all_jobs[0].append(build_worker(atoms, name, x_pvec, indices,
                                    potential.types))

for worker in all_jobs[0]:
   all_jobs[1].append(compute_energy(worker, y_pvec))
   all_jobs[2].append(compute_forces(worker, y_pvec))

status =  get_status(all_jobs, dfk)
print_table(status)

executor = dfk.executors[list(dfk.executors.keys())[0]].executor

completed = count_state(status, C_str)
running = len(executor.outstanding)

# print("Currently using {0}/{1} threads".format(running, NUM_THREADS))
print("Completed {:d}/{:d} jobs in {:.0f} seconds".format(completed,
    status.size, time.time() - start), flush=True, end='\r')

while completed < status.size:
    status =  get_status(all_jobs, dfk)
    reset_cursor(len(all_jobs[0]) + 5)
    print_table(status)

    completed = count_state(status, C_str)
    running = len(executor.outstanding)

    # print("Currently using {0}/{1} threads".format(running, NUM_THREADS))
    print("Completed {:d}/{:d} jobs in {:.0f} seconds".format(completed,
        status.size, time.time() - start), flush=True, end='\r')

    time.sleep(1)

print()
# py_results = np.vstack([job.result() for job in all_jobs[1]])
# lammps_results = np.vstack([potential.get_lammps_results(allstructs[
#                      test_name])['energy'] for test_name in allstructs.keys()])
