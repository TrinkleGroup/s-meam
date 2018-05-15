import os
import pickle
import glob
import array
import numpy as np
np.random.seed(42)
np.set_printoptions(precision=16, suppress=True)

import h5py
from parsl import *
from deap import base, creator, tools, algorithms

from src.worker import Worker

os.chdir("/home/jvita/scripts/s-meam/project/")

from mpi4py import MPI

MASTER = 0 # rank of master node

# MPI tags
TOOLBOX_INIT = 1

# GA settings
POP_SIZE = 2
NUM_GENS = 100
CXPB = 1.0
MUTPB = 0.5


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == MASTER:
        # initialize testing and fitting (workers) databases
        testing_database = get_hdf5_testing_database(load=False)

        database_weights = {key:1 for key in testing_database.keys()}

        workers = load_workers_from_database(testing_database, database_weights)

        testing_database.close()

        true_value_dicts  = load_true_energies_and_forces_dicts(list(
            workers.keys()))
        true_forces, true_energies = true_value_dicts
    else:
        workers = None
        database_weights = None
        true_forces = None
        true_energies = None

    workers = comm.bcast(workers, root=0)
    database_weights = comm.bcast(database_weights, root=0)
    true_forces = comm.bcast(true_forces, root=0)
    true_energies = comm.bcast(true_energies, root=0)

    # initialize toolbox with necessary functions for performing the GA 
    PVEC_LEN = workers[list(workers.keys())[0]].len_param_vec
    toolbox = build_ga_toolbox(PVEC_LEN)

    # set up statistics recorder
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "min", "max", "avg", "std"

    # for slave_num in range(1, NUM_WORKERS):
    #     comm.send([toolbox, stats, logbook], dest=slave_num,
    #             tag=TOOLBOX_INIT)
    # else:
    #     toolbox = stats = logbook = None

    # all workers should now have the tools necessary to run the GA
    # comm.bcast([toolbox, stats, logbook], root=0)

    if rank == MASTER:
        pop = toolbox.population(n=POP_SIZE)
    else:
        pop = None

    pop = comm.scatter(pop, root=0)

    print("rank", rank, "has", len(pop), "individuals")

################################################################################

def load_workers_from_database(hdf5_file, weights):
    """Builds a dictionary of worker objects (index by structure name) using
    values stored in an HDF5 file

    Args:
        hdf5_file (h5py.File): an open file object
        weights (list): a weighting list; assumed to match order of sorted keys
    """

    workers = {}

    for name in hdf5_file.keys():
        if weights[name] > 0:
            workers[name] = Worker.from_hdf5(hdf5_file, name)

    return workers

def get_hdf5_testing_database(load=False):
    """Loads HDF5 database of workers using the h5py 'core' driver (in memory)

    Args:
        load (bool): True if database already exists; default is False
    """

    return h5py.File("data/fitting_databases/seed_42/workers.hdf5-copy", 'a',
            driver='core')

def load_true_energies_and_forces_dicts(worker_names):
    """Loads true forces and energies from formatted text files"""
    path = "data/fitting_databases/seed_42/info."

    true_forces = {}
    true_energies = {}

    for struct_name in worker_names:

        fcs = np.genfromtxt(open(path+struct_name, 'rb'), skip_header=1)
        eng = np.genfromtxt(open(path+struct_name, 'rb'), max_rows=1)

        true_forces[struct_name] = fcs
        true_energies[struct_name] = eng

    return true_forces, true_energies

def build_ga_toolbox(pvec_len):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", array.array, typecode='d',
            fitness=creator.CostFunctionMinimizer)

    # for testing purposes, all individuals will be a perturbed version of the
    # true parameter set
    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/seed_42/seed_42.meam')
    _, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    indices.append(len(indices))
    ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            (-0.2, 0.4), (-2, 3), (-7.5, 12), (-7, 2), (-1, 0.2), (-1, 1)]

    def ret_pvec(arr_fxn, rng):
        return arr_fxn(rng(len(y_pvec)))

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual,
            np.random.random)

    toolbox.register("population", tools.initRepeat, list, toolbox.parameter_set,)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def evaluate_pop(population, workers, weights, true_f, true_e):
        """Computes the energies and forces of the entire population at once.
        Note that forces and energies are ordered to match the workers list."""

        pop_params = np.vstack(population)

        for i in range(len(ranges)):
            a, b = ranges[i]
            i_lo, i_hi = indices[i], indices[i+1]

            pop_params[:, i_lo:i_hi] = pop_params[:, i_lo:i_hi]*(b-a) + a

        all_S = np.zeros(len(population)) # error function

        for name in workers.keys():
            w = workers[name]

            fcs_err = w.compute_forces(pop_params) / w.natoms - true_f[name]
            eng_err = w.compute_energy(pop_params) / w.natoms - true_e[name]

            fcs_err = np.linalg.norm(fcs_err, axis=(1,2))

            all_S += fcs_err*fcs_err*weights[name]
            all_S += eng_err*eng_err*weights[name]

        return all_S

    toolbox.register("evaluate_pop", evaluate_pop)

    return toolbox

################################################################################

if __name__ == "__main__":
    main()
