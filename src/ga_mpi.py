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
FITNESSES = 2

# GA settings
POP_SIZE = 200
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

        # group structures to be passed to sub-processes
        tup = group_for_mpi_scatter(workers, database_weights, true_forces,
                true_energies, size)

        workers = tup[0]
        database_weights = tup[1]
        true_forces = tup[2]
        true_energies = tup[3]
    else:
        workers = None
        database_weights = None
        true_forces = None
        true_energies = None

    workers = comm.scatter(workers, root=0)
    database_weights = comm.scatter(database_weights, root=0)
    true_forces = comm.scatter(true_forces, root=0)
    true_energies = comm.scatter(true_energies, root=0)

    print("rank", rank, "has", len(workers), "structures", flush=True)

    # have every process build the toolbox
    pvec_len = workers[list(workers.keys())[0]].len_param_vec
    toolbox = build_ga_toolbox(pvec_len)

    # set up statistics recorder
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max", "avg", "std"

    # compute initial fitnesses
    if rank == MASTER:
        pop = toolbox.population(n=POP_SIZE)
    else:
        pop = None

    pop = comm.bcast(pop, root=0)

    fitnesses = toolbox.evaluate_pop(
            pop, workers, database_weights, true_forces, true_energies)

    all_fitnesses = comm.gather(fitnesses, root=0)

    if rank == MASTER:
        for ind,fit in zip(pop, np.sum(all_fitnesses, axis=0)):
            ind.fitness.values = fit,

        record = stats.compile(pop)
        logbook.record(gen=0, **record)
        print(logbook.stream, flush=True)

    i = 0
    while (i < NUM_GENS):
        if rank == MASTER:
            survivors = tools.selBest(pop, len(pop)//2)
            breeders = list(map(toolbox.clone, survivors))

            j = 0
            while j < (POP_SIZE - len(breeders)):
                mom, dad = tools.selTournament(breeders, 2, 5, 'fitness')

                if CXPB >= np.random.random():
                    kid, _ = toolbox.mate(toolbox.clone(mom), toolbox.clone(dad))
                    del kid.fitness.values

                    survivors.append(kid)
                    j += 1

            survivors = tools.selBest(survivors, len(survivors))

            for ind in survivors[10:]:
                if MUTPB >= np.random.random():
                    toolbox.mutate(ind)
        else:
            survivors = None

        survivors = comm.bcast(survivors, root=0)

        fitnesses = toolbox.evaluate_pop(survivors,
                workers, database_weights, true_forces, true_energies)

        all_fitnesses = comm.gather(fitnesses, root=0)

        if rank == MASTER:
            for ind,fit in zip(survivors, np.sum(all_fitnesses, axis=0)):
                ind.fitness.values = fit,

            pop = survivors

            record = stats.compile(pop)
            logbook.record(gen=i+1, **record)
            print(logbook.stream, flush=True)

        i += 1

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

def group_for_mpi_scatter(workers,database_weights, true_forces, true_energies,
        size):
    grouped_keys = np.array_split(list(workers.keys()), size)

    grouped_workers = []
    grouped_weights = []
    grouped_forces = []
    grouped_energies = []

    for group in grouped_keys:
        tmp = {}
        tmp2 = {}
        tmp3 = {}
        tmp4 = {}

        for name in group:
            tmp[name] = workers[name]
            tmp2[name] = database_weights[name]
            tmp3[name] = true_forces[name]
            tmp4[name] = true_energies[name]

        grouped_workers.append(tmp)
        grouped_weights.append(tmp2)
        grouped_forces.append(tmp3)
        grouped_energies.append(tmp4)

    return grouped_workers, grouped_weights, grouped_forces, grouped_energies

################################################################################

if __name__ == "__main__":
    main()
