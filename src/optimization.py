"""Library for computing cost functions and performing parameter optimization.

Authors: Josh Vita (UIUC), Dallas Trinkle (UIUC)"""

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

################################################################################

def load_workers_from_database(hdf5_file, weights):
    workers = {}

    for name in hdf5_file.keys():
        if weights[name] > 0:
            workers[name] = Worker.from_hdf5(hdf5_file, name)

    return workers

def get_hdf5_testing_database(load=False):
    """Builds or loads a new HDF5 database of workers.

    Args:
        load (bool): True if database already exists; default is False
    """

    return h5py.File("data/fitting_databases/seed_42/workers.hdf5-copy", 'a',
            driver='core')

def get_structure_list():
    full_paths = glob.glob("data/fitting_databases/seed_42/evaluator.*")
    file_names = [os.path.split(path)[-1] for path in full_paths]

    return [os.path.splitext(f_name)[-1][1:] for f_name in file_names]

def load_workers(all_names):
    path = "data/fitting_databases/seed_42/evaluator."

    return [pickle.load(open(path + name, 'rb')) for name in all_names]

def load_true_energies_and_forces_dicts(worker_names):
    # TODO: true values should be attributes of the HDF5 db? unless u want text
    path = "data/fitting_databases/seed_42/info."

    true_forces = {}
    true_energies = {}
    # properties = []

    # for struct_name in glob.glob(path + '*'):
    for struct_name in worker_names:
        # f_name = os.path.split(struct_name)[-1]
        # atoms_name = os.path.splitext(f_name)[-1][1:]

        fcs = np.genfromtxt(open(path+struct_name, 'rb'), skip_header=1)
        eng = np.genfromtxt(open(path+struct_name, 'rb'), max_rows=1)

        true_forces[struct_name] = fcs
        true_energies[struct_name] = eng
        # properties.append(fcs)
        # properties.append(eng)

    return true_forces, true_energies
    # return properties

def build_ga_toolbox(pvec_len):

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", array.array, typecode='d',
            fitness=creator.CostFunctionMinimizer)

    # TODO: figure out how to initialize individuals throughout parameter space

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
    # def ret_pvec(arr_fxn, rng):
        # return arr_fxn(y_pvec + np.ones(y_pvec.shape)*1*rng())

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual,
            np.random.random)

    toolbox.register("population", tools.initRepeat, list, toolbox.parameter_set,)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("select", tools.selBest)

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

def serial_ga_with_workers():
    """A serial genetic algorithm using actual Worker objects; intended for
    educational purposes before moving on to parallel version."""

    # initialize testing and fitting (workers) databases
    testing_database = get_hdf5_testing_database(load=False)
    testing_db_size = len(testing_database)

    initial_weights = np.ones(testing_db_size) # chooses F = T
    initial_weights = {key:1 for key in testing_database.keys()}

    workers = load_workers_from_database(testing_database, initial_weights)

    true_value_dicts  = load_true_energies_and_forces_dicts(list(
        workers.keys()))
    true_forces, true_energies = true_value_dicts

    PVEC_LEN = workers[list(workers.keys())[0]].len_param_vec
    POP_SIZE = 300
    NUM_GENS = 200
    CXPB = 1.0
    MUTPB = 0.5

    print("POP_SIZE:", POP_SIZE)
    print("NUM_GENS:", NUM_GENS)
    print("CXPB:", CXPB)
    print("MUTPB:", MUTPB)

    # initialize toolbox with necessary functions for performing the GA 
    toolbox = build_ga_toolbox(PVEC_LEN)

    pop = toolbox.population(n=POP_SIZE)

    # record statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    fitnesses = toolbox.evaluate_pop(
            pop, workers, initial_weights, true_forces, true_energies)
    # print(fitnesses)

    for ind,val in zip(pop, fitnesses):
        ind.fitness.values = val,

    # print("{:10}{:10}{:10}{:10}".format("max", "min", "avg", "std"))

    logbook = tools.Logbook()
    logbook.header = "min", "max", "avg", "std"

    record = stats.compile(pop)

    logbook.record(**record)
    print(logbook.stream)

    # run GA
    i = 0
    while (i < NUM_GENS) and (len(pop) > 1):
        survivors = tools.selBest(pop, len(pop)//2)
        # best = list(map(toolbox.clone, survivors[:10]))
        # survivors = [ind for ind in pop if ind.fitness.values[0] < 5000.]
        # print(len(survivors))
        # best = survivors[0]
        breeders = list(map(toolbox.clone, survivors))

        # mate
        # for child1, child2 in zip(breeders[::2], breeders[1::2]):
        #     if CXPB > np.random.random():
        #         toolbox.mate(child1, child2)
        #         del child1.fitness.values
        #         del child2.fitness.values

        j = 0
        while j < (POP_SIZE - len(breeders)):
            mom, dad = tools.selTournament(breeders, 2, 5, 'fitness')

            if CXPB >= np.random.random():
                kid, _ = toolbox.mate(toolbox.clone(mom), toolbox.clone(dad))
                del kid.fitness.values

                survivors.append(kid)
                j += 1

        # evaluate fitnesses for new generation
        # invalid_ind = [ind for ind in breeders if not ind.fitness.valid]
        # invalid_ind = [ind for ind in survivors if not ind.fitness.valid]

        # fitnesses = toolbox.evaluate_pop(invalid_ind,
        #         workers, initial_weights, true_forces, true_energies)
        # 
        # for ind,val in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = val,
        # 
        # pop = survivors + breeders

        survivors = tools.selBest(survivors, len(survivors))
        # mutate, preserving top 10
        for ind in survivors[10:]:
            if MUTPB >= np.random.random():
                toolbox.mutate(ind)

        fitnesses = toolbox.evaluate_pop(survivors,
                workers, initial_weights, true_forces, true_energies)

        for ind,val in zip(survivors, fitnesses):
            ind.fitness.values = val,

        pop = survivors

        record = stats.compile(pop)
        logbook.record(**record)
        print(logbook.stream)

        i += 1

    top10 = tools.selBest(pop, 10)
    print([ind.fitness.values[0] for ind in top10])

    compare_to_true(top10[0], 'data/plots/ga_res-pre_cg')

    # best_guess = np.atleast_2d(top10[0])
    # better_guess = cg(best_guess, toolbox.evaluate_pop, workers,
    #     initial_weights, true_forces, true_energies)
    # 
    # better_fitness = toolbox.evaluate_pop(better_guess, workers,
    #         initial_weights, true_forces, true_energies,)
    # 
    # print("Final fitness:", better_fitness[0])
    # 
    # compare_to_true(better_guess, 'data/plots/ga_res')

def cg(guess, fxn, *args):
    from scipy.optimize import fmin_cg

    org_shape = guess.shape
    f2 = lambda x: fxn(x.reshape(org_shape), *args)
    f3 = lambda x: print(f2(x))
    return fmin_cg(f2, guess, callback=f3)

def compare_to_true(new_pvec, fname=''):
    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/seed_42/seed_42.meam')
    x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    new_pot = MEAM.from_pvec(x_pvec, new_pvec, indices, ['H', 'He'])

    splines1 = pot.splines
    splines2 = new_pot.splines

    import matplotlib.pyplot as plt

    for i,(s1,s2) in enumerate(zip(splines1, splines2)):

        low,high = s1.cutoff
        low -= abs(0.2*low)
        high += abs(0.2*high)

        x = np.linspace(low,high,1000)
        y1 = list(map(lambda e: s1(e) if s1.in_range(e) else s1.extrap(e), x))
        y2 = list(map(lambda e: s2(e) if s2.in_range(e) else s2.extrap(e), x))

        yi = list(map(lambda e: s1(e), s1.x))

        plt.figure()
        plt.plot(s1.x, yi, 'o')
        plt.plot(x, y1, 'b', label='true')
        plt.plot(x, y2, 'r', label='new')
        plt.legend()

        plt.savefig(fname + str(i+1) + ".png")

def parsl_ga_example():
    """A simple genetic algorithm using the DEAP library"""

    from deap import base, creator, tools, algorithms

    # define base classes for the fitness evaluator and the individual class
    creator.create("FitnessMin", base.Fitness, weights=(-1., 1.))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    SIZE = 10

    # register a function 'individual' that constructs an individual
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.random)
    toolbox.register("individual",
            tools.initRepeat, creator.Individual, toolbox.attr_float, n=SIZE)

    # register a function to build populations of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=100)

    # evaluate individuals
    def evaluate(individual):
        l = np.array(individual)

        a = par_sum(l).result()
        b = par_len(l).result()

        return a, 1./b

    # add the above tools to the toolbox; condenses the algorithm into the box
    toolbox.register("evaluate", evaluate)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selBest)

    # record statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
            ngen=50, stats=stats, verbose=True,)

num_threads = 4

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
def par_sum(nums):
    return sum(nums)

@App('python', dfk)
def par_len(nums):
    return len(nums)

def genetic_algorithm_example():
    """A simple genetic algorithm using the DEAP library"""

    from deap import base, creator, tools, algorithms

    # define base classes for the fitness evaluator and the individual class
    creator.create("FitnessMin", base.Fitness, weights=(-1., 1.))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    SIZE = 10

    # register a function 'individual' that constructs an individual
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.random)
    toolbox.register("individual",
            tools.initRepeat, creator.Individual, toolbox.attr_float, n=SIZE)

    indiv1 = toolbox.individual()
    print("One individual:\n{}".format(indiv1))

    # register a function to build populations of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=100)
    print("A population of 5 individuals:\n{}".format(pop))

    # evaluate individuals
    def evaluate(individual):
        a = sum(individual)
        b = len(individual)

        return a, 1./b

    for indiv in pop:
        indiv.fitness.values = evaluate(indiv)

    print("Evaluation of population:\n{}".format([ind.fitness for ind in pop]))

    # perform mutations; indiv1 is untouched; mutant and indiv2 are identical
    mutant = toolbox.clone(indiv1)
    indiv2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
    del mutant.fitness.values # delete old values copied from indiv1

    print(indiv1)
    print(mutant)
    print("indiv2 == mutant:", indiv2 is mutant)

    # blend two individuals; again, copy beforehand
    child1 = toolbox.clone(pop[0])
    child2 = toolbox.clone(pop[1])
    print("Before blend:\n", child1, child2)

    tools.cxBlend(child1, child2, 0.5)

    print("After blend:\n", child1, child2)

    # select the 2 best individuals using their fitness values
    selected = tools.selBest(pop, 2)
    print("Top 2 fitnesses:", [ind.fitness.values for ind in selected])

    # add the above tools to the toolbox; condenses the algorithm into the box
    toolbox.register("evaluate", evaluate)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selBest)

    val = evaluate(pop[0])
    new1, new2 = toolbox.mate(child1, child2)
    mutant2 = toolbox.mutate(mutant)
    top2 = toolbox.select(pop, 2)

    # record statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
            ngen=50, stats=stats, verbose=False,)

    print(logbook)

# def mcmc(start):
#     STEP_SIZE = 0.01
#     MAX_STEPS = 10000
#
#     prev = start.copy()
#
#     i = 0
#     while i < MAX_STEPS:
#         rand_index = int(np.random.randint(len(guess), size=1))
#
#         attempt = prev[i] += STEP_SIZE
#
#         i += 1

def force_matching(computed_forces, true_forces, computed_others=[],
        true_others=[], weights=[]):
    """Computes the cost function using the force-matching method, as outlined
    in the original paper by Ercolessi. Notation matches original paper.

    Args:
        computed_forces (list): computed force matrices
        true_forces (list): true force matrices
        computed_others(list): other constraints (e.g. energies, lattice
            constants, etc)
        true_others (list): true values for other constraints; order matches
            computed
        weights (list): weights on additional constraints

    Returns:
        Z (float): the evaluation of the cost function
    """

    # TODO: this function needs to be profiled and optimized
    # TODO: vectorize for multiple paramsets at once

    if computed_others:
        if not true_others: raise ValueError("must specify true values")
        elif not weights: raise ValueError("must specify weights")

    Zf = sum([np.sum((comp - true)**2)
                for comp,true in zip(computed_forces, true_forces)])

    Zf /= 3*sum([force_mat.shape[0] for force_mat in computed_forces])

    Zc = 0
    if computed_others:
        Zc = sum([w*((comp - true)**2)
                for w,comp,true in zip(weights, computed_others, true_others)])

    return Zf + Zc

def error_function(computed, true):
    """The UN-weighted squared error between the computed and true values for each
    struct. S(theta, F) from the original paper.

    Args:
        computed (list): all computed properties (forces, energies, etc.)
        true (list): the true values of the properties

    Returns:
        val (float): the weighted squared error of all fitting properties

    Note:
        Energies (computed and true) are expected to already have their
        reference structure value subtrated off, if one is being used
    """

    return sum([np.sum((np.array(a)-np.array(b))**2)
                for (a,b) in zip(computed, true)])

def likelihood_function(S_mle, S_theta):
    """Computes the value of the likelihood function. L(F, theta) from the
    original paper

    Args:
        S_mle (float): the value of the error function using theta_MLE
        S_theta (float): the value of the error function using a new theta

    Returns:
        val (float): the value of the likelihood function
    """

    return np.exp(-S_theta / S_mle)

def objective_function():
    """TODO: the objective function of the fitting database"""

if __name__ == "__main__":
    serial_ga_with_workers()
    # genetic_algorithm_example()
    # parsl_ga_example()
