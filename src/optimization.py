"""Library for computing cost functions and performing parameter optimization.

Authors: Josh Vita (UIUC), Dallas Trinkle (UIUC)"""

import os
import time
import pickle
import glob
import array
import numpy as np
np.random.seed(42)
np.set_printoptions(precision=16, suppress=True)

from scipy.optimize import fmin_cg, fmin_powell

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

    return h5py.File("data/fitting_databases/lj/workers.hdf5-copy", 'a',
            driver='core')

def get_structure_list():
    full_paths = glob.glob("data/fitting_databases/lj/evaluator.*")
    file_names = [os.path.split(path)[-1] for path in full_paths]

    return [os.path.splitext(f_name)[-1][1:] for f_name in file_names]

def load_true_energies_and_forces_dicts(worker_names):
    # TODO: true values should be attributes of the HDF5 db? unless u want text
    path = "data/fitting_databases/lj/info."

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

def build_pso_toolbox(pvec_len):

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    # creator.create("Individual", array.array, typecode='d',
    #         fitness=creator.CostFunctionMinimizer, velocity=array.array,
    #         best_pos=array.array, best_val=float)
    creator.create("Individual", np.ndarray, typecode='d',
            fitness=creator.CostFunctionMinimizer, velocity=0,
            best_pos=0, best_val=0)

    def ret_pvec(arr_fxn, rng):
        return arr_fxn(rng(pvec_len))

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual,
            np.random.random)

    toolbox.register("population", tools.initRepeat, list, toolbox.parameter_set,)

    w = c1 = c2 = 1.

    def move(ind, glob_best):
        v_new = w*ind.velocity + c1*(ind.best_pos - ind)*np.random.random() +\
            c2*(glob_best - ind)*np.random.random()

        ind += v_new

    toolbox.register("move_particle", move)

    return toolbox

def build_ga_toolbox(pvec_len):

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", array.array, typecode='d',
            fitness=creator.CostFunctionMinimizer)

    # TODO: figure out how to initialize individuals throughout parameter space

    def ret_pvec(arr_fxn, rng):
        return arr_fxn(rng(pvec_len))
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

    testing_database.close()

    true_value_dicts  = load_true_energies_and_forces_dicts(list(
        workers.keys()))
    true_forces, true_energies = true_value_dicts

    PVEC_LEN = workers[list(workers.keys())[0]].len_param_vec
    POP_SIZE = 20
    NUM_GENS = 100
    CXPB = 1.0
    MUTPB = 0.5

    print("POP_SIZE:", POP_SIZE)
    print("NUM_GENS:", NUM_GENS)
    print("CXPB:", CXPB)
    print("MUTPB:", MUTPB)

    # initialize toolbox with necessary functions for performing the GA 
    toolbox = build_ga_toolbox(PVEC_LEN)

    # for testing purposes, all individuals will be a perturbed version of the
    # true parameter set
    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/lj/lj.meam')
    _, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    indices.append(len(indices))


    ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            (-0.2, 0.4), (-2, 3), (-7.5, 12), (-7, 2), (-1, 0.2), (-1, 1)]

    def evaluate_pop(population):
        """Computes the energies and forces of the entire population at once.
        Note that forces and energies are ordered to match the workers list."""

        if len(population) > 1: pop_params = np.vstack(population)
        else: pop_params = np.array(population)

        for i in range(len(ranges)-1):
            a, b = ranges[i]
            i_lo, i_hi = indices[i], indices[i+1]

            pop_params[:, i_lo:i_hi] = pop_params[:, i_lo:i_hi]*(b-a) + a

        all_S = np.zeros(len(population)) # error function

        for name in workers.keys():
            w = workers[name]

            fcs_err = w.compute_forces(pop_params) / w.natoms - true_forces[name]
            eng_err = w.compute_energy(pop_params) / w.natoms - true_energies[name]

            fcs_err = np.linalg.norm(fcs_err, axis=(1,2))

            all_S += fcs_err*fcs_err*initial_weights[name]
            all_S += eng_err*eng_err*initial_weights[name]

        return all_S

    def gradient(x, h):
        """Computes the gradient on the fitness surface at point x in parameter
        space via the second order centered difference method

        Args:
            x (np.arr): a 1D vector of parameters
            h (float): step size for doing centered difference
            toolbox (DEAP.toolbox): for evaluating the fitness
            workers (dict): Worker objects indexed by structure name
            weights (dict): weights [0,1] for each structure
            forces (dict): force matrices for each structure
            energies (dict): energy values for each structure
        Returns:
            grad (np.arr): the gradient at point x
        """

        N = x.shape[-1]

        population = np.tile(x, (3,N)).reshape((3,N,N))

        diag1, diag2 = np.diag_indices_from(population[0,:,:])

        population[0, diag1, diag2] -= h
        population[2, diag1, diag2] += h

        population = population.reshape((3*N,N))

        fitnesses = evaluate_pop(population, workers, weights, forces, energies)

        fitnesses = fitnesses.reshape((3,N)).T
        fitnesses[:,1] *= -2

        return np.sum(fitnesses, axis=1) / h / h

    toolbox.register("evaluate_pop", evaluate_pop)
    toolbox.register("gradient", gradient)

    pop = toolbox.population(n=POP_SIZE)

    # record statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    fitnesses = toolbox.evaluate_pop(pop)

    for ind,val in zip(pop, fitnesses):
        ind.fitness.values = val,

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max", "avg", "std"

    record = stats.compile(pop)

    logbook.record(gen=0, **record)
    print(logbook.stream)

    run_new_ga = False
    if run_new_ga:
        # run GA
        i = 0
        while (i < NUM_GENS) and (len(pop) > 1):
            survivors = tools.selBest(pop, len(pop)//2)
            breeders = list(map(toolbox.clone, survivors))

            # mate
            j = 0
            while j < (POP_SIZE - len(breeders)):
                mom, dad = tools.selTournament(breeders, 2, 5, 'fitness')

                if CXPB >= np.random.random():
                    kid, _ = toolbox.mate(toolbox.clone(mom), toolbox.clone(dad))
                    del kid.fitness.values

                    survivors.append(kid)
                    j += 1

            survivors = tools.selBest(survivors, len(survivors))

            # mutate, preserving top 10
            for ind in survivors[10:]:
                if MUTPB >= np.random.random():
                    toolbox.mutate(ind)

            fitnesses = toolbox.evaluate_pop(survivors)

            for ind,val in zip(survivors, fitnesses):
                ind.fitness.values = val,

            pop = survivors

            record = stats.compile(pop)
            logbook.record(gen=i+1, **record)
            print(logbook.stream)

            i += 1

        top10 = tools.selBest(pop, 10)
        print([ind.fitness.values[0] for ind in top10])

        compare_to_true(top10[0], 'data/plots/ga_res-pre_cg')

        best_guess = top10[0]
    else:
        import src.meam
        from src.meam import MEAM

        pot = MEAM.from_file('data/fitting_databases/lj/lj.meam')
        _, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

        best_guess = y_pvec + np.random.normal(size=len(y_pvec), scale=1e-7)

        start = time.time()
        print("starting powell minimization")
        f = lambda x: toolbox.evaluate_pop(np.atleast_2d(x))
        cb = lambda x: print(f(x))
        old_fit = f(best_guess)
        print("old_fit:", old_fit)
        better_guess = fmin_powell(f, best_guess, maxiter=2000, callback=cb)
        print("improved fitness by", old_fit - f(better_guess))
        print("local minimization time:", time.time() - start)
       # best_guess = np.genfromtxt("data/ga_results/2018-05-17_300-300-1.0-0.5_ga.dat",
                # max_rows=1)

    # f = lambda pp: toolbox.evaluate_pop(pp, initial_weights)
    # f2 = lambda point: toolbox.gradient(point, )
    # print("Performing some steepest descent stuff")

    # steepest(best_guess, f, toolbox.gradient, h=1e-5, maxsteps=2000)
    # print("Monte Carlo")
    # better_guess = monte_carlo(best_guess, f, h=1e-3, T=0.01, maxsteps=2000)
    # better_guess = cg(best_guess, toolbox.evaluate_pop, toolbox.gradient, 1e-4,
    #         workers, initial_weights, true_forces, true_energies)
    # 
    # better_fitness = toolbox.evaluate_pop(better_guess, workers,
    #         initial_weights, true_forces, true_energies,)
    # 
    # print("Final fitness:", better_fitness[0])
    # 
    # compare_to_true(better_guess, 'data/plots/ga_res')

def pso():
    print("Particle swarm optimization ... ")

    # initialize testing and fitting (workers) databases
    testing_database = get_hdf5_testing_database(load=False)
    testing_db_size = len(testing_database)

    initial_weights = np.ones(testing_db_size) # chooses F = T
    initial_weights = {key:1 for key in testing_database.keys()}

    workers = load_workers_from_database(testing_database, initial_weights)

    testing_database.close()

    true_value_dicts  = load_true_energies_and_forces_dicts(list(
        workers.keys()))
    true_forces, true_energies = true_value_dicts

    PVEC_LEN = workers[list(workers.keys())[0]].len_param_vec
    POP_SIZE = 300
    NUM_STEPS = 1000

    print("POP_SIZE:", POP_SIZE)
    print("NUM_STEPS:", NUM_STEPS)

    toolbox = build_pso_toolbox(PVEC_LEN)

    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/lj/lj.meam')
    _, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    indices.append(len(indices))

    ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            (-0.2, 0.4), (-2, 3), (-7.5, 12), (-7, 2), (-1, 0.2), (-1, 1)]

    def evaluate_pop(population, weights):
        """Computes the energies and forces of the entire population at once.
        Note that forces and energies are ordered to match the workers list."""

        if len(population) > 1: pop_params = np.vstack(population)
        else: pop_params = np.array(population)

        for i in range(len(ranges)):
            a, b = ranges[i]
            i_lo, i_hi = indices[i], indices[i+1]

            pop_params[:, i_lo:i_hi] = pop_params[:, i_lo:i_hi]*(b-a) + a

        all_S = np.zeros(len(population)) # error function

        for name in workers.keys():
            w = workers[name]

            fcs_err = w.compute_forces(pop_params) / w.natoms - true_forces[name]
            eng_err = w.compute_energy(pop_params) / w.natoms - true_energies[name]

            fcs_err = np.linalg.norm(fcs_err, axis=(1,2))

            all_S += fcs_err*fcs_err*weights[name]
            all_S += eng_err*eng_err*weights[name]

        return all_S

    toolbox.register("evaluate_pop", evaluate_pop)

    # record statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max", "avg", "std"

    pop = toolbox.population(n=POP_SIZE)
    fitnesses = toolbox.evaluate_pop(pop, initial_weights)

    glob_best_idx = np.argmin(fitnesses)
    glob_best_pos = np.array(pop[glob_best_idx])
    glob_best_val = fitnesses[glob_best_idx]

    for ind,f in zip(pop, fitnesses):
        ind.fitness.values = f,
        ind.best_pos = np.array(ind)
        ind.best_val = f

        vel_params = np.random.random(PVEC_LEN)

        for i in range(len(ranges)):
            a, b = ranges[i]
            i_lo, i_hi = indices[i], indices[i+1]

            vel_params[i_lo:i_hi] = vel_params[i_lo:i_hi]*(b-a) + a

        ind.velocity = vel_params

    i = 0
    while i < NUM_STEPS:
        for ind in pop:
            toolbox.move_particle(ind, glob_best_pos)

        fitnesses = toolbox.evaluate_pop(pop, initial_weights)

        record = stats.compile(pop)

        logbook.record(gen=i+1, **record)
        print(logbook.stream)

        for ind,f in zip(pop, fitnesses):
            ind.fitness.values = f,

            if f < ind.best_val:
                ind.best_pos = np.array(ind)
                ind.best_val = f

            if f < glob_best_val:
                glob_best_pos = np.array(ind)
                glob_best_val = f

        i += 1

def monte_carlo(start, fxn, h=1e-3, T=1, maxsteps=10000, error_thresh=1e-7):
    """Performs a monte carlo simulation starting from x using a maximum of
    nsteps steps. Acceptance calculated using the typical Metropolis: rng <
    exp(-dU/T) where U is the cost function value

    Args:
        start (np.arr): 1D starting parameter vector
        fxn (callable): the function to compute values of
        h (float): standard deviation of normal distribution for step sizes
        T (float): "temperature" for computing acceptance probabilities
        maxsteps (int): maximum allowed number of steps
        error_thresh (float): the maximum error, under which the cost is 'good'

    Returns:
        new_x (np.arr): final parameter vector
    """

    N = len(start)

    old_x = np.array(start)
    old_U = fxn([old_x])

    stopping_criterion_met = False
    i = 0
    while (i < maxsteps) and not stopping_criterion_met:

        mutation_index = np.random.randint(N)

        new_x = old_x.copy()
        new_x[mutation_index] += np.random.normal(scale=h)

        new_U = fxn([new_x])
        dU = new_U - old_U

        if np.random.random() < np.exp(-min(dU,0.) / T):
            old_x = new_x
            old_U = new_U

            if new_U < error_thresh: stopping_criterion_met = True

        if i % 10 == 0: print(new_U)

        i += 1

    return new_x

def steepest(guess, fxn, grad, h, maxsteps):
    x = np.array(guess)

    i = 0
    while i < maxsteps:
        print(i, fxn([x]), flush=True)

        direction = grad([x])

        x = x + direction*h

def cg(guess, fxn, grad, h, workers, weights, forces, energies):

    f1 = lambda x: fxn(np.atleast_2d(x), workers, weights, forces, energies)
    f2 = lambda x: grad(np.atleast_2d(x), h, workers, weights, forces, energies)

    cb = lambda x: print("Fitness:", f1(x), flush=True)
    return fmin_cg(f1, guess, fprime=f2, callback=cb)

def compare_to_true(new_pvec, fname=''):
    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/lj/lj.meam')
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
    # pso()
    serial_ga_with_workers()
    # genetic_algorithm_example()
    # parsl_ga_example()
