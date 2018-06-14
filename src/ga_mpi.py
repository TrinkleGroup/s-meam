import os
os.chdir("/home/jvita/scripts/s-meam/project/")

import numpy as np
import random
np.set_printoptions(precision=16, linewidth=np.inf, suppress=True)
np.random.seed(42)
random.seed(42)

import pickle
import glob
import array
import h5py
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import fmin_powell
from scipy.interpolate import CubicSpline
from mpi4py import MPI

from deap import base, creator, tools, algorithms

import src.meam
from src.worker import Worker
from src.meam import MEAM
from src.spline import Spline


################################################################################
"""MPI settings"""

MASTER_RANK = 0

################################################################################
"""MEAM potential settings"""

# ACTIVE_SPLINES = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ACTIVE_SPLINES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

################################################################################
"""GA settings"""

POP_SIZE = 50
NUM_GENS = 2000
CXPB = 1.0
MUTPB = 0.5

RUN_NEW_GA = True
DO_POWELL = False # ALWAYS does initial/final powell; True means pow every step

NUM_POWELL_STEPS = 5

CHECKPOINT_FREQUENCY = 10

################################################################################ 
"""I/O settings"""

CHECK_BEFORE_OVERWRITE = True

LOAD_PATH = "data/fitting_databases/pinchao-rhophi/"
SAVE_PATH = "data/ga_results/"
SETTINGS_STR = "{}-{}-{}-{}".format(POP_SIZE, NUM_GENS, CXPB, MUTPB)

DB_FILE_NAME = LOAD_PATH + 'structures.hdf5'
DB_INFO_FILE_NAME = LOAD_PATH + '/info'
POP_FILE_NAME = SAVE_PATH + SETTINGS_STR + "/pop.dat"
LOG_FILE_NAME = SAVE_PATH + SETTINGS_STR + "/ga.log"
TRACE_FILE_NAME = SAVE_PATH + SETTINGS_STR + "/trace.dat"

################################################################################ 

def main():
    # Record MPI settings
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    is_master_node = (rank == MASTER_RANK)

    if is_master_node:

        # Initialize database and variables to prepare for GA
        print_settings()

        print("MASTER: Preparing save directory/files ... ", flush=True)
        prepare_save_directory()

        stats, logbook = build_stats_and_log()

        # print("MASTER: Loading structures ...", flush=True)
        # structures, weights = load_structures_on_master()
        print("MASTER: assigning structures to processors ... ", flush=True)
        struct_names = np.array_split(np.array(get_all_struct_names()), mpi_size)

        # print("MASTER: Loading energy/forces database ... ", flush=True)
        # true_forces, true_energies = load_true_values(structures.keys())

        # print("MASTER: Determining potential information ...", flush=True)
        # ex_struct = structures[list(structures.keys())[0]]
        # type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
        # pvec_len = ex_struct.len_param_vec

        print("MASTER: Preparing to send structures to slaves ... ", flush=True)
        # grouped_tup = group_for_mpi_scatter(structures, weights, true_forces,
        # grouped_tup = group_for_mpi_scatter(glob.glob(LOAD_PATH +
            # 'structures/*'), weights, true_forces, true_energies, mpi_size)

        # structures = grouped_tup[0]
        # struct_names = grouped_tup[0]
        # weights = grouped_tup[1]
        # true_forces = grouped_tup[2]
        # true_energies = grouped_tup[3]
    else:
        spline_indices = None
        # structures = None
        struct_names = None
        # weights = None
        # true_forces = None
        # true_energies = None

    # Send all necessary information to slaves
    # spline_indices = comm.bcast(spline_indices, root=0)

    # structures = comm.scatter(structures, root=0)
    # print(glob.glob(LOAD_PATH + 'structures/*')[:5])
    # struct_names = comm.scatter(np.split(np.array(glob.glob(LOAD_PATH +
        # 'structures/*')), mpi_size), root=0)
    # struct_names = comm.scatter(np.split(np.array([1,2,3,4,5]), mpi_size), root=0)

    # struct_names = comm.bcast(struct_names, root=0)
    struct_names = comm.scatter(struct_names, root=0)
    # print(struct_names)
    structures = load_locally(struct_names)
    weights = load_weights(structures.keys())
    true_forces, true_energies = load_true_values(structures.keys())

    print("MASTER: Determining potential information ...", flush=True)
    ex_struct = structures[list(structures.keys())[0]]
    type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
    pvec_len = ex_struct.len_param_vec
    # weights = comm.scatter(weights, root=0)
    # true_forces = comm.scatter(true_forces, root=0)
    # true_energies = comm.scatter(true_energies, root=0)

    print("SLAVE: Rank", rank, "received", len(structures), 'structures',
            flush=True)

    # Have every process build the toolbox
    pvec_len = structures[list(structures.keys())[0]].len_param_vec
    toolbox, creator = build_ga_toolbox(pvec_len, spline_indices)

    eval_fxn = build_evaluation_function(structures, weights, true_forces,
            true_energies, spline_indices)

    toolbox.register("evaluate_population", eval_fxn)

    # Compute initial fitnesses
    if is_master_node:
        pop = toolbox.population(n=POP_SIZE)
    else:
        pop = None

    pop = comm.bcast(pop, root=0)

    fitnesses = toolbox.evaluate_population(pop)
    all_fitnesses = comm.gather(fitnesses, root=0)

    # Have master gather fitnesses and update individuals
    if is_master_node:
        all_fitnesses = np.sum(all_fitnesses, axis=0)

        for ind,fit in zip(pop, all_fitnesses):
            ind.fitness.values = fit,

        # Sort population; best on top
        pop = tools.selBest(pop, len(pop))

        print_statistics(pop, 0, stats, logbook)

        checkpoint(pop, logbook, pop[0], 0)
        ga_start = time.time()

    # Begin GA
    if RUN_NEW_GA:
        i = 0
        while (i < NUM_GENS):
            if is_master_node:

                # Preserve top 2, cross others with top 2
                for j in range(2, len(pop)):
                    mom = pop[np.random.randint(2)]
                    dad = pop[j]

                    kid, _ = toolbox.mate(toolbox.clone(mom), toolbox.clone(dad))
                    pop[j] = kid

                # Mutate randomly everyone except top 2
                for ind in pop[2:]:
                    if MUTPB >= np.random.random():
                        toolbox.mutate(ind)
            else:
                pop = None

            # Send out updated population
            pop = comm.bcast(pop, root=0)

            # Run local minimization on best individual if desired
            if DO_POWELL and (i % 10 == 0):
                guess = pop[0]

                if is_master_node:
                    print("MASTER: performing powell minimization on best individual", flush=True)

                # time-saver for testing purposes; TODO: delete later
                was_run_prev = False

                if not was_run_prev:
                    improved = run_powell_on_best(guess, toolbox,
                            is_master_node, comm, NUM_POWELL_STEPS)

                if is_master_node:
                    if not was_run_prev:
                        np.savetxt("after_powell_temp-crystal-first.dat", improved)
                    else:
                        improved = np.genfromtxt("after_powell_temp.dat")

                    prev_pop = list(pop)
                    pop[0] = creator.Individual(improved)

            pop = comm.bcast(pop, root=0)

            # Compute fitnesses with mated/mutated/optimized population
            fitnesses = toolbox.evaluate_population(pop)
            all_fitnesses = comm.gather(fitnesses, root=0)

            # Update individuals with new fitnesses
            if is_master_node:
                all_fitnesses = np.sum(all_fitnesses, axis=0)

                for ind,fit in zip(pop, all_fitnesses):
                    ind.fitness.values = fit,

                # Sort
                pop = tools.selBest(pop, len(pop))

            # Print statistics to screen and checkpoint
            if is_master_node:
                print_statistics(pop, i+1, stats, logbook)

                if (i % CHECKPOINT_FREQUENCY == 0):
                    best = np.array(tools.selBest(pop, 1)[0])
                    checkpoint(pop, logbook, best, i)

            i += 1

        best_guess = pop[0]
    else:
        pop = np.genfromtxt(POP_FILE_NAME)
        best_guess = creator.Individual(pop[0])

    # Perform a final local optimization on the final results of the GA
    best_guess = comm.bcast(best_guess, root=0)

    best_fitness = toolbox.evaluate_population([best_guess])
    best_fitness = comm.gather(best_fitness, root=0)

    if is_master_node:
        ga_runtime = time.time() - ga_start
        print("MASTER: GA runtime = {:.2f} (s)".format(ga_runtime), flush=True)
        print("MASTER: Average time per step = {:.2f}"
                " (s)".format(ga_runtime/NUM_GENS), flush=True)

        best_fitness = np.sum(best_fitness, axis=0)

        print("MASTER: Fitness before final Powell = ", best_fitness[0],
                flush=True)

        pow_start = time.time()

    improved = run_powell_on_best(best_guess, toolbox,
            is_master_node, comm)

    if is_master_node:
        print("MASTER: Powell runtime = {:.2f} (s)".format(time.time() - pow_start))

        np.savetxt("after_powell_temp-crystal-final.dat", improved)

    optimized_fitness = toolbox.evaluate_population([improved])
    optimized_fitness = comm.gather(optimized_fitness, root=0)

    if is_master_node:
        optimized_fitness = np.sum(optimized_fitness, axis=0)[0]

        improved = creator.Individual(improved)
        improved.fitness.values = optimized_fitness,
        print("MASTER: Fitness after Powell =", optimized_fitness)

    # Save final results
    if is_master_node:
        improved_arr = np.array(improved)

        np.savetxt(SAVE_PATH + SETTINGS_STR + 'final_potential.dat', improved_arr)
        np.savetxt(open(TRACE_FILE_NAME, 'ab'), [improved_arr])

        plot_best_individual()

################################################################################ 

def build_ga_toolbox(pvec_len, index_ranges):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray,
            fitness=creator.CostFunctionMinimizer)

    def ret_pvec(arr_fxn, rng):
        # hard-coded version for pair-pots only
        ind = np.zeros(37)

        ranges = [(-0.5, 0.5), (-1, 4), (-1, 1)]

        ind[:10] += np.linspace(0.2*(-0.5), 0.8*(0.5), 10)[::-1]
        ind[:10] += np.random.normal(size=(10,), scale=0.1)

        ind[12:22] += np.linspace(0.2*(-1), 0.8*(4), 10)[::-1]
        ind[12:22] += np.random.normal(size=(10,), scale=(5)*0.1)

        ind[24:34] += np.linspace(0.2*(-1), 0.8, 10)[::-1]
        ind[24:34] += np.random.normal(size=(10,), scale=(2)*0.1)

        return arr_fxn(ind)

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual, np.random.normal)
    toolbox.register("population", tools.initRepeat, list, toolbox.parameter_set,)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    return toolbox, creator

def group_for_mpi_scatter(structs, database_weights, true_forces, true_energies,
        size):
    """Splits database information into groups to be scattered to nodes.

    Args:
        structs (dict): dictionary of Worker objects (key=name)
        database_weights (dict): dictionary of weights (key=name)
        true_forces (dict): dictionary of 'true' forces (key=name)
        true_energies (dict): dictionary of 'true' energies (key=name)
        size (int): number of MPI nodes available

    Returns:
        lists of grouped dictionaries of all input arguments
    """

    grouped_keys = np.array_split(list(database_weights.keys()), size)

    grouped_structs = []
    grouped_weights = []
    grouped_forces = []
    grouped_energies = []

    for group in grouped_keys:
        # tmp = {}
        tmp = []
        tmp2 = {}
        tmp3 = {}
        tmp4 = {}

        for name in group:
            # tmp[name] = structs[name]
            tmp.append(name)
            tmp2[name] = database_weights[name]
            tmp3[name] = true_forces[name]
            tmp4[name] = true_energies[name]

        grouped_structs.append(tmp)
        grouped_weights.append(tmp2)
        grouped_forces.append(tmp3)
        grouped_energies.append(tmp4)

    return grouped_structs, grouped_weights, grouped_forces, grouped_energies

def plot_best_individual():
    """Builds an animated plot of the trace of the GA. The final frame should be
    the final results after local optimization
    """

    trace = np.genfromtxt(TRACE_FILE_NAME)

    # currently only plots the 1st pair potential
    fig, ax = plt.subplots()
    ax.set_ylim([-2,2])
    ax.set_xlabel("0")

    sp = CubicSpline(np.arange(10), trace[0,:10])

    xlin = np.linspace(0, 9, 100)
    line, = ax.plot(xlin, sp(xlin))
    line2, = ax.plot(np.arange(10), trace[0,:10], 'bo')

    def animate(i):
        label = "{}".format(i)

        sp = CubicSpline(np.arange(10), trace[i,:10])
        line.set_ydata(sp(xlin))
        line2.set_ydata(trace[i,:10])

        ax.set_xlabel(label)
        return line, ax

    ani = animation.FuncAnimation(fig, animate, np.arange(1, trace.shape[0]),
            interval=200)

    ani.save('trace_of_best.gif', writer='imagemagick')

def prepare_save_directory():
    """Creates directories to store results"""

    print()
    print("Save location:", SAVE_PATH + SETTINGS_STR)
    if os.path.isdir(SAVE_PATH + SETTINGS_STR) and CHECK_BEFORE_OVERWRITE:
        print()
        print("/" + "*"*30 + " WARNING " + "*"*30 + "/")
        print("A folder already exists for these settings.\nPress Enter"
                " to ovewrite old data, or Ctrl-C to quit")
        input("/" + "*"*30 + " WARNING " + "*"*30 + "/")
    else:
        os.mkdir(SAVE_PATH + SETTINGS_STR)

    print()

def print_settings():
    """Prints settings to screen"""

    print("POP_SIZE:", POP_SIZE, flush=True)
    print("NUM_GENS:", NUM_GENS, flush=True)
    print("CXPB:", CXPB, flush=True)
    print("MUTPB:", MUTPB, flush=True)
    print("NUM_POWELL_STEPS:", NUM_POWELL_STEPS, flush=True)
    print("CHECKPOINT_FREQUENCY:", CHECKPOINT_FREQUENCY, flush=True)
    print()

def load_structures_on_master():
    """Builds Worker objects from the HDF5 database of stored values. Note that
    database weights are determined HERE.
    """

    # database = h5py.File(DB_FILE_NAME, 'a',)
    # weights = {key:1 for key in database.keys()}

    structures = {}
    weights = {}

    # for name in database.keys():
    for name in glob.glob(LOAD_PATH + 'structures/*')[:5]:
        if 'dimer' in name:
            short_name = os.path.split(name)[-1]
            short_name = os.path.splitext(short_name)[0]
            weights[short_name] = 1

            if weights[short_name] > 0:
                structures[short_name] = pickle.load(open(name, 'rb'))
        # if 'dimer' in name:
            # structures[name] = Worker.from_hdf5(database, name)

    # database.close()

    return structures, weights

def load_true_values(all_names):
    """Loads the 'true' values according to the database provided"""

    true_forces = {}
    true_energies = {}

    for name in all_names:

        fcs = np.genfromtxt(open(LOAD_PATH + 'info/info.' + name, 'rb'),
                skip_header=1)
        eng = np.genfromtxt(open(LOAD_PATH + 'info/info.' + name, 'rb'),
                max_rows=1)

        true_forces[name] = fcs
        true_energies[name] = eng

    return true_forces, true_energies

def find_spline_type_deliminating_indices(worker):
    """Finds the indices in the parameter vector that correspond to start/end
    (inclusive/exclusive respectively) for each spline group. For example,
    phi_range[0] is the index of the first know of the phi splines, while
    phi_range[1] is the next knot that is NOT part of the phi splines

    Args:
        worker (WorkerSpline): example worker that holds all spline objects
    """

    ntypes = worker.ntypes
    nphi = worker.nphi

    splines = worker.phis + worker.rhos + worker.us + worker.fs + worker.gs
    indices = [s.index for s in splines]

    phi_range = (indices[0], indices[nphi])
    rho_range = (indices[nphi], indices[nphi + ntypes])
    u_range = (indices[nphi + ntypes], indices[nphi + 2*ntypes])
    f_range = (indices[nphi + 2*ntypes], indices[nphi + 3*ntypes])
    g_range = (indices[nphi + 3*ntypes], -1)

    return [phi_range, rho_range, u_range, f_range, g_range], indices

def build_evaluation_function(structures, weights, true_forces, true_energies,
        spline_indices):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn(population):
        # Convert list of Individuals into a numpy array
        full = np.vstack(population)

        # hard-coded for phi splines only TODO: remove this later
        full = np.hstack([full, np.zeros((full.shape[0], 79))])

        fitnesses = np.zeros(full.shape[0])

        # Compute error for each worker on MPI node
        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(full) - true_forces[name]
            eng_err = w.compute_energy(full) - true_energies[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitnesses += fcs_err*fcs_err*weights[name]
            fitnesses += eng_err*eng_err*weights[name]

        return fitnesses

    return fxn

def build_stats_and_log():
    """Initialize DEAP Statistics and Logbook objects"""

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "size", "min", "max", "avg", "std"

    return stats, logbook

def print_statistics(pop, gen_num, stats, logbook):
    """Use Statistics and Logbook objects to output results to screen"""

    record = stats.compile(pop)
    logbook.record(gen=gen_num, size=len(pop), **record)
    print(logbook.stream, flush=True)

def run_powell_on_best(guess, toolbox, is_master_node, comm, num_steps=None):
    """Wrapper for scipy.optimize.fmin_powell function"""

    def cb(x):
        """Callback function called at end of every Powell step"""

        val = toolbox.evaluate_population([x])
        val = comm.gather(val, root=0)

        if is_master_node:
            val = np.sum(val, axis=0)
            print("MASTER: powell step:", val[0], flush=True)

    def parallel_wrapper(x, stop):
        """Wrapper to allow parallelization of fmin_powell. Explanation and code
        provided by Stackoverflow user 'francis'.

        Link: https://stackoverflow.com/questions/37159923/parallelize-a-function-call-with-mpi4py?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        """

        stop[0] = comm.bcast(stop[0], root=0)
        x = comm.bcast(x, root=0)

        value = 0

        if stop[0] == 0: # i.e. DON'T stop
            cost = toolbox.evaluate_population([x])
            all_costs = comm.gather(cost, root=0)

            if is_master_node:
                value = np.sum(all_costs, axis=0)

        return value

    if is_master_node:
        stop = [0]
        locally_optimized_pot = fmin_powell(parallel_wrapper, guess,
                args=(stop,), maxiter=num_steps, callback=cb, disp=0,
                ftol=1e-6, xtol=1e-6)
        stop = [1]
        parallel_wrapper(guess, stop)

        optimized_fitness = toolbox.evaluate_population([locally_optimized_pot])[0]
    else:
        stop = [0]
        while stop[0] == 0:
            parallel_wrapper(guess, stop)

        locally_optimized_pot = None
        optimized_fitness = None

    improved = comm.bcast(locally_optimized_pot, root=0)
    fitness = comm.bcast(optimized_fitness, root=0)

    return improved

def scale_into_range(original, index_ranges):
    # TODO: do you actually need this?

    new = original.copy()

    ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            (-0.2, 0.4), (-2, 3), (-7.5, 12), (-7, 2), (-1, 0.2), (-1, 1)]

    indices = index_ranges + [-1]

    mask = np.zeros(new.shape)

    for i,is_active in enumerate(ACTIVE_SPLINES):
        if is_active == 1:
            a, b = ranges[i]
            start, stop = indices[i], indices[i+1]

            block = new[start:stop]
            block -= np.min(block)
            block /= np.max(block)

            block = (b-a)*block + a

            new[start:stop] = block

            mask[start:stop] = 1

    return np.multiply(new, mask)

def checkpoint(population, logbook, trace_update, i):
    """Saves information to files for later use"""

    np.savetxt(POP_FILE_NAME + str(i), population)
    pickle.dump(logbook, open(LOG_FILE_NAME, 'wb'))
    np.savetxt(open(TRACE_FILE_NAME, 'ab'), [trace_update])

def load_locally(long_names):
    structures = {}

    for name in long_names:
        short_name = os.path.split(name)[-1]
        # short_name = os.path.splitext(short_name)[0]
        structures[short_name] = pickle.load(open(LOAD_PATH + 'structures/' +
            name + '.pkl', 'rb'))

    return structures

def get_all_struct_names():
    path_list = glob.glob(LOAD_PATH + 'structures/*')
    short_names = [os.path.split(name)[-1] for name in path_list]
    no_ext = [os.path.splitext(name)[0] for name in short_names]

    return no_ext

def load_weights(names):
    return {key:1 for key in names}

################################################################################

if __name__ == "__main__":
    main()
