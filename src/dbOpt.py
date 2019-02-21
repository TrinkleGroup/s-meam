import os
# TODO: BW settings
# os.chdir("/mnt/c/Users/jvita/scripts/s-meam/")
import sys

sys.path.append('./')

import numpy as np
import random

# np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)

import pickle
import glob
import array
import h5py
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import least_squares, fmin_cg, fmin_bfgs
from scipy.interpolate import CubicSpline
# from scipy.sparse import csr_matrix
from mpi4py import MPI
# from memory_profiler import profile

from deap import base, creator, tools, algorithms

import src.meam
import src.partools as partools
from src.meam import MEAM
from src.worker import Worker
from src.meam import MEAM
from src.spline import Spline
from src.database import Database
from src.potential_templates import Template
from src.node import Node
from src.manager import Manager

################################################################################
"""MPI settings"""

MASTER_RANK = 0

################################################################################
"""MEAM potential settings"""

NTYPES = 2
# ACTIVE_SPLINES = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ACTIVE_SPLINES = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

################################################################################
"""GA settings"""

if len(sys.argv) > 1:
    POP_SIZE = int(sys.argv[1])
else:
    POP_SIZE = 1

# TODO: BW settings
if len(sys.argv) > 2:
    NUM_GENS = int(sys.argv[2])
else:
    NUM_GENS = 10

if len(sys.argv) > 3:
    MUTPB = float(sys.argv[3])
else:
    MUTPB = 0.5

CXPB = 1.0

RUN_NEW_GA = True

FLATTEN_LANDSCAPE = False  # define fitness as fitness of partially-minimized pot
FLAT_NSTEPS = 5

DO_LMIN = False
LMIN_FREQUENCY = 1
NUM_LMIN_STEPS = 20
# INIT_NSTEPS = 30
# INTER_NSTEPS = 5
# FINAL_NSTEPS = 30

CHECKPOINT_FREQUENCY = 1

MATING_ALPHA = 0.2

################################################################################
"""I/O settings"""

date_str = datetime.datetime.now().strftime("%Y-%m-%d")

CHECK_BEFORE_OVERWRITE = False

# TODO: BW settings
BASE_PATH = ""
BASE_PATH = "/home/jvita/scripts/s-meam/"

LOAD_PATH = "/projects/sciteam/baot/pz-unfx-cln/"
LOAD_PATH = BASE_PATH + "data/fitting_databases/pinchao/"
SAVE_PATH = BASE_PATH + "data/results/"

SAVE_DIRECTORY = SAVE_PATH + date_str + "-" + "meam" + "{}-{}".format(NUM_GENS,
                                                                      MUTPB)

if os.path.isdir(SAVE_DIRECTORY):
    SAVE_DIRECTORY = SAVE_DIRECTORY + '-' + str(np.random.randint(100000))

DB_PATH = LOAD_PATH + 'structures'
DB_INFO_FILE_NAME = LOAD_PATH + 'full/info'
POP_FILE_NAME = SAVE_DIRECTORY + "/pop.dat"
LOG_FILE_NAME = SAVE_DIRECTORY + "/ga.log"
TRACE_FILE_NAME = SAVE_DIRECTORY + "/trace.dat"


################################################################################

def main():
    # Record MPI settings
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
        # Prepare directories and files
        print_settings()

        print("MASTER: Preparing save directory/files ... ", flush=True)
        prepare_save_directory()

        # Trace file to be appended to later
        f = open(TRACE_FILE_NAME, 'ab')
        f.close()

        # GA tools
        stats, logbook = build_stats_and_log()

        # Prepare database and potential template
        potential_template = partools.initialize_potential_template(LOAD_PATH)
        # potential_template.print_statistics()
        # print()

        struct_files = glob.glob(DB_PATH + "/*")

        fitting_database = Database(DB_PATH, DB_INFO_FILE_NAME)
        testing_database = Database(DB_PATH, DB_INFO_FILE_NAME)

        fitting_database.read_pinchao_formatting(
            os.path.join(LOAD_PATH, 'Database-Structures'), 'fitting'
        )

        testing_database.read_pinchao_formatting(
            os.path.join(LOAD_PATH, 'Database-Structures'), 'testing'
        )

        # fitting_database.print_metadata()

        # all_struct_names  , structures = zip(*fitting_database.structures.items())
        fitting_struct_names = fitting_database.unique_structs
        testing_struct_names = testing_database.unique_structs

        fitting_struct_natoms = fitting_database.unique_natoms
        testing_struct_natoms = testing_database.unique_natoms

        print([(entry.struct_name, entry.ref_struct, entry.type) for entry in fitting_database.entries])

        fitting_num_entries = len(fitting_database.entries)
        testing_num_entries = len(testing_database.entries)

        fitting_worker_ranks = partools.compute_procs_per_subset(
            fitting_struct_natoms, world_size
        )

        testing_worker_ranks = partools.compute_procs_per_subset(
            testing_struct_natoms, world_size
        )

        pz_weights = [
           0.283, 0.03, 0.0682, 0.152, 0.00362, 0.0101, 0.0460, 0.0948, 0.07665,
           0.0898, 0.0372, 0.0689, 0.0395
        ]

        print("fitting_worker_ranks:", fitting_worker_ranks)
    else:
        potential_template = None
        fitting_database = None
        testing_database = None
        # num_structs = None
        fitting_worker_ranks = None
        testing_worker_ranks = None
        fitting_struct_names = None
        testing_struct_names = None
        fitting_num_entries = None
        testing_num_entries = None
        pz_weights = None

    potential_template = world_comm.bcast(potential_template, root=0)

    fitting_struct_names = world_comm.bcast(fitting_struct_names, root=0)
    testing_struct_names = world_comm.bcast(testing_struct_names, root=0)

    fitting_num_entries = world_comm.bcast(fitting_num_entries, root=0)
    testing_num_entries = world_comm.bcast(testing_num_entries, root=0)

    # Send structs to managers and build communicators
    if is_master:
        print("Preparing Managers for fitting database ...", flush=True)

    fitting_manager_stuff = prepare_managers(
        world_comm, fitting_worker_ranks, world_rank, fitting_struct_names,
        potential_template
    )

    fitting_manager = fitting_manager_stuff[0]
    is_fitting_manager = fitting_manager_stuff[1]
    fitting_manager_comm = fitting_manager_stuff[2]
    fitting_worker_comm = fitting_manager_stuff[3]

    if is_master:
        print("Preparing Managers for testing database ...", flush=True)

    testing_manager_stuff = prepare_managers(
        world_comm, testing_worker_ranks, world_rank, testing_struct_names,
        potential_template
    )

    testing_manager = testing_manager_stuff[0]
    is_testing_manager = testing_manager_stuff[1]
    testing_manager_comm = testing_manager_stuff[2]
    testing_worker_comm = testing_manager_stuff[3]

    # TODO: these should actually be called "error functions" or something
    # Define functions for computing fitnesses and gradients
    fitting_fxn, fitting_grad = partools.build_evaluation_functions(
        potential_template, fitting_database, fitting_struct_names,
        fitting_manager, is_master, is_fitting_manager, fitting_manager_comm,
        flatten=FLATTEN_LANDSCAPE
    )

    testing_fxn, testing_grad = partools.build_evaluation_functions(
        potential_template, testing_database, testing_struct_names,
        testing_manager, is_master, is_testing_manager, testing_manager_comm,
        flatten=FLATTEN_LANDSCAPE
    )

    # Have every process build the toolbox
    toolbox, creator = build_ga_toolbox(fitting_num_entries)

    objective_fxn = partools.build_objective_function(
        testing_database, testing_fxn, is_master
    )

    # Create the original population of weights
    if is_master:
        weight_pop = toolbox.population(n=POP_SIZE)
    else:
        weight_pop = 0

    weight_pop = np.array(weight_pop)
    weight_pop = world_comm.bcast(weight_pop, root=0)

    weight_pop_shape = world_comm.bcast(weight_pop.shape, root=0)

    all_mle = []
    obj_fxn_values = []
    for weights in weight_pop:
        init_guess = potential_template.generate_random_instance()[
            np.where(potential_template.active_mask)[0]
        ]

        mle = local_minimization(
            np.atleast_2d(init_guess), fitting_fxn, fitting_grad, weights,
            world_comm, is_master, nsteps=NUM_LMIN_STEPS
        )
        all_mle.append(mle)
        obj_fxn_values.append(objective_fxn(mle, weights))

    if is_master:

        print(
            "MASTER: initial objective function values:", obj_fxn_values, flush=True
        )

        print(
            "avg min max:", np.average(obj_fxn_values), np.min(obj_fxn_values),
            np.max(obj_fxn_values), flush=True
        )

    # Have master gather fitnesses and update individuals
    if is_master:

        pop_copy = []
        for ind in weight_pop:
            pop_copy.append(creator.Individual(ind))

        weight_pop = pop_copy

        for ind, fit in zip(weight_pop, obj_fxn_values):
            ind.fitness.values = fit,

        # Sort population; best on top
        weight_pop = tools.selBest(weight_pop, len(weight_pop))

        print_statistics(weight_pop, 0, stats, logbook)

        checkpoint(weight_pop, logbook, weight_pop[0], 0)
        ga_start = time.time()

    # Begin GA
    if RUN_NEW_GA:
        generation_number = 1
        while (generation_number < NUM_GENS):
            if is_master:

                # TODO: currently using blend for mutate

                # Preserve top 50%, breed survivors
                for pot_num in range(len(weight_pop) // 2, len(weight_pop)):
                    mom_idx = np.random.randint(len(weight_pop) // 2)

                    dad_idx = mom_idx
                    while dad_idx == mom_idx:
                        dad_idx = np.random.randint(len(weight_pop) // 2)

                    mom = weight_pop[mom_idx]
                    dad = weight_pop[dad_idx]

                    kid, _ = toolbox.mate(toolbox.clone(mom),
                                          toolbox.clone(dad))
                    weight_pop[pot_num] = kid

                # TODO: debug to make sure pop is always sorted here

                # Mutate randomly everyone except top 10% (or top 2)
                for mut_ind in weight_pop[max(2, int(POP_SIZE / 10)):]:
                    if np.random.random() >= MUTPB: toolbox.mutate(mut_ind)

            # Compute fitnesses with mated/mutated/optimized population
            # fitnesses, max_ni = toolbox.evaluate_population(weight_pop, True)

            new_mle = []
            obj_fxn_values = []
            for old_mle, weights in zip(all_mle, weight_pop):
                mle = local_minimization(
                    old_mle, fitting_fxn, fitting_grad, weights, world_comm,
                    is_master, nsteps=NUM_LMIN_STEPS
                )
                obj_fxn_values.append(objective_fxn(mle, weights))
                new_mle.append(mle)
            all_mle = new_mle

            # Update individuals with new fitnesses
            if is_master:
                pop_copy = []
                for ind in weight_pop:
                    pop_copy.append(creator.Individual(ind))

                weight_pop = pop_copy

                for ind, fit in zip(weight_pop, obj_fxn_values):
                    ind.fitness.values = fit,

                # Sort
                weight_pop = tools.selBest(weight_pop, len(weight_pop))

                # Print statistics to screen and checkpoint
                print_statistics(weight_pop, generation_number, stats, logbook)

                if (generation_number % CHECKPOINT_FREQUENCY == 0):
                    best = np.array(tools.selBest(weight_pop, 1)[0])
                    checkpoint(weight_pop, logbook, best, generation_number)

                # best_guess = weight_pop[0]
            else:
                pass
                # best_guess = None

            generation_number += 1
    else:
        weight_pop = np.genfromtxt(POP_FILE_NAME)
        # best_guess = creator.Individual(weight_pop[0])

    # best_guess = world_comm.bcast(best_guess, root=0)

    # TODO: figure out how to still use parallel LM, even when each has diff weight

    new_mle = []
    obj_fxn_values = []
    for old_mle, weights in zip(all_mle, weight_pop):
        mle = local_minimization(
            np.atleast_2d(old_mle), fitting_fxn, fitting_grad, weights, world_comm,
            is_master, nsteps=NUM_LMIN_STEPS
        )[0]
        obj_fxn_values.append(objective_fxn(mle, weights))
        new_mle.append(mle)
    all_mle = new_mle

    # Perform a final local optimization on the final results of the GA
    if is_master:
        print(
            "MASTER: final objective function values:", obj_fxn_values, flush=True
        )

        print(
            "avg min max:", np.average(obj_fxn_values), np.min(obj_fxn_values),
            np.max(obj_fxn_values), flush=True
        )

        ga_runtime = time.time() - ga_start

        checkpoint(weight_pop, logbook, weight_pop[0], 1)
        print(np.array(all_mle))
        np.savetxt("final_mle_params.dat", np.array(all_mle))

        print("MASTER: GA runtime = {:.2f} (s)".format(ga_runtime), flush=True)
        print("MASTER: Average time per step = {:.2f}"
              " (s)".format(ga_runtime / NUM_GENS), flush=True)


################################################################################

def build_ga_toolbox(num_fitting_structs):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray,
                   fitness=creator.CostFunctionMinimizer)

    # def ret_pvec(arr_fxn):
    #     # hard-coded version for pair-pots only
    #     tmp = arr_fxn(potential_template.generate_random_instance())
    #
    #     return tmp[np.where(potential_template.active_mask)[0]]
    def ret_pvec():
        return np.random.random(num_fitting_structs)

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.parameter_set, )

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("mate", tools.cxBlend, alpha=MATING_ALPHA)
    # toolbox.register("mate", tools.cxTwoPoint)

    return toolbox, creator


def group_for_mpi_scatter(database, size):
    """Splits database information into groups to be scattered to nodes.

    Args:
        database (Database): structures and true values
        size (int): number of MPI nodes available

    Returns:
        lists of grouped dictionaries of all input arguments
    """

    grouped_keys = np.array_split(list(database.structures.keys()), size)

    grouped_databases = []

    for group in grouped_keys:
        tmp_structs = {}
        tmp_energies = {}
        tmp_forces = {}
        tmp_weights = {}

        for name in group:
            tmp_structs[name] = database.structures[name]
            tmp_energies[name] = database.true_energies[name]
            tmp_forces[name] = database.true_forces[name]
            tmp_weights[name] = 1

        grouped_databases.append(
            Database.manual_init(
                tmp_structs, tmp_energies, tmp_forces, tmp_weights,
                database.reference_struct, database.reference_energy
            )
        )

    return grouped_databases


def plot_best_individual():
    """Builds an animated plot of the trace of the GA. The final frame should be
    the final results after local optimization
    """

    trace = np.genfromtxt(TRACE_FILE_NAME)

    # currently only plots the 1st pair potential
    fig, ax = plt.subplots()
    ax.set_ylim([-2, 2])
    ax.set_xlabel("0")

    sp = CubicSpline(np.arange(10), trace[0, :10])

    xlin = np.linspace(0, 9, 100)
    line, = ax.plot(xlin, sp(xlin))
    line2, = ax.plot(np.arange(10), trace[0, :10], 'bo')

    def animate(i):
        label = "{}".format(i)

        sp = CubicSpline(np.arange(10), trace[i, :10])
        line.set_ydata(sp(xlin))
        line2.set_ydata(trace[i, :10])

        ax.set_xlabel(label)
        return line, ax

    ani = animation.FuncAnimation(fig, animate, np.arange(1, trace.shape[0]),
                                  interval=200)

    ani.save('trace_of_best.gif', writer='imagemagick')


def prepare_save_directory():
    """Creates directories to store results"""

    print()
    print("Save location:", SAVE_DIRECTORY)
    if os.path.isdir(SAVE_DIRECTORY) and CHECK_BEFORE_OVERWRITE:
        print()
        print("/" + "*" * 30 + " WARNING " + "*" * 30 + "/")
        print("A folder already exists for these settings.\nPress Enter"
              " to ovewrite old data, or Ctrl-C to quit")
        input("/" + "*" * 30 + " WARNING " + "*" * 30 + "/\n")
    print()

    # os.rmdir(SAVE_DIRECTORY)
    os.mkdir(SAVE_DIRECTORY)


def print_settings():
    """Prints settings to screen"""

    print("POP_SIZE:", POP_SIZE, flush=True)
    print("NUM_GENS:", NUM_GENS, flush=True)
    print("CXPB:", CXPB, flush=True)
    print("MUTPB:", MUTPB, flush=True)
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

    i = 0
    # for name in database.keys():
    for name in glob.glob(DB_PATH + '*'):

        # if 'dimer' in name:
        short_name = os.path.split(name)[-1]
        short_name = os.path.splitext(short_name)[0]
        weights[short_name] = 1

        if weights[short_name] > 0:
            i += 1
            structures[short_name] = pickle.load(open(name, 'rb'))

    # database.close()

    return structures, weights


def load_true_values(all_names):
    """Loads the 'true' values according to the database provided"""

    true_forces = {}
    true_energies = {}

    for name in all_names:
        fcs = np.genfromtxt(open(DB_INFO_FILE_NAME + '/info.' + name, 'rb'),
                            skip_header=1)
        eng = np.genfromtxt(open(DB_INFO_FILE_NAME + '/info.' + name, 'rb'),
                            max_rows=1)

        true_forces[name] = fcs
        true_energies[name] = eng

    return true_forces, true_energies


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


# @profile
def local_minimization(
    parameters, eval_fxn, grad_fxn, weights, world_comm, is_master, nsteps=20
    ):

    pad = 26

    def lm_fxn_wrap(parameters_raveled, original_shape):
        val = eval_fxn(
            parameters_raveled.reshape(original_shape), weights
        )

        val = world_comm.bcast(val, root=0)
        # TODO: add optional padding in input file for LM

        # pad with zeros since num structs is less than num knots
        return np.concatenate([val.ravel(), np.zeros(pad*original_shape[0])])

    def lm_grad_wrap(parameters_raveled, original_shape):
        # shape: (num_pots, num_structs*2, num_params)

        grads = grad_fxn(
            parameters_raveled.reshape(original_shape), weights
        )

        grads = world_comm.bcast(grads, root=0)

        num_pots, num_structs_2, num_params = grads.shape

        padded_grad = np.zeros(
            (num_pots, num_structs_2, num_pots, num_params)
        )

        for pot_id, g in enumerate(grads):
            padded_grad[pot_id, :, pot_id, :] = g

        padded_grad = padded_grad.reshape(
            (num_pots * num_structs_2, num_pots * num_params)
        )


        # also pad with zeros since num structs is less than num knots

        return np.vstack([
            padded_grad,
            np.zeros((pad*num_pots, num_pots * num_params))]
        )

    parameters = world_comm.bcast(parameters, root=0)
    parameters = np.array(parameters)

    opt_results = least_squares(
        lm_fxn_wrap, parameters.ravel(), lm_grad_wrap,
        method='lm', max_nfev=nsteps, args=(parameters.shape,)
    )

    # def wrap(x):
    #     val = np.sum(lm_fxn_wrap(x, master_pop.shape))
    #     val = world_comm.bcast(val, root=0)
    #     return val
    #
    # # write_count = 0
    # import time
    # def cb(x):
    #     if is_master:
    #         # write_count += 1
    #         np.savetxt("poop_" + str(time.time()) + ".dat", x)
    #     print(wrap(x), flush=True)
    #
    # # opt_results = fmin_cg(wrap, master_pop, callback=cb)
    # opt_results = fmin_bfgs(wrap, master_pop, callback=cb)

    if is_master:
        new_pop = opt_results['x'].reshape(parameters.shape)
        # new_pop = opt_results
    else:
        new_pop = None

    org_fits = eval_fxn(parameters, weights)
    new_fits = eval_fxn(new_pop, weights)

    if is_master:
        updated_params = list(parameters)

        for i, ind in enumerate(new_pop):
            if np.sum(new_fits[i]) < np.sum(org_fits[i]):
                updated_params[i] = creator.Individual(new_pop[i])
            else:
                updated_params[i] = creator.Individual(updated_params[i])

        parameters = updated_params

    obj_fxn_values = eval_fxn(parameters, weights)

    return parameters


def old_local_minimization(guess, toolbox, is_master, comm, num_steps=None, ):
    """Wrapper for local minimization function"""

    def parallel_wrapper(x, stop):
        """Wrapper to allow parallelization of local minimization. Explanation
        and code provided by Stackoverflow user 'francis'.

        Link: https://stackoverflow.com/questions/37159923/parallelize-a-function-call-with-mpi4py?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        """

        stop[0] = comm.bcast(stop[0], root=0)
        x = comm.bcast(x, root=0)

        value = 0

        if stop[0] == 0:  # i.e. DON'T stop
            cost = toolbox.evaluate_population([x])
            all_costs = comm.gather(cost, root=0)

            if is_master:
                print('costs', [x.shape for x in all_costs], flush=True)
                value = np.concatenate(all_costs)

        return value

    def parallel_grad_wrapper(x, stop):
        """Wrapper to allow parallelization of gradient calculation."""

        stop[0] = comm.bcast(stop[0], root=0)
        x = comm.bcast(x, root=0)

        value = 0

        if stop[0] == 0:  # i.e. DON'T stop
            grad = toolbox.gradient([x])
            all_grads = comm.gather(grad, root=0)

            if is_master:
                print('grads', [x.shape for x in all_grads], flush=True)
                value = np.vstack(all_grads)

        return value

    def big_wrapper(x, stop):

        stop[0] = comm.bcast(stop[0], root=0)
        x = comm.bcast(x, root=0)

        value = 0

        if stop[0] == 0:  # i.e. DON'T stop
            grad = toolbox.gradient([x])
            all_grads = comm.gather(grad, root=0)

            if is_master:
                print('grads', [x.shape for x in all_grads], flush=True)
                value = np.vstack(all_grads)

        return value

    if is_master:
        stop = [0]

        # opt_pot = fmin_cg(parallel_wrapper, guess,
        #                                 parallel_grad_wrapper, args=(stop,),
        #                                 maxiter=num_steps,
        #                                 callback=cb, disp=0, gtol=1e-6)

        opt_results = least_squares(
            parallel_wrapper, guess, parallel_grad_wrapper, method='lm',
            args=(stop,), max_nfev=num_steps * 2
        )

        opt_pot = opt_results['x']

        stop = [1]
        parallel_wrapper(guess, stop)
        parallel_grad_wrapper(guess, stop)
        # optimized_fitness = toolbox.evaluate_population([opt_pot])[0]
    else:
        stop = [0]
        while stop[0] == 0:
            parallel_wrapper(guess, stop)
            parallel_grad_wrapper(guess, stop)

        opt_pot = None
        optimized_fitness = None

    improved = comm.bcast(opt_pot, root=0)
    fitness = comm.bcast(optimized_fitness, root=0)

    return improved


def checkpoint(population, logbook, trace_update, i):
    """Saves information to files for later use"""

    np.savetxt(POP_FILE_NAME + str(i), population)
    pickle.dump(logbook, open(LOG_FILE_NAME, 'wb'))

    f = open(TRACE_FILE_NAME, 'ab')
    np.savetxt(f, [np.array(trace_update)])
    f.close()


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
    return {key: 1 for key in names}


def find_spline_type_deliminating_indices(worker):
    """Finds the indices in the parameter vector that correspond to start/end
    (inclusive/exclusive respectively) for each spline group. For example,
    phi_range[0] is the index of the first knot of the phi splines, while
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
    u_range = (indices[nphi + ntypes], indices[nphi + 2 * ntypes])
    f_range = (indices[nphi + 2 * ntypes], indices[nphi + 3 * ntypes])
    g_range = (indices[nphi + 3 * ntypes], -1)

    return [phi_range, rho_range, u_range, f_range, g_range], indices



def split_population(a, n):
    """
    Stackoverflow credits (User = "tixxit"):
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/37414115
    """
    k, m = divmod(len(a), n)
    return list(
        a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def rescale_rhos(pop, per_u_max_ni, potential_template):
    ntypes = len(potential_template.u_ranges)
    nphi = int(ntypes * (ntypes + 1) / 2)

    rho_indices = potential_template.spline_indices[nphi - 1: nphi - 1 + ntypes]

    pop_arr = np.array(pop)

    for i, r_ind in enumerate(rho_indices):
        start, stop = r_ind

        # pull scaling factors, only scale if ni fall out of U range
        scaling = np.clip(per_u_max_ni[:, i], 1, None)

        pop_arr[:, start:stop] /= scaling[:, np.newaxis]

    return pop_arr

def prepare_managers(
    world_comm, worker_ranks, world_rank, struct_names, potential_template
    ):

    # each Manager is in charge of a single structure
    world_group = world_comm.Get_group()

    rank_lists = world_comm.bcast(worker_ranks, root=0)

    # Tell workers which manager they are a part of
    worker_ranks = None
    manager_ranks = []
    for per_manager_ranks in rank_lists:
        manager_ranks.append(per_manager_ranks[0])

        if world_rank in per_manager_ranks:
            worker_ranks = per_manager_ranks

    # manager_comm connects all manager processes
    manager_group = world_group.Incl(manager_ranks)
    manager_comm = world_comm.Create(manager_group)

    is_manager = (manager_comm != MPI.COMM_NULL)

    # One manager per structure
    if is_manager:
        manager_rank = manager_comm.Get_rank()

        struct_name = manager_comm.scatter(struct_names, root=0)

        print(
            "Manager", manager_rank, "received structure", struct_name, "plus",
            len(worker_ranks), "processors for evaluation", flush=True
        )

    else:
        struct_name = None
        manager_rank = None

    worker_group = world_group.Incl(worker_ranks)
    worker_comm = world_comm.Create(worker_group)

    struct_name = worker_comm.bcast(struct_name, root=0)
    manager_rank = worker_comm.bcast(manager_rank, root=0)

    manager = Manager(manager_rank, worker_comm, potential_template)

    manager.struct_name = struct_name
    manager.struct = manager.load_structure(
        manager.struct_name, DB_PATH + "/"
    )

    manager.struct = manager.broadcast_struct(manager.struct)

    return manager, is_manager, manager_comm, worker_comm

################################################################################

if __name__ == "__main__":
    main()
