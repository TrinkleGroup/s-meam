import os
# TODO: BW settings
# os.chdir("/mnt/c/Users/jvita/scripts/s-meam/")
import sys

sys.path.append('./')

import numpy as np
import random

np.set_printoptions(precision=8, suppress=True)

import pickle
import glob
import array
import h5py
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import least_squares
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
INIT_NSTEPS = 5
INTER_NSTEPS = 5
FINAL_NSTEPS = 30

CHECKPOINT_FREQUENCY = 1

MATING_ALPHA = 0.2

################################################################################
"""I/O settings"""

date_str = datetime.datetime.now().strftime("%Y-%m-%d")

CHECK_BEFORE_OVERWRITE = False

# TODO: BW settings
BASE_PATH = ""
BASE_PATH = "/home/jvita/scripts/s-meam/"

# LOAD_PATH = "data/fitting_databases/fixU/"
LOAD_PATH = BASE_PATH + "data/fitting_databases/pinchao/"
# LOAD_PATH = "/projects/sciteam/baot/pz-unfx-cln/"
SAVE_PATH = BASE_PATH + "data/results/"

SAVE_DIRECTORY = SAVE_PATH + date_str + "-" + "meam" + "{}-{}".format(NUM_GENS,
                                                                      MUTPB)

if os.path.isdir(SAVE_DIRECTORY):
    SAVE_DIRECTORY = SAVE_DIRECTORY + '-' + str(np.random.randint(100000))

DB_PATH = LOAD_PATH + 'mini_structures'
DB_INFO_FILE_NAME = LOAD_PATH + 'mini/info'
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
        potential_template = initialize_potential_template()
        potential_template.print_statistics()
        print()

        # rzm: natoms is reading from true values

        struct_files = glob.glob(DB_PATH + "/*")

        master_database = Database(DB_PATH, DB_INFO_FILE_NAME)

        if 'pinchao' in LOAD_PATH:
            master_database.read_pinchao_formatting(
                os.path.join(LOAD_PATH, 'Database-Structures')
            )

        master_database.print_metadata()

        # all_struct_names, structures = zip(*master_database.structures.items())
        all_struct_names, struct_natoms = zip(*master_database.natoms.items())
        num_structs = len(struct_natoms)

        worker_ranks = partools.compute_procs_per_subset(
            struct_natoms, world_size
        )

        print("worker_ranks:", worker_ranks)
    else:
        structures = None
        manager_subsets = None
        potential_template = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

    potential_template = world_comm.bcast(potential_template, root=0)
    num_structs = world_comm.bcast(num_structs, root=0)

    # each Manager is in charge of a single structure
    world_group = world_comm.Get_group()

    all_rank_lists = world_comm.bcast(worker_ranks, root=0)

    # Tell workers which manager they are a part of
    worker_ranks = None
    manager_ranks = []
    for per_manager_ranks in all_rank_lists:
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

        struct_name = manager_comm.scatter(all_struct_names, root=0)

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

    # manager.struct = manager.broadcast_struct(tmp_struct)

    # for rank_id in range(1, worker_comm.Get_size()):
    #     if manager_rank == 0:
    #         worker_comm.send(tmp_struct, dest=rank_id, tag=1)
    #     elif manager_rank == rank_id:
    #         manager.struct = worker_comm.recv(tmp_struct, source=0, tag=1)

    def fxn_wrap(master_pop):
        """Master: returns all potentials for all structures"""
        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
            # pop = np.ones((POP_SIZE, 5))
        else:
            pop = None

        if FLATTEN_LANDSCAPE:
            master_pop = local_minimization(
                master_pop, toolbox, world_comm, is_master, nsteps=FLAT_NSTEPS
            )

            pop = manager_comm.bcast(master_pop, root=0)

        # pop = worker_comm.bcast(pop, root=0)
        eng = manager.compute_energy(pop)
        fcs = manager.compute_forces(pop)

        # fitnesses = csr_matrix((pop.shape[0], 2 * num_structs))
        fitnesses = 0

        if is_manager:
            mgr_eng = manager_comm.gather(eng, root=0)
            mgr_fcs = manager_comm.gather(fcs, root=0)

            if is_master:
                # note: can't stack mgr_fcs b/c different dimensions per struct
                all_eng = np.vstack(mgr_eng)
                all_fcs = mgr_fcs

                w_energies = np.zeros((len(pop), num_structs))
                t_energies = np.zeros(num_structs)

                fcs_fitnesses = np.zeros((len(pop), num_structs))

                for s_id, name in enumerate(all_struct_names):
                    w_energies[:, s_id] = all_eng[s_id]
                    t_energies[s_id] = master_database.true_energies[name]

                    # if name == master_database.reference_struct:
                    #     ref_energy = w_energies[:, s_id]

                    w_fcs = all_fcs[s_id]
                    true_fcs = master_database.true_forces[name]

                    fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
                    fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

                    fcs_fitnesses[:, s_id] = fcs_err

                # w_energies -= ref_energy
                # t_energies -= master_database.reference_energy

                # eng_fitnesses = np.zeros((len(pop), num_structs))
                #
                # for s_id, (w_eng,t_eng) in enumerate(
                #     zip(w_energies.T, t_energies)):
                #     eng_fitnesses[:, s_id] = (w_eng - t_eng) ** 2

                # TODO: this only works for how Pinchao's DB is formatted

                eng_fitnesses = np.zeros(
                    (len(pop), len(master_database.reference_structs))
                )

                for fit_id, (s_name, ref) in enumerate(
                        master_database.reference_structs.items()):
                    r_name = ref.ref_struct
                    true_ediff = ref.energy_difference

                    # find index of structures to know which energies to use
                    s_id = all_struct_names.index(s_name)
                    r_id = all_struct_names.index(r_name)

                    comp_ediff = w_energies[:, s_id] - w_energies[:, r_id]
                    # comp_ediff = 0

                    eng_fitnesses[:, fit_id] = (comp_ediff - true_ediff) ** 2

                fitnesses = np.hstack([eng_fitnesses, fcs_fitnesses])

                # print(np.sum(fitnesses, axis=1), flush=True)

        return fitnesses

    # @profile
    def grad_wrap(master_pop):
        """Evalautes the gradient for all potentials in the population"""

        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
        else:
            pop = None

        # pop = worker_comm.bcast(pop, root=0)

        eng = manager.compute_energy(pop)
        fcs = manager.compute_forces(pop)

        eng_grad = manager.compute_energy_grad(pop)
        fcs_grad = manager.compute_forces_grad(pop)

        # gradient = csr_matrix((pop.shape[0], 2 * num_structs, pop.shape[1]))
        gradient = 0

        if is_manager:
            mgr_eng = manager_comm.gather(eng, root=0)
            mgr_fcs = manager_comm.gather(fcs, root=0)

            mgr_eng_grad = manager_comm.gather(eng_grad, root=0)
            mgr_fcs_grad = manager_comm.gather(fcs_grad, root=0)

            if is_master:
                # note: can't stack mgr_fcs b/c different dimensions per struct
                all_eng = np.vstack(mgr_eng)
                all_fcs = mgr_fcs

                w_energies = np.zeros((len(pop), num_structs))
                t_energies = np.zeros(num_structs)

                fcs_grad_vec = np.zeros(
                    (len(pop), potential_template.pvec_len, num_structs)
                )

                ref_energy = 0

                for s_id, name in enumerate(all_struct_names):
                    w_energies[:, s_id] = all_eng[s_id]
                    t_energies[s_id] = master_database.true_energies[name]

                    if name == master_database.reference_struct:
                        ref_energy = w_energies[:, s_id]

                    w_fcs = all_fcs[s_id]
                    true_fcs = master_database.true_forces[name]

                    fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2

                    fcs_grad = mgr_fcs_grad[s_id]

                    scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
                    summed = scaled.sum(axis=1).sum(axis=1)

                    fcs_grad_vec[:, :, s_id] += (2 * summed / 10)

                # w_energies -= ref_energy
                # t_energies -= database.reference_energy

                # eng_grad_vec = np.zeros(
                #     (len(pop), potential_template.pvec_len, num_structs)
                # )
                #
                # for s_id, (name, w_eng, t_eng), in enumerate(
                #     zip(all_struct_names, w_energies.T, t_energies)):
                #
                #     eng_err = (w_eng - t_eng)
                #     eng_grad = mgr_eng_grad[s_id]
                #
                #     eng_grad_vec[:, :, s_id] += (
                #         eng_err[:, np.newaxis] * eng_grad * 2
                #     )

                eng_grad_vec = np.zeros(
                    (len(pop), potential_template.pvec_len,
                     len(master_database.reference_structs))
                )

                for fit_id, (s_name, ref) in enumerate(
                        master_database.reference_structs.items()):
                    r_name = ref.ref_struct
                    true_ediff = ref.energy_difference

                    # find index of structures to know which energies to use
                    s_id = all_struct_names.index(s_name)
                    r_id = all_struct_names.index(r_name)

                    comp_ediff = w_energies[:, s_id] - w_energies[:, r_id]

                    eng_err = comp_ediff - true_ediff
                    s_grad = mgr_eng_grad[s_id]
                    r_grad = mgr_eng_grad[r_id]

                    eng_grad_vec[:, :, fit_id] += (
                            eng_err[:, np.newaxis] * (s_grad - r_grad) * 2
                    )

                indices = np.where(potential_template.active_mask)[0]
                tmp_eng = eng_grad_vec[:, indices, :]
                tmp_fcs = fcs_grad_vec[:, indices, :]

                gradient = np.dstack([tmp_eng, tmp_fcs]).swapaxes(1, 2)

        return gradient

    # def fd_grad(pop):
    #     h = 1e-4
    #     pop = np.array(pop)
    #     N = pop.shape[0]
    #     cd_points = np.array([pop]*N*2)
    #
    #     for l in range(N):
    #         cd_points[2*l, l] += h
    #         cd_points[2*l+1, l] -= h
    #
    #     cd_evaluated = fxn_wrap(cd_points)
    #     fx = fxn_wrap(np.atleast_2d(pop))
    #
    #     gradient = np.zeros((2, 13))
    #     if is_master:
    #         print(cd_evaluated)
    #         print(cd_evaluated.shape)
    #         for l in range(N):
    #             # gradient[l] = (cd_evaluated[l] - fx) / h
    #             gradient[l] = (cd_evaluated[l] - cd_evaluated[l+1]) / h / 2
    #
    #         gradient = gradient.T
    #
    #     return gradient

    # Have every process build the toolbox
    toolbox, creator = build_ga_toolbox(potential_template)

    toolbox.register("evaluate_population", fxn_wrap)
    toolbox.register("gradient", grad_wrap)
    # toolbox.register("fd_gradient", fd_grad)

    # Create the original population
    if is_master:
        master_pop = toolbox.population(n=POP_SIZE)
        # master_pop = np.ones(np.array(master_pop).shape)
    else:
        master_pop = 0

    # TODO: making array probably destroys Individuals
    master_pop = np.array(master_pop)

    master_pop_shape = world_comm.bcast(master_pop.shape, root=0)

    # @profile
    # def call_grad():
    #     return toolbox.gradient(master_pop)

    init_fit = toolbox.evaluate_population(master_pop)
    # seed = potential_template.pvec.copy()[np.where(potential_template.active_mask)[0]]
    # my_init_grad = toolbox.gradient(np.atleast_2d(seed))
    # fd_init_grad = fd_grad(seed)
    #
    if is_master:
        init_fit = np.sum(init_fit, axis=1)
        print("MASTER: initial (UN-minimized) fitnesses:", init_fit, flush=True)
        print("Average value:", np.average(init_fit), flush=True)
        # print("my_init_grad:", my_init_grad, flush=True)
        # print("fd_init_grad:", fd_init_grad, flush=True)
        #
        # print("my_init_grad.shape:", my_init_grad.shape, flush=True)
        # print("fd_init_grad.shape:", fd_init_grad.shape, flush=True)


    master_pop = local_minimization(
        master_pop, toolbox, world_comm, is_master, nsteps=INIT_NSTEPS
    )

    # new_pop = world_comm.bcast(new_pop, root=0)
    new_fit = toolbox.evaluate_population(master_pop)

    if is_master:
        new_fit = np.sum(new_fit, axis=1)
        print("MASTER: initial (minimized) fitnesses:", new_fit, flush=True)

    # TODO: who has the updated potentials?

    # Have master gather fitnesses and update individuals
    if is_master:
        for ind, fit in zip(master_pop, new_fit):
            ind.fitness.values = fit,

        # Sort population; best on top
        master_pop = tools.selBest(master_pop, len(master_pop))

        print_statistics(master_pop, 0, stats, logbook)

        checkpoint(master_pop, logbook, master_pop[0], 0)
        ga_start = time.time()

    # Begin GA
    if RUN_NEW_GA:
        generation_number = 1
        while (generation_number < NUM_GENS):
            if is_master:

                # TODO: currently using crossover; used to use blend

                # Preserve top 50%, breed survivors
                for pot_num in range(len(master_pop) // 2, len(master_pop)):
                    mom_idx = np.random.randint(len(master_pop) // 2)

                    dad_idx = mom_idx
                    while dad_idx == mom_idx:
                        dad_idx = np.random.randint(len(master_pop) // 2)

                    mom = master_pop[mom_idx]
                    dad = master_pop[dad_idx]

                    kid, _ = toolbox.mate(toolbox.clone(mom),
                                          toolbox.clone(dad))
                    master_pop[pot_num] = kid

                # TODO: debug to make sure pop is always sorted here

                # Mutate randomly everyone except top 10% (or top 2)
                for mut_ind in master_pop[max(2, int(POP_SIZE / 10)):]:
                    if np.random.random() >= MUTPB: toolbox.mutate(mut_ind)
            # else:
            #     master_pop = None

            # Run local minimization on best individual if desired
            if DO_LMIN and (generation_number % LMIN_FREQUENCY == 0):
                master_pop = local_minimization(
                    master_pop, toolbox, world_comm, is_master,
                    nsteps=INTER_NSTEPS
                )

            # Compute fitnesses with mated/mutated/optimized population
            fitnesses = toolbox.evaluate_population(master_pop)

            # Update individuals with new fitnesses
            if is_master:
                new_fit = np.sum(fitnesses, axis=1)

                for ind, fit in zip(master_pop, new_fit):
                    ind.fitness.values = fit,

                # Sort
                master_pop = tools.selBest(master_pop, len(master_pop))

                # Print statistics to screen and checkpoint
                print_statistics(master_pop, generation_number, stats, logbook)

                if (generation_number % CHECKPOINT_FREQUENCY == 0):
                    best = np.array(tools.selBest(master_pop, 1)[0])
                    checkpoint(master_pop, logbook, best, generation_number)

                best_guess = master_pop[0]

            generation_number += 1
    else:
        master_pop = np.genfromtxt(POP_FILE_NAME)
        best_guess = creator.Individual(master_pop[0])

    # Perform a final local optimization on the final results of the GA
    if is_master:
        ga_runtime = time.time() - ga_start

        print("MASTER: GA runtime = {:.2f} (s)".format(ga_runtime), flush=True)
        print("MASTER: Average time per step = {:.2f}"
              " (s)".format(ga_runtime / NUM_GENS), flush=True)


################################################################################

def build_ga_toolbox(potential_template):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray,
                   fitness=creator.CostFunctionMinimizer)

    def ret_pvec(arr_fxn):
        # hard-coded version for pair-pots only
        tmp = arr_fxn(potential_template.generate_random_instance())

        return tmp[np.where(potential_template.active_mask)[0]]

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual, )
    # np.random.random)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.parameter_set, )
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    # toolbox.register("mate", tools.cxBlend, alpha=MATING_ALPHA)
    toolbox.register("mate", tools.cxTwoPoint)

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
def local_minimization(master_pop, toolbox, world_comm, is_master, nsteps=20):
    def lm_fxn_wrap(raveled_pop, original_shape):
        val = toolbox.evaluate_population(
            raveled_pop.reshape(original_shape)
        )

        val = world_comm.bcast(val, root=0)
        return val.ravel()

    def lm_grad_wrap(raveled_pop, original_shape):
        # shape: (num_pots, num_structs*2, num_params)

        grads = toolbox.gradient(
            raveled_pop.reshape(original_shape)
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

        return padded_grad

    # lm_grad_wrap = '2-point'

    master_pop = world_comm.bcast(master_pop, root=0)
    master_pop = np.array(master_pop)

    opt_results = least_squares(
        lm_fxn_wrap, master_pop.ravel(), lm_grad_wrap,
        method='lm', max_nfev=nsteps, args=(master_pop.shape,)
    )

    if is_master:
        new_pop = opt_results['x'].reshape(master_pop.shape)
    else:
        new_pop = None

    org_fits = toolbox.evaluate_population(master_pop)
    new_fits = toolbox.evaluate_population(new_pop)

    if is_master:
        updated_master_pop = list(master_pop)

        for i, ind in enumerate(new_pop):
            if np.sum(new_fits[i]) < np.sum(org_fits[i]):
                updated_master_pop[i] = creator.Individual(new_pop[i])

        master_pop = updated_master_pop

    return master_pop


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


def initialize_potential_template():
    # TODO: BW settings
    # potential_template = Template(
    #     pvec_len=137,
    #     u_ranges = [(-1, 1), (-1, 1)],
    #     spline_ranges=[(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3),
    #                    (-0.5, 1), (-0.5, 1), (-2, 3), (-2, 3), (-7, 2),
    #                    (-7, 2), (-7, 2)],
    #     spline_indices=[(0, 15), (15, 30), (30, 45), (45, 58), (58, 71),
    #                      (71, 77), (77, 83), (83, 95), (95, 107),
    #                      (107, 117), (117, 127), (127, 137)]
    # )
    #
    # mask = np.ones(potential_template.pvec.shape)
    #
    # potential_template.pvec[12] = 0; mask[12] = 0 # rhs phi_A knot
    # potential_template.pvec[14] = 0; mask[14] = 0 # rhs phi_A deriv
    #
    # potential_template.pvec[27] = 0; mask[27] = 0 # rhs phi_B knot
    # potential_template.pvec[29] = 0; mask[29] = 0 # rhs phi_B deriv
    #
    # potential_template.pvec[42] = 0; mask[42] = 0 # rhs phi_B knot
    # potential_template.pvec[44] = 0; mask[44] = 0 # rhs phi_B deriv
    #
    # potential_template.pvec[55] = 0; mask[55] = 0 # rhs rho_A knot
    # potential_template.pvec[57] = 0; mask[57] = 0 # rhs rho_A deriv
    #
    # potential_template.pvec[68] = 0; mask[68] = 0 # rhs rho_B knot
    # potential_template.pvec[70] = 0; mask[70] = 0 # rhs rho_B deriv
    #
    # potential_template.pvec[92] = 0; mask[92] = 0 # rhs f_A knot
    # potential_template.pvec[94] = 0; mask[94] = 0 # rhs f_A deriv
    #
    # potential_template.pvec[104] = 0; mask[104] = 0 # rhs f_B knot
    # potential_template.pvec[106] = 0; mask[106] = 0 # rhs f_B deriv
    #
    # # potential_template.pvec[83:] = 0; mask[83:] = 0 # EAM params only
    # # potential_template.pvec[45:] = 0; mask[45:] = 0 # EAM params only
    # potential_template.pvec[5:] = 0; mask[5:] = 0 # mini params only
    #
    # potential_template.active_mask = mask

    potential = MEAM.from_file(LOAD_PATH + 'TiO.meam.spline')

    x_pvec, seed_pvec, indices = src.meam.splines_to_pvec(
        potential.splines)

    potential_template = Template(
        pvec_len=116,
        u_ranges=[(-1, 1), (-1, 1)],
        spline_ranges=[(-1, 4), (-0.5, 0.5), (-1, 1), (-9, 3), (-30, 15),
                       (-0.5, 1), (-0.2, -0.4), (-2, 3), (-7.5, 12.5),
                       (-8, 2), (-1, 1), (-1, 0.2)],
        spline_indices=[(0, 15), (15, 22), (22, 37), (37, 50), (50, 57),
                        (57, 63), (63, 70), (70, 82), (82, 89),
                        (89, 99), (99, 106), (106, 116)]
    )

    potential_template.pvec = seed_pvec.copy()
    mask = np.ones(potential_template.pvec_len)

    mask[:15] = 0 # phi_Ti

    potential_template.pvec[19] = 0;
    mask[19] = 0  # rhs phi_TiO knot
    potential_template.pvec[21] = 0;
    mask[21] = 0  # rhs phi_TiO deriv

    mask[22:37] = 0  # phi_O
    mask[37:50] = 0  # rho_Ti

    potential_template.pvec[54] = 0;
    mask[54] = 0  # rhs rho_O knot
    potential_template.pvec[56] = 0;
    mask[56] = 0  # rhs rho_O deriv

    mask[57:63] = 0  # U_Ti
    mask[70:82] = 0  # f_Ti

    potential_template.pvec[86] = 0;
    mask[86] = 0  # rhs f_O knot
    potential_template.pvec[88] = 0;
    mask[88] = 0  # rhs f_O deriv

    mask[89:99] = 0  # g_Ti
    mask[106:116] = 0  # g_O

    potential_template.active_mask = mask

    return potential_template


def minimize_population(pop, toolbox, comm, mpi_size, max_nsteps):
    my_indivs = comm.scatter(np.array_split(pop, mpi_size))

    new_indivs = []
    my_fitnesses = []

    for indiv in my_indivs:
        opt_results = least_squares(toolbox.evaluate_population, indiv,
                                    toolbox.gradient, method='lm',
                                    max_nfev=max_nsteps)

        indiv = creator.Individual(opt_results['x'])

        new_indivs.append(creator.Individual(indiv))

        fitnesses = np.sum(toolbox.evaluate_population(indiv))
        my_fitnesses.append(fitnesses)

    return np.vstack(comm.gather(pop, root=0)), \
           np.concatenate(comm.gather(my_fitnesses, root=0))


def split_population(a, n):
    """
    Stackoverflow credits (User = "tixxit"):
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/37414115
    """
    k, m = divmod(len(a), n)
    return list(
        a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def build_fxn_wrapper(eval_fxn, is_master, is_manager, is_node_head,
                      world_comm, manager_comm, head_comm, node_comm):
    def fxn_wrap(full, world_rank=None, procs_get_same_pop=False):
        """
        Args:
            full (list[Individual]): all potential_templates
            procs_get_same_pop (bool): True if population NOT scattered

        Notes:
        Evaluates the function and gathers the results at every level of
        parallelization.

        Process -- evaluates on subset of subset of database with subset of pop
        Node -- gathers full population for subset of subset of database
        Manager -- gathers full population for subset of database
        Master -- gathers full population for full database
        """

        full = np.atleast_2d(full)

        was_split = False
        if is_node_head:
            if procs_get_same_pop:
                inp = head_comm.bcast(full, root=0)
            else:
                split_full = split_population(full, nodes_per_manager)
                inp = head_comm.scatter(split_full, root=0)
                was_split = True
        else:
            inp = None

        inp = node_comm.bcast(inp, root=0)

        # evaluate subset of population on subset OF SUBSET of database
        cost_eng, cost_fcs = eval_fxn(inp)

        # subset of population on node's subset of database
        node_eng_costs = node_comm.gather(cost_eng, root=0)
        node_fcs_costs = node_comm.gather(cost_fcs, root=0)

        if is_node_head:
            node_eng_costs = np.concatenate(node_eng_costs, axis=1)
            node_fcs_costs = np.concatenate(node_fcs_costs, axis=1)

            # full population on manager's subset of database
            head_nodes_eng = head_comm.gather(node_eng_costs, root=0)
            head_nodes_fcs = head_comm.gather(node_fcs_costs, root=0)

        if is_manager:
            if procs_get_same_pop:
                head_nodes_eng = np.array(head_nodes_eng)
                head_nodes_fcs = np.array(head_nodes_fcs)

                head_nodes_eng = np.sum(head_nodes_eng, axis=0)
                head_nodes_fcs = np.sum(head_nodes_fcs, axis=0)

            else:
                head_nodes_eng = np.vstack(head_nodes_eng)
                head_nodes_fcs = np.vstack(head_nodes_fcs)

                # full population on full database
            all_eng_costs = manager_comm.gather(head_nodes_eng, root=0)
            all_fcs_costs = manager_comm.gather(head_nodes_fcs, root=0)

        if is_master:
            all_eng_costs = np.concatenate(all_eng_costs, axis=1)
            all_fcs_costs = np.concatenate(all_fcs_costs, axis=1)

            value = np.concatenate([all_eng_costs, all_fcs_costs], axis=1)
        else:
            value = None

        if is_manager:
            value = manager_comm.bcast(value, root=0)

        if is_node_head:
            value = head_comm.bcast(value, root=0)

        value = world_comm.bcast(value, root=0)

        if is_master:
            print(np.sum(value), flush=True)

        return value


def build_grad_wrapper(grad_fxn, is_master, is_manager, is_node_head,
                       world_comm, manager_comm, head_comm, node_comm):
    def grad_wrap(full, procs_get_same_pop=False):

        full = np.atleast_2d(full)

        if is_node_head:
            if procs_get_same_pop:
                inp = full
            else:
                split_full = split_population(full, nodes_per_manager)
                inp = head_comm.scatter(split_full, root=0)
        else:
            inp = None

        inp = node_comm.bcast(inp, root=0)

        eng_grad_val, fcs_grad_val = grad_fxn(inp)

        # evaluate subset of population on subset OF SUBSET of database
        node_eng_grad = node_comm.gather(eng_grad_val, root=0)
        node_fcs_grad = node_comm.gather(fcs_grad_val, root=0)

        if is_node_head:
            node_eng_grad = np.dstack(node_eng_grad)
            node_fcs_grad = np.dstack(node_fcs_grad)

            # print('node_eng_grad.shape', node_eng_grad.shape, flush=True)

            # full population on manager's subset of database
            head_nodes_eng_grad = head_comm.gather(node_eng_grad, root=0)
            head_nodes_fcs_grad = head_comm.gather(node_fcs_grad, root=0)

        if is_manager:
            if procs_get_same_pop:
                head_nodes_eng_grad = np.array(head_nodes_eng_grad)
                head_nodes_fcs_grad = np.array(head_nodes_fcs_grad)

                head_nodes_eng_grad = np.sum(head_nodes_eng_grad, axis=0)
                head_nodes_fcs_grad = np.sum(head_nodes_fcs_grad, axis=0)
            else:
                head_nodes_eng_grad = np.dstack(head_nodes_eng_grad)
                head_nodes_fcs_grad = np.dstack(head_nodes_fcs_grad)

            # full population on full database
            all_eng_grad = manager_comm.gather(head_nodes_eng_grad, root=0)
            all_fcs_grad = manager_comm.gather(head_nodes_fcs_grad, root=0)

        if is_master:
            all_eng_grad = np.dstack(all_eng_grad)
            all_fcs_grad = np.dstack(all_fcs_grad)

            # print('all_eng_grad.shape', all_eng_grad.shape, flush=True)

            # all_grads = all_eng_grad + all_fcs_grad # list join

            grad = np.dstack([all_eng_grad, all_fcs_grad])

            # print('grad.shape', grad.shape, flush=True)
        else:
            grad = None

        if is_manager:
            grad = manager_comm.bcast(grad, root=0)

        if is_node_head:
            grad = head_comm.bcast(grad, root=0)

        grad = node_comm.bcast(grad, root=0)
        # print('grad.shape', grad.shape, flush=True)

        return grad.T


################################################################################

if __name__ == "__main__":
    main()
