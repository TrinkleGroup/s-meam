import os
# TODO: BW settings
os.chdir("/mnt/c/Users/jvita/scripts/s-meam/")
import sys

sys.path.append('./')

import numpy as np
import random

np.set_printoptions(precision=8, linewidth=np.inf, suppress=True)

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
from mpi4py import MPI

from deap import base, creator, tools, algorithms

import src.meam
from src.meam import MEAM
from src.worker import Worker
from src.meam import MEAM
from src.spline import Spline
from src.database import Database
from src.potential_templates import Template
from src.node import Node

################################################################################
"""MPI settings"""

MASTER_RANK = 0

if len(sys.argv) > 1:
    num_managers = int(sys.argv[1]) # how many subsets to break the database into
    nodes_per_manager = int(sys.argv[2]) # number of available compute nodes
    procs_per_node = int(sys.argv[3])
else:
    num_managers = 2
    nodes_per_manager = 2
    procs_per_node = 2

num_nodes = nodes_per_manager * num_managers

################################################################################
"""MEAM potential settings"""

NTYPES = 2
# ACTIVE_SPLINES = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ACTIVE_SPLINES = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

################################################################################
"""GA settings"""

if len(sys.argv) > 4:
    POP_SIZE = int(sys.argv[4])
else:
    POP_SIZE = 4

NUM_GENS = 1
CXPB = 1.0

if len(sys.argv) > 5:
    MUTPB = float(sys.argv[5])
else:
    MUTPB = 0.5

RUN_NEW_GA = True

DO_LMIN = False
LMIN_FREQUENCY = 1
INIT_NSTEPS = 30
INTER_NSTEPS = 10
FINAL_NSTEPS = 4

CHECKPOINT_FREQUENCY = 1

MATING_ALPHA = 0.2

################################################################################
"""I/O settings"""

date_str = datetime.datetime.now().strftime("%Y-%m-%d")

CHECK_BEFORE_OVERWRITE = False

# TODO: BW settings

# LOAD_PATH = "data/fitting_databases/fixU/"
LOAD_PATH = "../data/fitting_databases/leno-redo/"
# LOAD_PATH = "/projects/sciteam/baot/fixU-clean/"
SAVE_PATH = "../data/results/"

SAVE_DIRECTORY = SAVE_PATH + date_str + "-" + "meam" + "{}-{}".format(NUM_GENS,
                                                                      MUTPB)

if os.path.isdir(SAVE_DIRECTORY):
    SAVE_DIRECTORY = SAVE_DIRECTORY + '-' + str(np.random.randint(100000))

DB_PATH = LOAD_PATH + 'structures'
DB_INFO_FILE_NAME = LOAD_PATH + 'rhophi/info'
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

        # TODO: have each manager read in its subset
        master_database = Database(DB_PATH, DB_INFO_FILE_NAME)
        master_database.print_metadata()

        manager_subsets, _, _ = group_database_subsets(
            master_database, num_managers
        )

        # TODO: each structure should READ the worker, not get passed it
        # currently, this will probably require twice as much mem (during scat)

    else:
        manager_subsets = None
        potential_template = None

    potential_template = world_comm.bcast(potential_template, root=0)

    rank_list = np.arange(world_size, dtype=int)

    manager_ranks = rank_list[::procs_per_node*nodes_per_manager]

    world_group = world_comm.Get_group()

    # manager_comm connects all manager processes
    manager_group = world_group.Incl(manager_ranks)
    manager_comm = world_comm.Create(manager_group)

    manager_color = world_rank // (procs_per_node * nodes_per_manager)

    # head_comm communicates with all node heads of ONE manager
    start = manager_color * procs_per_node * nodes_per_manager
    stop = start + (procs_per_node * nodes_per_manager)

    head_ranks = rank_list[start:stop:procs_per_node]
    head_group = world_group.Incl(head_ranks)
    head_comm = world_comm.Create(head_group)

    # node comm links all processes corresponding to a single node
    node_color = world_rank // procs_per_node
    node_comm = world_comm.Split(node_color, world_rank)

    node_rank = node_comm.Get_rank()

    is_manager = (manager_comm != MPI.COMM_NULL)

    if is_manager:
        manager_rank = manager_comm.Get_rank()
        manager_subset = manager_comm.scatter(manager_subsets, root=0)

        print(
            "Manager", manager_rank, "received",
            len(manager_subset.structures), "structures", flush=True
        )
    else:
        manager_subset = None

    is_node_head = (head_comm != MPI.COMM_NULL)

    if is_node_head:
        head_rank = head_comm.Get_rank()
        head_copy = head_comm.bcast(manager_subset, root=0)

        print(
            "Node", head_rank, "received", len(head_copy.structures),
            "structures", flush=True
        )

        proc_subset, _, _ = group_database_subsets(head_copy, procs_per_node)
    else:
        proc_subset = None

    database = node_comm.scatter(proc_subset, root=0)

    # TODO: number of managers s.t. subset size fits on one node

    # Have every process build the toolbox
    toolbox, creator = build_ga_toolbox(potential_template)

    eval_fxn, grad_fxn = build_evaluation_functions(
        database, potential_template
    )

    # TODO: have multiple "minimizer" processes doing LM over multiple pots?
    # TODO: managers and head nodes may have some duplicate structs

    def fxn_wrap(inp):
        """
        Evaluates the function and gathers the results at every level of
        parallelization.

        Process -- evaluates on subset of subset of database with subset of pop
        Node -- gathers full population for subset of subset of database
        Manager -- gathers full population for subset of database
        Master -- gathers full population for full database
        """

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
            head_nodes_eng = np.vstack(head_nodes_eng)
            head_nodes_fcs = np.vstack(head_nodes_fcs)

            # full population on full database
            all_eng_costs = manager_comm.gather(head_nodes_eng, root=0)
            all_fcs_costs = manager_comm.gather(head_nodes_fcs, root=0)

        if is_master:
            all_eng_costs = np.concatenate(all_eng_costs, axis=1)
            all_fcs_costs = np.concatenate(all_fcs_costs, axis=1)

            all_costs = all_eng_costs + all_fcs_costs # list join

            value = np.concatenate(all_costs)
        else:
            value = None

        if is_manager:
            value = manager_comm.bcast(value, root=0)

        if is_node_head:
            value = head_comm.bcast(value, root=0)

        value = node_comm.bcast(value, root=0)

        return value

    def grad_wrap(inp):

        eng_grad_val, fcs_grad_val = grad_fxn(inp)
        # eng_grad_val = eng_grad_val.T
        # fcs_grad_val = fcs_grad_val.T

        # evaluate subset of population on subset OF SUBSET of database
        node_eng_grad = node_comm.gather(eng_grad_val, root=0)
        node_fcs_grad = node_comm.gather(fcs_grad_val, root=0)

        if is_node_head:
            print([el.shape for el in node_eng_grad], flush=True)
            node_eng_grad = np.dstack(node_eng_grad)
            node_fcs_grad = np.dstack(node_fcs_grad)

            print('node_eng_grad.shape', node_eng_grad.shape, flush=True)

            # full population on manager's subset of database
            head_nodes_eng_grad = head_comm.gather(node_eng_grad, root=0)
            head_nodes_fcs_grad = head_comm.gather(node_fcs_grad, root=0)

        if is_manager:
            # print([el.shape for el in head_nodes_eng_grad], flush=True)
            head_nodes_eng_grad = np.vstack(head_nodes_eng_grad)
            head_nodes_fcs_grad = np.vstack(head_nodes_fcs_grad)

            print('head_nodes_eng_grad.shape', head_nodes_eng_grad.shape, flush=True)

            # full population on full database
            all_eng_grad = manager_comm.gather(head_nodes_eng_grad, root=0)
            all_fcs_grad = manager_comm.gather(head_nodes_fcs_grad, root=0)

        # TODO: should all of these should be concatenates?
        # TODO: need some kind of ordering?

        if is_master:
            # print([el.shape for el in all_eng_grad], flush=True)
            all_eng_grad = np.concatenate(all_eng_grad, axis=1)
            all_fcs_grad = np.concatenate(all_fcs_grad, axis=1)

            print('all_eng_grad.shape', all_eng_grad.shape, flush=True)

            all_grads = all_eng_grad + all_fcs_grad # list join

            grad = np.concatenate(all_grads)

            print('grad.shape', grad.shape, flush=True)
        else:
            grad = None

        if is_manager:
            grad = manager_comm.bcast(grad, root=0)

        if is_node_head:
            grad = head_comm.bcast(grad, root=0)

        grad = node_comm.bcast(grad, root=0)
        print('grad.shape', grad.shape, flush=True)

        return grad

    toolbox.register("evaluate_population", fxn_wrap)
    toolbox.register("gradient", grad_wrap)

    # Create the original population
    if is_master:
        pop = toolbox.population(n=POP_SIZE)
    else:
        pop = None

    # Send full population to managers
    if is_manager:
        pop = manager_comm.bcast(pop, root=0)
        pop = np.array_split(np.array(pop), procs_per_node)
    else:
        pop = None

    # Managers split population across their nodes
    if is_node_head:
        pop = head_comm.scatter(pop, root=0)
        print(
            "Head node", head_rank, "received", pop.shape[0], "potentials",
            flush=True
        )
    else:
        pop = None

    # Nodes broadcast population subset to all of their processes
    pop = node_comm.bcast(pop, root=0)

    minimized_fitnesses = fxn_wrap(pop)

    for ind_num, indiv in enumerate(pop):
        if is_master:
            print(
                "Minimizing potential %d/%d" % (ind_num, len(pop)), flush=True
            )

        opt_results = least_squares(
            fxn_wrap, indiv, grad_wrap, method='lm', max_nfev=INTER_NSTEPS * 2
        )

        opt_indiv = creator.Individual(opt_results)

        opt_fitness = np.sum(fxn_wrap(opt_indiv))
        prev_fitness = np.sum(fxn_wrap(indiv))

        if opt_fitness < prev_fitness:
            pop[ind_num] = opt_indiv
            minimized_fitnesses[ind_num] = opt_fitness
        else:
            minimized_fitnesses[ind_num] = prev_fitness

    # Have master gather fitnesses and update individuals
    if is_master:
        all_fitnesses = np.vstack(all_fitnesses)
        all_fitnesses = all_fitnesses.ravel()
        print("MASTER: initial fitnesses:", all_fitnesses, flush=True)

        join_pop = []
        for slave_pop in pop:
            for ind in slave_pop:
                join_pop.append(ind)

        pop = join_pop

        for ind, fit in zip(pop, all_fitnesses):
            ind.fitness.values = fit,

        # Sort population; best on top
        pop = tools.selBest(pop, len(pop))

        print_statistics(pop, 0, stats, logbook)

        checkpoint(pop, logbook, pop[0], 0)
        ga_start = time.time()

    # Begin GA
    if RUN_NEW_GA:
        i = 1
        while (i < NUM_GENS):
            if is_master:

                # TODO: currently using crossover; used to use blend

                # Preserve top 50%, breed survivors
                for j in range(len(pop) // 2, len(pop)):
                    mom_idx = np.random.randint(len(pop) // 2)

                    dad_idx = mom_idx
                    while dad_idx == mom_idx:
                        dad_idx = np.random.randint(len(pop) // 2)

                    mom = pop[mom_idx]
                    dad = pop[dad_idx]

                    kid, _ = toolbox.mate(toolbox.clone(mom),
                                          toolbox.clone(dad))
                    pop[j] = kid

                # TODO: debug to make sure pop is always sorted here

                # Mutate randomly everyone except top 10% (or top 2)
                for mut_ind in pop[max(2, int(POP_SIZE / 10)):]:
                    if np.random.random() >= MUTPB: toolbox.mutate(mut_ind)

                pop = np.array_split(pop, mpi_size)
            else:
                pop = None

            # Send out updated population
            my_indivs = comm.scatter(pop, root=0)

            # Run local minimization on best individual if desired
            if DO_LMIN and (i % LMIN_FREQUENCY == 0):
                new_indivs = []

                for indiv in my_indivs:
                    opt_results = local_minimization(
                        indiv, toolbox, is_master, comm, num_steps=None
                    )

                    # opt_results = least_squares(toolbox.evaluate_population, indiv,
                    #                             toolbox.gradient, method='lm',
                    #                             max_nfev=INTER_NSTEPS * 2)

                    opt_indiv = creator.Individual(opt_results)

                    opt_fitness = np.sum(toolbox.evaluate_population(opt_indiv))
                    prev_fitness = np.sum(toolbox.evaluate_population(indiv))

                    if opt_fitness < prev_fitness:
                        indiv = opt_indiv

                    new_indivs.append(creator.Individual(indiv))

                my_indivs = new_indivs

            # Compute fitnesses with mated/mutated/optimized population
            fitnesses = np.sum(toolbox.evaluate_population(indiv))

            all_fitnesses = comm.gather(fitnesses, root=0)
            pop = comm.gather(my_indivs, root=0)

            # Update individuals with new fitnesses
            if is_master:
                all_fitnesses = np.vstack(all_fitnesses)
                all_fitnesses = all_fitnesses.ravel()

                join_pop = []
                for slave_pop in pop:
                    for ind in slave_pop:
                        join_pop.append(creator.Individual(ind))

                pop = join_pop

                for ind, fit in zip(pop, all_fitnesses):
                    ind.fitness.values = fit,

                # Sort
                pop = tools.selBest(pop, len(pop))

                # Print statistics to screen and checkpoint
                print_statistics(pop, i, stats, logbook)

                if (i % CHECKPOINT_FREQUENCY == 0):
                    best = np.array(tools.selBest(pop, 1)[0])
                    checkpoint(pop, logbook, best, i)

                best_guess = pop[0]

            i += 1
    else:
        pop = np.genfromtxt(POP_FILE_NAME)
        best_guess = creator.Individual(pop[0])

    # Perform a final local optimization on the final results of the GA
    if is_master:
        ga_runtime = time.time() - ga_start

        print("MASTER: GA runtime = {:.2f} (s)".format(ga_runtime), flush=True)
        print("MASTER: Average time per step = {:.2f}"
              " (s)".format(ga_runtime / NUM_GENS), flush=True)

        # best_fitness = np.sum(toolbox.evaluate_population(best_guess))

    new_indivs = []
    for indiv in my_indivs:
        print("SLAVE: Rank", rank, "performing final minimization ... ", flush=True)
        opt_results = local_minimization(
            indiv, toolbox, is_master, comm, num_steps=None
        )

        # opt_results = least_squares(toolbox.evaluate_population, indiv,
        #                             toolbox.gradient, method='lm',
        #                             max_nfev=INTER_NSTEPS * 2)

        final = creator.Individual(opt_results)

        new_indivs.append(final)

    print("SLAVE: Rank", rank, "minimized fitness:",
          np.sum(toolbox.evaluate_population(final)))

    fitnesses = np.sum(toolbox.evaluate_population(final))

    all_fitnesses = comm.gather(fitnesses, root=0)
    pop = comm.gather(final, root=0)

    # Save final results
    if is_master:
        print("MASTER: final fitnesses:", all_fitnesses, flush=True)
        all_fitnesses = np.vstack(all_fitnesses)
        all_fitnesses = all_fitnesses.ravel()

        join_pop = []
        for slave_pop in pop:
            for ind in slave_pop:
                join_pop.append(creator.Individual(ind))

        pop = join_pop

        for ind, fit in zip(pop, all_fitnesses):
            ind.fitness.values = fit,

        # Sort
        pop = tools.selBest(pop, len(pop))

        # Print statistics to screen and checkpoint
        print_statistics(pop, i, stats, logbook)

        best = np.array(tools.selBest(pop, 1)[0])

        recheck = np.sum(toolbox.evaluate_population(best))

        print("MASTER: confirming best fitness:", recheck)

        final_arr = np.array(best)

        checkpoint(pop, logbook, final_arr, i)
        np.savetxt(SAVE_DIRECTORY + '/final_potential.dat', final_arr)

        # plot_best_individual()


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


def build_evaluation_functions(database, potential_template):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn(pot):
        full = np.atleast_2d(pot)
        full = potential_template.insert_active_splines(full)

        w_energies = np.zeros((full.shape[0], len(database.structures)))
        t_energies = np.zeros(len(database.structures))

        fcs_fitnesses = np.zeros((full.shape[0], len(database.structures)))

        # TODO: reference energy needs to be sent from master
        # ref_energy = 0

        keys = sorted(list(database.structures.keys()))
        for j, name in enumerate(keys):

            w = database.structures[name]

            w_energies[:, j] = w.compute_energy(full, potential_template.u_ranges)
            t_energies[j] = database.true_energies[name]

            # if name == database.reference_struct:
            #     ref_energy = w_energies[j]

            w_fcs = w.compute_forces(full, potential_template.u_ranges)
            true_fcs = database.true_forces[name]

            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
            fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

            fcs_fitnesses[:, j] = fcs_err

        # w_energies -= ref_energy
        # t_energies -= database.reference_energy

        eng_fitnesses = np.zeros((full.shape[0], len(database.structures)))

        for j, (w_eng, t_eng) in enumerate(zip(w_energies.T, t_energies)):
            eng_fitnesses[:, j] = (w_eng - t_eng) ** 2

        fitnesses = np.concatenate([eng_fitnesses, fcs_fitnesses])

        # print("Fitness shape:", fitnesses.shape, flush=True)

        return eng_fitnesses, fcs_fitnesses

    def grad(pot):
        full = np.atleast_2d(pot)
        full = potential_template.insert_active_splines(full)

        fcs_grad_vec = np.zeros(
            (full.shape[0], len(database.structures),
            len(potential_template.pvec))
        )

        w_energies = np.zeros((full.shape[0], len(database.structures)))
        t_energies = np.zeros(len(database.structures))

        # ref_energy = 0

        keys = sorted(list(database.structures.keys()))
        for j, name in enumerate(keys):
            w = database.structures[name]

            w_energies[:, j] = w.compute_energy(full, potential_template.u_ranges)
            t_energies[j] = database.true_energies[name]

            # if name == database.reference_struct:
            #     ref_energy = w_energies[j]

            w_fcs = w.compute_forces(full, potential_template.u_ranges)
            true_fcs = database.true_forces[name]

            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10))

            fcs_grad = w.forces_gradient_wrt_pvec(full, potential_template.u_ranges)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            fcs_grad_vec[:, j] += (2 * summed / 10).ravel()

        # w_energies -= ref_energy
        # t_energies -= database.reference_energy

        eng_grad_vec = np.zeros(
            (full.shape[0], len(database.structures),
            len(potential_template.pvec))
        )

        for j, (name, w_eng, t_eng) in enumerate(
                zip(keys, w_energies, t_energies)):

            w = database.structures[name]

            eng_err = (w_eng - t_eng)
            eng_grad = w.energy_gradient_wrt_pvec(full, potential_template.u_ranges)

            print(j, eng_grad_vec.shape, eng_grad.shape, flush=True)
            eng_grad_vec[:, j, :] += (eng_err * eng_grad * 2)

        tmp_eng = eng_grad_vec[:, :, np.where(potential_template.active_mask)[0]]
        tmp_fcs = fcs_grad_vec[:, :, np.where(potential_template.active_mask)[0]]

        return tmp_eng, tmp_fcs

    return fxn, grad


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


def local_minimization(guess, toolbox, is_master, comm, num_steps=None,):
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
            args=(stop,), max_nfev=num_steps*2
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
    potential_template = Template(
        pvec_len=137,
        u_ranges = [(-1, 1), (-1, 1)],
        spline_ranges=[(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3),
                       (-0.5, 1), (-0.5, 1), (-2, 3), (-2, 3), (-7, 2),
                       (-7, 2), (-7, 2)],
        spline_indices=[(0, 15), (15, 30), (30, 45), (45, 58), (58, 71),
                         (71, 77), (77, 83), (83, 95), (95, 107),
                         (107, 117), (117, 127), (127, 137)]
    )

    mask = np.ones(potential_template.pvec.shape)

    potential_template.pvec[12] = 0; mask[12] = 0 # rhs phi_A knot
    potential_template.pvec[14] = 0; mask[14] = 0 # rhs phi_A deriv

    potential_template.pvec[27] = 0; mask[27] = 0 # rhs phi_B knot
    potential_template.pvec[29] = 0; mask[29] = 0 # rhs phi_B deriv

    potential_template.pvec[42] = 0; mask[42] = 0 # rhs phi_B knot
    potential_template.pvec[44] = 0; mask[44] = 0 # rhs phi_B deriv

    potential_template.pvec[55] = 0; mask[55] = 0 # rhs rho_A knot
    potential_template.pvec[57] = 0; mask[57] = 0 # rhs rho_A deriv

    potential_template.pvec[68] = 0; mask[68] = 0 # rhs rho_B knot
    potential_template.pvec[70] = 0; mask[70] = 0 # rhs rho_B deriv

    potential_template.pvec[92] = 0; mask[92] = 0 # rhs f_A knot
    potential_template.pvec[94] = 0; mask[94] = 0 # rhs f_A deriv

    potential_template.pvec[104] = 0; mask[104] = 0 # rhs f_B knot
    potential_template.pvec[106] = 0; mask[106] = 0 # rhs f_B deriv

    # potential_template.pvec[83:] = 0; mask[83:] = 0 # EAM params only
    potential_template.pvec[45:] = 0; mask[45:] = 0 # EAM params only

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

    return np.vstack(comm.gather(pop, root=0)),\
           np.concatenate(comm.gather(my_fitnesses, root=0))

def compute_relative_weights(database):
    work_weights = []

    name_list = list(database.structures.keys())
    name_list = sorted(name_list)

    for name in name_list:
        work_weights.append(database.structures[name].natoms)

    work_weights = np.array(work_weights)
    work_weights = work_weights / np.min(work_weights)
    work_weights = work_weights*work_weights # cost assumed to scale as N^2

    return work_weights, name_list


def group_database_subsets(database, mpi_size):
    """Groups workers based on evaluation time to help with load balancing.

    Returns:
        distributed_work (list): the partitioned work
        work_per_proc (float): approximate work done by each processor
    """
    # TODO: record scaling on first run, then redistribute to load balance

    unassigned_structs = sorted(list(database.structures.keys()))

    work_weights, name_list = compute_relative_weights(database)

    work_per_proc = np.sum(work_weights) / mpi_size

    work_weights = work_weights.tolist()

    assignments = []
    grouped_work = []

    for _ in range(mpi_size):
        cumulated_work = 0

        names = []

        while unassigned_structs and (cumulated_work < work_per_proc):
            names.append(unassigned_structs.pop())
            cumulated_work += work_weights.pop()

        mini_database = Database.manual_init(
            {name: database.structures[name] for name in names},
            {name: database.true_energies[name] for name in names},
            {name: database.true_forces[name] for name in names},
            {name: database.weights[name] for name in names},
            database.reference_struct,
            database.reference_energy
        )

        assignments.append(mini_database)
        grouped_work.append(cumulated_work)

    return assignments, grouped_work, work_per_proc


################################################################################

if __name__ == "__main__":
    main()
