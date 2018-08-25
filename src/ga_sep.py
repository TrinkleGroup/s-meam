import os
# os.chdir("/home/jvita/scripts/s-meam/project/")
import sys
sys.path.append('./')

import numpy as np
import random
np.set_printoptions(precision=8, linewidth=np.inf, suppress=True)
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
from scipy.optimize import fmin_powell, fmin_cg, least_squares
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
"""GA settings"""

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()

POP_SIZE = mpi_size
NUM_GENS = 1
CXPB = 1.0

if len(sys.argv) > 1:
    MUTPB = float(sys.argv[1])
else:
    MUTPB = 0.5

RUN_NEW_GA = True

DO_LMIN = True
LMIN_FREQUENCY = 1
INIT_NSTEPS = 1
INTER_NSTEPS = 1
FINAL_NSTEPS = 1

CHECKPOINT_FREQUENCY = 1

################################################################################ 
# Record MPI settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

POP_SIZE = mpi_size

is_master_node = (rank == MASTER_RANK)

################################################################################ 
"""I/O settings"""

date_str = datetime.datetime.now().strftime("%Y-%m-%d")

CHECK_BEFORE_OVERWRITE = False

# TODO: BW settings

LOAD_PATH = "data/fitting_databases/leno-redo/"
# LOAD_PATH = "/projects/sciteam/baot/leno-redo/"
SAVE_PATH = "data/ga_results/"
SAVE_DIRECTORY = SAVE_PATH + date_str + "-" + "sep" + "{}-{}".format(NUM_GENS, MUTPB)

DB_PATH = LOAD_PATH + 'structures/'
DB_INFO_FILE_NAME = LOAD_PATH + 'full/info'
POP_FILE_NAME = SAVE_DIRECTORY + "/pop.dat"
LOG_FILE_NAME = SAVE_DIRECTORY + "/ga.log"
TRACE_FILE_NAME = SAVE_DIRECTORY + "/trace.dat"

################################################################################ 

def main():
    if is_master_node:
        print_settings()

        print("MASTER: Preparing save directory/files ... ", flush=True)
        prepare_save_directory()

    if is_master_node:
        print("MASTER: running EAM GA", flush=True)
    eam_results = do_a_ga(83, 'eam')

    if is_master_node:
        print("MASTER: running MEAM GA", flush=True)
    meam_results = do_a_ga(54, 'meam')

    if is_master_node:
        final_results = np.hstack([eam_results, meam_results])

        np.savetxt(SAVE_DIRECTORY + '/full_final_potential.dat', final_results)

def do_a_ga(pvec_len, name, thresh=1e-8):
    if is_master_node:
        # Initialize database and variables to prepare for GA
        f = open(TRACE_FILE_NAME, 'ab')
        f.close()

        stats, logbook = build_stats_and_log()

        print("MASTER: Loading structures ...", flush=True)
        structures, weights = load_structures_on_master()

        print("MASTER: Loading energy/forces database ... ", flush=True)
        true_forces, true_energies = load_true_values(structures.keys())

        print("MASTER: Determining potential information ...", flush=True)
        ex_struct = structures[list(structures.keys())[0]]
        type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
        pvec_len = ex_struct.len_param_vec

        print("MASTER: Preparing to send structures to slaves ... ", flush=True)
    else:
        spline_indices = None
        structures = None
        weights = None
        true_forces = None
        true_energies = None
        pvec_len = None

    # Send all necessary information to slaves
    spline_indices = comm.bcast(spline_indices, root=0)
    pvec_len = comm.bcast(pvec_len, root=0)
    structures = comm.bcast(structures, root=0)
    weights = comm.bcast(weights, root=0)
    true_forces = comm.bcast(true_forces, root=0)
    true_energies = comm.bcast(true_energies, root=0)

    # Have every process build the toolbox
    toolbox, creator = build_ga_toolbox(pvec_len, spline_indices)

    # eval_fxn, min_fxn, grad_fxn = build_evaluation_functions(structures, weights,
    eval_fxn, grad_fxn = build_evaluation_functions(structures, weights,
                                                    true_forces, true_energies,
                                                    spline_indices)

    eam_eval_fxn, eam_grad_fxn = build_evaluation_functions(structures, weights,
                                                    true_forces, true_energies,
                                                    spline_indices)

    meam_eval_fxn, meam_grad_fxn = build_evaluation_functions(structures, weights,
                                                    true_forces, true_energies,
                                                    spline_indices)

    if name == 'eam':
        eval_fxn = eam_eval_fxn
        grad_fxn = eam_grad_fxn
    elif name == 'meam':
        eval_fxn = meam_eval_fxn
        grad_fxn = meam_grad_fxn

    toolbox.register("evaluate_population", eval_fxn)
    toolbox.register("gradient", grad_fxn)

    # Compute initial fitnesses
    if is_master_node:
        pop = toolbox.population(n=POP_SIZE)
    else:
        pop = None

    # print("SLAVE: Rank", rank, "performing initial LMIN ...", flush=True)
    indiv = comm.scatter(pop, root=0)
    opt_results = least_squares(toolbox.evaluate_population, indiv,
                                toolbox.gradient, method='lm',
                                max_nfev=INIT_NSTEPS)

    indiv = creator.Individual(opt_results['x'])

    fitnesses = np.sum(toolbox.evaluate_population(indiv))
    # fitnesses = np.sum(min_fxn(indiv))

    print("SLAVE: Rank", rank, "minimized fitness:", fitnesses, flush=True)

    all_fitnesses = comm.gather(fitnesses, root=0)
    pop = comm.gather(indiv, root=0)

    # Have master gather fitnesses and update individuals
    if is_master_node:
        print("MASTER: received fitnesses:", all_fitnesses, flush=True)
        all_fitnesses = np.vstack(all_fitnesses)
        print(all_fitnesses)

        for ind,fit in zip(pop, all_fitnesses):
            ind.fitness.values = fit,

        # Sort population; best on top
        pop = tools.selBest(pop, len(pop))

        best_fit = np.sum(toolbox.evaluate_population(pop[0]))

        print_statistics(pop, 0, stats, logbook)

        checkpoint(pop, logbook, pop[0], 0)
        ga_start = time.time()
    else:
        best_fit = None

    best_fit = comm.bcast(best_fit, root=0)

    # Begin GA
    if RUN_NEW_GA:
        i = 1
        while (i < NUM_GENS) and (best_fit > thresh):
            if is_master_node:

                # TODO: fit EAM first, then ffg

                # Preserve top 50%, breed survivors
                for j in range(len(pop)//2, len(pop)):
                    # mom = pop[np.random.randint(len(pop)//2)]
                    # dad = pop[j]
                    mom_idx = np.random.randint(len(pop)//2)
                    
                    dad_idx = mom_idx
                    while dad_idx == mom_idx:
                        dad_idx = np.random.randint(len(pop)//2)

                    mom = pop[mom_idx]
                    dad = pop[dad_idx]

                    kid,_ = toolbox.mate(toolbox.clone(mom), toolbox.clone(dad))
                    pop[j] = kid

                # TODO: debug to make sure pop is always sorted here

                # Mutate randomly everyone except top 10% (or top 2)
                for mut_ind in pop[max(2, int(POP_SIZE/10)):]:
                    if np.random.random() >= MUTPB: toolbox.mutate(mut_ind)
            else:
                pop = None

            # Send out updated population
            indiv = comm.scatter(pop, root=0)

            # Run local minimization on best individual if desired
            if DO_LMIN and (i % LMIN_FREQUENCY == 0):
                # if is_master_node:
                    # print("MASTER: performing intermediate LMIN ...", flush=True)

                opt_results = least_squares(toolbox.evaluate_population, indiv,
                                            toolbox.gradient, method='lm',
                                            max_nfev=INTER_NSTEPS*2)


                opt_indiv = creator.Individual(opt_results['x'])

                opt_fitness = np.sum(toolbox.evaluate_population(opt_indiv))
                prev_fitness = np.sum(toolbox.evaluate_population(indiv))
                # opt_fitness =  np.sum(min_fxn(opt_indiv))
                # prev_fitness = np.sum(min_fxn(indiv))

                if opt_fitness < prev_fitness:
                    indiv = opt_indiv
                # else:
                #     print("MASTER: LM was unable to reduce the cost", flush=True)

                # print("SLAVE: Rank", rank, "minimized fitness:", np.sum(toolbox.evaluate_population(indiv)))

            # Compute fitnesses with mated/mutated/optimized population
            fitnesses = np.sum(toolbox.evaluate_population(indiv))
            # fitnesses = np.sum(min_fxn(indiv))

            all_fitnesses = comm.gather(fitnesses, root=0)
            pop = comm.gather(indiv, root=0)

            # Update individuals with new fitnesses
            if is_master_node:
                # all_fitnesses = np.sum(all_fitnesses, axis=0)
                #print("MASTER: received fitnesses:", all_fitnesses, flush=True)
                all_fitnesses = np.vstack(all_fitnesses)

                for ind,fit in zip(pop, all_fitnesses):
                    ind.fitness.values = fit,

                # Sort
                pop = tools.selBest(pop, len(pop))
                best_fit = np.sum(toolbox.evaluate_population(pop[0]))

                # Print statistics to screen and checkpoint
                print_statistics(pop, i, stats, logbook)

                if (i % CHECKPOINT_FREQUENCY == 0):
                    best = np.array(tools.selBest(pop, 1)[0])
                    checkpoint(pop, logbook, best, i)

                best_guess = pop[0]
            else:
                best_fit = None

            best_fit = comm.bcast(best_fit, root=0)

            i += 1
    else:
        pop = np.genfromtxt(POP_FILE_NAME)
        best_guess = creator.Individual(pop[0])

    # Perform a final local optimization on the final results of the GA
    if is_master_node:
        ga_runtime = time.time() - ga_start

        print("MASTER: GA runtime = {:.2f} (s)".format(ga_runtime), flush=True)
        print("MASTER: Average time per step = {:.2f}"
                " (s)".format(ga_runtime/NUM_GENS), flush=True)

        lmin_start = time.time()

        # best_fitness = np.sum(toolbox.evaluate_population(best_guess))

    print("SLAVE: Rank", rank,  "performing final minimization ... ", flush=True)
    opt_results = least_squares(toolbox.evaluate_population, indiv,
                                toolbox.gradient, method='lm',
                                max_nfev=FINAL_NSTEPS)

    final = creator.Individual(opt_results['x'])

    print("SLAVE: Rank", rank, "minimized fitness:", np.sum(toolbox.evaluate_population(final)))
    # print("SLAVE: Rank", rank, "minimized fitness:", np.sum(min_fxn(final)))
    fitnesses = np.sum(toolbox.evaluate_population(final))
    # fitnesses = np.sum(min_fxn(final))

    all_fitnesses = comm.gather(fitnesses, root=0)
    pop = comm.gather(final, root=0)

    # Save final results
    final_arr = np.zeros(pvec_len)
    if is_master_node:
        print("MASTER: final fitnesses:", all_fitnesses, flush=True)
        all_fitnesses = np.vstack(all_fitnesses)

        for ind,fit in zip(pop, all_fitnesses):
            ind.fitness.values = fit,

        # Sort
        pop = tools.selBest(pop, len(pop))

        # Print statistics to screen and checkpoint
        print_statistics(pop, i, stats, logbook)

        best = np.array(tools.selBest(pop, 1)[0])

        recheck = np.sum(toolbox.evaluate_population(best))
        # recheck = np.sum(min_fxn(best))
        print("MASTER: confirming best fitness:", recheck)

        final_arr = np.array(best)

        checkpoint(pop, logbook, final_arr, i)
        np.savetxt(SAVE_DIRECTORY + '/' + name + '_final_potential.dat', final_arr)

        # plot_best_individual()
    return final_arr

################################################################################ 

def build_ga_toolbox(pvec_len, index_ranges):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray,
            fitness=creator.CostFunctionMinimizer)

    def ret_pvec(arr_fxn, rng):
        scale_mag = 0.3
        # hard-coded version for pair-pots only
        # ind = np.zeros(83)
        # ind = np.zeros(137)
        ind = np.zeros(pvec_len)

        ranges = [(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3), (-0.5, 1),
                (-0.5, 1), (-2,3), (-2, 3), (-7,2), (-7,2), (-7,2)]

        indices = [(0,13), (15,28), (30,43), (45,56), (58,69), (71,75), (77,81),
                (83,93), (95,105), (107,115), (117,125), (127,135)]

        for rng,ind_tup in zip(ranges, indices):
            r_lo, r_hi = rng
            i_lo, i_hi = ind_tup

            ind[i_lo:i_hi] = np.random.random(i_hi-i_lo)*(r_hi-r_lo) + r_lo

        return arr_fxn(ind)

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual, np.random.normal)
    toolbox.register("population", tools.initRepeat, list, toolbox.parameter_set,)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    # toolbox.register("mate", tools.cxBlend, alpha=MATING_ALPHA)
    toolbox.register("mate", tools.cxTwoPoint)

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
        tmp = {}
        tmp2 = {}
        tmp3 = {}
        tmp4 = {}

        for name in group:
            tmp[name] = structs[name]
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
    print("Save location:", SAVE_DIRECTORY)
    if os.path.isdir(SAVE_DIRECTORY) and CHECK_BEFORE_OVERWRITE:
        print()
        print("/" + "*"*30 + " WARNING " + "*"*30 + "/")
        print("A folder already exists for these settings.\nPress Enter"
                " to ovewrite old data, or Ctrl-C to quit")
        input("/" + "*"*30 + " WARNING " + "*"*30 + "/\n")
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
    print("LMIN_FREQUENCY:", LMIN_FREQUENCY, flush=True)
    print("INIT_NSTEPS:", INIT_NSTEPS, flush=True)
    print("INTER_NSTEPS:", INTER_NSTEPS, flush=True)
    print("FINAL_NSTEPS:", FINAL_NSTEPS, flush=True)
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

def build_evaluation_functions(structures, weights, true_forces, true_energies,
        spline_indices):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""
    def fxn(pot):
        # Convert list of Individuals into a numpy array
        pot = np.atleast_2d(pot)
        # full = np.hstack([pot, np.zeros((pot.shape[0], 108))])
        # full = np.hstack([pot, np.zeros((pot.shape[0], 54))])
        full = pot.copy()
        fitness = np.zeros(2*len(structures))

        # Compute error for each worker on MPI node
        i = 0
        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(full) - true_forces[name]
            eng_err = w.compute_energy(full) - true_energies[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitness[i] += eng_err*eng_err*weights[name]
            fitness[i+1] += fcs_err*fcs_err*weights[name]

            i += 2

        # print(fitness, flush=True)
        # print(np.sum(fitness), flush=True)

        return fitness

    def grad(pot):
        pot = np.atleast_2d(pot)
        # full = np.hstack([pot, np.zeros((pot.shape[0], 108))])
        # full = np.hstack([pot, np.zeros((pot.shape[0], 54))])
        full = pot.copy()

        grad_vec = np.zeros((2*len(structures), full.shape[1]))

        i = 0
        for name in structures.keys():
            w = structures[name]

            eng_err = w.compute_energy(full) - true_energies[name]
            fcs_err = (w.compute_forces(full) - true_forces[name])

            # Scale force errors

            # compute gradients
            eng_grad = w.energy_gradient_wrt_pvec(full)
            fcs_grad = w.forces_gradient_wrt_pvec(full)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            grad_vec[i] += (eng_err[:, np.newaxis]*eng_grad*2).ravel()
            grad_vec[i+1] += (2*summed / 10).ravel()

            i += 2

        # return grad_vec[:,:36]
        # return grad_vec[:,:83]
        return grad_vec

    return fxn, grad


def build_eam_eval_functions(structures, weights, true_forces, true_energies,
        spline_indices):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""
    def fxn(pot):
        # Convert list of Individuals into a numpy array
        pot = np.atleast_2d(pot)
        # full = np.hstack([pot, np.zeros((pot.shape[0], 108))])
        full = np.hstack([pot, np.zeros((pot.shape[0], 54))])
        fitness = np.zeros(2*len(structures))

        # Compute error for each worker on MPI node
        i = 0
        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(full) - true_forces[name]
            eng_err = w.compute_energy(full) - true_energies[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitness[i] += eng_err*eng_err*weights[name]
            fitness[i+1] += fcs_err*fcs_err*weights[name]

            i += 2

        # print(fitness, flush=True)
        # print(np.sum(fitness), flush=True)

        return fitness

    def grad(pot):
        pot = np.atleast_2d(pot)
        # full = np.hstack([pot, np.zeros((pot.shape[0], 108))])
        full = np.hstack([pot, np.zeros((pot.shape[0], 54))])
        # full = pot.copy()

        grad_vec = np.zeros((2*len(structures), full.shape[1]))

        i = 0
        for name in structures.keys():
            w = structures[name]

            eng_err = w.compute_energy(full) - true_energies[name]
            fcs_err = (w.compute_forces(full) - true_forces[name])

            # Scale force errors

            # compute gradients
            eng_grad = w.energy_gradient_wrt_pvec(full)
            fcs_grad = w.forces_gradient_wrt_pvec(full)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            grad_vec[i] += (eng_err[:, np.newaxis]*eng_grad*2).ravel()
            grad_vec[i+1] += (2*summed / 10).ravel()

            i += 2

        # return grad_vec[:,:36]
        return grad_vec[:,:83]

    return fxn, grad

def build_meam_eval_functions(structures, weights, true_forces, true_energies,
        spline_indices):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""
    def fxn(pot):
        # Convert list of Individuals into a numpy array
        pot = np.atleast_2d(pot)

        full = np.hstack([np.zeros((pot.shape[0], 83)), pot])
        fitness = np.zeros(2*len(structures))

        # Compute error for each worker on MPI node
        i = 0
        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(full) - true_forces[name]
            eng_err = w.compute_energy(full) - true_energies[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitness[i] += eng_err*eng_err*weights[name]
            fitness[i+1] += fcs_err*fcs_err*weights[name]

            i += 2

        return fitness

    def grad(pot):
        pot = np.atleast_2d(pot)
        full = np.hstack([np.zeros((pot.shape[0], 83)), pot])

        grad_vec = np.zeros((2*len(structures), full.shape[1]))

        i = 0
        for name in structures.keys():
            w = structures[name]

            eng_err = w.compute_energy(full) - true_energies[name]
            fcs_err = (w.compute_forces(full) - true_forces[name])

            # Scale force errors

            # compute gradients
            eng_grad = w.energy_gradient_wrt_pvec(full)
            fcs_grad = w.forces_gradient_wrt_pvec(full)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            grad_vec[i] += (eng_err[:, np.newaxis]*eng_grad*2).ravel()
            grad_vec[i+1] += (2*summed / 10).ravel()

            i += 2

        return grad_vec[:,83:]

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

def local_minimization(guess, toolbox, is_master_node, comm, num_steps=None,
                       thresh=None):
    """Wrapper for local minimization function"""

    def parallel_wrapper(x, stop):
        """Wrapper to allow parallelization of local minimization. Explanation
        and code provided by Stackoverflow user 'francis'.

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

    def parallel_grad_wrapper(x, stop):
        """Wrapper to allow parallelization of gradient calculation."""

        stop[0] = comm.bcast(stop[0], root=0)
        x = comm.bcast(x, root=0)

        value = 0

        if stop[0] == 0: # i.e. DON'T stop
            cost = toolbox.gradient([x])
            all_costs = comm.gather(cost, root=0)

            if is_master_node:
                value = np.sum(all_costs, axis=0)

        return value

    def cb(x):
        pass
        # print('-'*30 + "CG step " + '-'*30)

    if is_master_node:
        stop = [0]
        locally_optimized_pot = fmin_cg(parallel_wrapper, guess,
                parallel_grad_wrapper, args=(stop,), maxiter=num_steps,
                callback=cb, disp=0, gtol=1e-6)

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
    return {key:1 for key in names}

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

################################################################################

if __name__ == "__main__":
    main()
