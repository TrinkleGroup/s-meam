import os
import pickle
import glob
import array
import datetime
import numpy as np
np.random.seed(42)
np.set_printoptions(precision=16, linewidth=np.inf, suppress=True)
import random
random.seed(42)
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.optimize import fmin_powell

import h5py
from parsl import *
from deap import base, creator, tools, algorithms

from src.worker import Worker
import src.meam
from src.meam import MEAM
from src.spline import Spline

os.chdir("/home/jvita/scripts/s-meam/project/")

from mpi4py import MPI

################################################################################
"""MPI settings"""

MASTER_RANK = 0 # rank of master node

################################################################################
"""MEAM potential settings"""

# ACTIVE_SPLINES = [1, 0, 0, 0, 0] # [phi, rho, u, f, g]
# ACTIVE_SPLINES = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ACTIVE_SPLINES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

################################################################################
"""GA settings"""

POP_SIZE = 13
NUM_GENS = 11
CXPB = 1.0
MUTPB = 0.5

DO_POWELL = True

################################################################################ 
"""I/O settings"""

load_path = "data/fitting_databases/lj/"
save_path = "data/ga_results/"
settings_str = "{}-{}-{}-{}".format(POP_SIZE, NUM_GENS, CXPB, MUTPB)

DB_FILE_NAME = load_path + 'structures.hdf5'
POP_FILE_NAME = save_path + settings_str + "/pop.dat"
FIT_FILE_NAME = save_path + settings_str + "/fit.dat"
MC_FILE_NAME = save_path + settings_str + "/mc.dat"
LOG_FILE_NAME = save_path + settings_str + "/ga.log"

################################################################################ 

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    is_master_node = (rank == MASTER_RANK)

    if is_master_node:
        print_settings()

        print("MASTER: Preparing save directory/files ... ", flush=True)
        prepare_save_directory()

        # popfile = open(POP_FILE_NAME, 'wb')
        # fitfile = open(FIT_FILE_NAME, 'wb')

        stats, logbook = build_stats_and_log()

        print("MASTER: Loading structures ...", flush=True)
        structures, weights = load_structures_on_master()

        print("MASTER: Loading energy/forces database ... ", flush=True)
        true_forces, true_energies = load_true_values(structures.keys())

        print("MASTER: Determining potential information ...", flush=True)
        ex_struct = structures[list(structures.keys())[0]]
        type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
        pvec_len = ex_struct.len_param_vec

        print("MASTER: Sending structures to slaves ... ", flush=True)
        grouped_tup = group_for_mpi_scatter(structures, weights, true_forces,
                true_energies, mpi_size)

        structures = grouped_tup[0]
        weights = grouped_tup[1]
        true_forces = grouped_tup[2]
        true_energies = grouped_tup[3]
    else:
        spline_indices = None
        structures = None
        weights = None
        true_forces = None
        true_energies = None

    spline_indices = comm.bcast(spline_indices, root=0)

    structures = comm.scatter(structures, root=0)
    weights = comm.scatter(weights, root=0)
    true_forces = comm.scatter(true_forces, root=0)
    true_energies = comm.scatter(true_energies, root=0)

    print("SLAVE: Rank", rank, "received", len(structures), 'structures',
            flush=True)

    # have every process build the toolbox
    pvec_len = structures[list(structures.keys())[0]].len_param_vec
    toolbox, creator = build_ga_toolbox(pvec_len, spline_indices)

    eval_fxn = build_evaluation_function(structures, weights, true_forces,
            true_energies, spline_indices)

    toolbox.register("evaluate_population", eval_fxn)

    # compute initial fitnesses
    if is_master_node:
        pop = toolbox.population(n=POP_SIZE)
    else:
        pop = None

    pop = comm.bcast(pop, root=0)

    fitnesses = toolbox.evaluate_population(pop)
    all_fitnesses = comm.gather(fitnesses, root=0)

    if is_master_node:
        trace = np.zeros((NUM_GENS+1, pvec_len))

        all_fitnesses = np.sum(all_fitnesses, axis=0)

        for ind,fit in zip(pop, all_fitnesses):
            ind.fitness.values = fit,

        best = np.array(tools.selBest(pop, 1)[0])
        trace[0,:] = np.array(best)

        print_statistics(pop, 0, stats, logbook)

    run_new_ga = True

    if run_new_ga:
        i = 0
        while (i < NUM_GENS):
            if is_master_node:

                # sort population; best on top
                pop = tools.selBest(pop, len(pop))
                pop2 = list(map(toolbox.clone, pop))

                # preserve top 2, cross others with top 2
                for j in range(2, len(pop)):
                    mom = pop[np.random.randint(2)]
                    dad = pop[j]

                    kid, _ = toolbox.mate(toolbox.clone(mom), toolbox.clone(dad))
                    pop[j] = kid

                # mutate randomly everyone except top 2
                for ind in pop[2:]:
                    if MUTPB >= np.random.random():
                        toolbox.mutate(ind)
            else:
                pop = None

            pop = comm.bcast(pop, root=0)

            if DO_POWELL and (i % 10 == 0):
                # pop = tools.selBest(pop, len(pop))
                guess = pop[0]

                if is_master_node:

                    print("MASTER: performing powell minimization on best individual", flush=True)
                #     print("MASTER: Fitness before Powell (but after breeding/mutating) =",
                #             guess.fitness.values, flush=True)

                was_run_prev = True

                if not was_run_prev:
                    improved = run_powell_on_best(guess, toolbox,
                            is_master_node, comm)

                if is_master_node:
                    if not was_run_prev:
                        np.savetxt("after_powell_temp.dat", improved)
                    else:
                        improved = np.genfromtxt("after_powell_temp.dat")

                    # improved = scale_into_range(improved, spline_indices)
                    # optimized_fitness = toolbox.evaluate_population([improved])

                    # optimized_fitness = np.sum(optimized_fitness, axis=0)

                    prev_pop = list(pop)
                    pop[0] = creator.Individual(improved)
                    # pop[0].fitness.values = optimized_fitness,
                    # print("MASTER: Fitness after Powell =", optimized_fitness)

            pop = comm.bcast(pop, root=0)

            fitnesses = toolbox.evaluate_population(pop)
            all_fitnesses = comm.gather(fitnesses, root=0)

            if is_master_node:
                all_fitnesses = np.sum(all_fitnesses, axis=0)

                for ind,fit in zip(pop, all_fitnesses):
                    ind.fitness.values = fit,

            if is_master_node:
                print_statistics(pop, i+1, stats, logbook)

                best = np.array(tools.selBest(pop, 1)[0])
                trace[i,:] = np.array(best)

                if (i % 10 == 0):
                    np.savetxt(open('best' + str(i), 'wb'), best)
                    pickle.dump(logbook, open(LOG_FILE_NAME, 'wb'))

            i += 1

    if is_master_node:
        if run_new_ga:
            best_guess = tools.selBest(pop, 1)[0]
        else:
            best_fits = np.genfromtxt(FIT_FILE_NAME, skip_header=4)
            best_fit_idx = np.argmin(best_fits[-1,:])

            best_guess = np.genfromtxt(POP_FILE_NAME, skip_header=4)
            best_guess = np.split(best_guess, NUM_GENS)[-1]
            best_guess = best_guess[best_fit_idx]
            best_fitness = toolbox.evaluate_population([best_guess])
    else:
        best_guess = None

    best_guess = comm.bcast(best_guess, root=0)

    best_fitness = toolbox.evaluate_population([best_guess])

    best_fitness = comm.gather(best_fitness, root=0)

    if is_master_node:
        best_fitness = np.sum(best_fitness, axis=0)

        print("MASTER: performing powell minimization on final result",
                flush=True)

    improved = run_powell_on_best(best_guess, toolbox,
            is_master_node, comm, 15)

    if is_master_node:
        # improved = scale_into_range(improved, spline_indices)
        optimized_fitness = toolbox.evaluate_population([improved])

        optimized_fitness = np.sum(optimized_fitness, axis=0)

        pop[0] = creator.Individual(improved)
        pop[0].fitness.values = optimized_fitness,
        print("MASTER: Fitness after Powell =", optimized_fitness)

    if is_master_node:
        trace[-1,:] = np.array(improved)
        plot_the_best_individual(trace)

        # popfile.close()
        # fitfile.close()

################################################################################ 

def get_hdf5_testing_database(load=False):
    """Loads HDF5 database of workers using the h5py 'core' driver (in memory)

    Args:
        load (bool): True if database already exists; default is False
    """

    return h5py.File("data/fitting_databases/lj/structures.hdf5", 'a',)

def load_true_energies_and_forces_dicts(worker_names):
    """Loads true forces and energies from formatted text files"""
    path = "data/fitting_databases/lj/info/info."

    true_forces = {}
    true_energies = {}

    for struct_name in worker_names:

        fcs = np.genfromtxt(open(path+struct_name, 'rb'), skip_header=1)
        eng = np.genfromtxt(open(path+struct_name, 'rb'), max_rows=1)

        true_forces[struct_name] = fcs
        true_energies[struct_name] = eng

    return true_forces, true_energies

def build_ga_toolbox(pvec_len, index_ranges):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray,
            fitness=creator.CostFunctionMinimizer)

    def ret_pvec(arr_fxn, rng):
        # ind = np.random.normal(size=(pvec_len,), scale=0.1)
        ind = np.zeros(pvec_len)

        ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
                (-0.2, 0.4), (-2, 3), (-7.5, 12), (-7, 2), (-1, 0.2), (-1, 1)]

        indices = index_ranges + [pvec_len]

        mask = np.zeros(ind.shape)

        for i,is_active in enumerate(ACTIVE_SPLINES):
            if is_active == 1:
                a, b = ranges[i]
                start, stop = indices[i], indices[i+1]

                ind[start:stop] += np.linspace(0.2*a, 0.8*b, stop-start)[::-1]
                ind[start:stop] += np.random.normal(size=(stop-start,),
                        scale=(b-a)*0.1)

                mask[start:stop] = 1

        return arr_fxn(np.multiply(ind, mask))

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual,
            np.random.normal)

    toolbox.register("population", tools.initRepeat, list, toolbox.parameter_set,)

    def my_mut(ind):
        tmp = tools.mutGaussian(ind, mu=0, sigma=1e-1, indpb=0.1)
        # tmp[0][36:] = 0

        return tmp

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    # toolbox.register("mutate", my_mut)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    return toolbox, creator

def group_for_mpi_scatter(structs, database_weights, true_forces, true_energies,
        size):
    grouped_keys = np.array_split(list(structs.keys()), size)

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

def compare_to_true(new_pvec, fname=''):
    ranges = [(-5., 5.), (-5., 5.), (-5., 5.), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

    # ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            # (-0.2, 0.4), (-2, 3), (-7.5, 12), (-7, 2), (-1, 0.2), (-1, 1)]

    # new_pvec = np.concatenate([new_pvec, np.zeros(144 - 36)])

    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/lj/lj.meam')
    x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    for i in range(len(ranges)-1):
        a, b = ranges[i]
        i_lo, i_hi = indices[i], indices[i+1]

        new_pvec[i_lo:i_hi] = new_pvec[i_lo:i_hi]*(b-a) + a

    print(new_pvec.shape)
    print(x_pvec.shape)
    new_pot = MEAM.from_pvec(x_pvec, new_pvec, indices, ['H', 'He'])

    splines1 = pot.splines
    splines2 = new_pot.splines

    import matplotlib.pyplot as plt

    for i,(s1,s2) in enumerate(zip(splines1, splines2)):

        low,high = s1.cutoff
        low -= abs(0.1*low)
        high += abs(0.1*high)

        x = np.linspace(low,high,1000)
        # y1 = list(map(lambda e: s1(e) if s1.in_range(e) else s1.extrap(e), x))
        # y2 = list(map(lambda e: s2(e) if s2.in_range(e) else s2.extrap(e), x))
        y1 = list(map(lambda e: s1(e), x))
        y2 = list(map(lambda e: s2(e), x))

        yi = list(map(lambda e: s1(e), s1.x))

        plt.figure()
        plt.plot(s1.x, yi, 'o')
        plt.plot(x, y1, 'b', label='true')
        plt.plot(x, y2, 'r', label='new')
        plt.legend()

        plt.savefig(fname + str(i+1) + ".png")
        plt.close()

def plot_the_best_individual(trace):
    from scipy.interpolate import CubicSpline

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
    if not os.path.isdir(save_path + settings_str):
        os.mkdir(save_path + settings_str)

def print_settings():
    print("POP_SIZE:", POP_SIZE, flush=True)
    print("NUM_GENS:", NUM_GENS, flush=True)
    print("CXPB:", CXPB, flush=True)
    print("MUTPB:", MUTPB, flush=True)

def load_structures_on_master():
    database = h5py.File(DB_FILE_NAME, 'a',)
    weights = {key:1 for key in database.keys()}

    structures = {}

    for name in database.keys():
        # if weights[name] > 0:
        if name == 'diamond_ab':
            structures[name] = Worker.from_hdf5(database, name)

    database.close()

    return structures, weights

def load_true_values(all_names):
    """Loads the 'true' values according to the database provided"""
    true_forces = {}
    true_energies = {}

    for name in all_names:

        fcs = np.genfromtxt(open(load_path + 'info/info.' + name, 'rb'),
                skip_header=1)
        eng = np.genfromtxt(open(load_path + 'info/info.' + name, 'rb'),
                max_rows=1)

        true_forces[name] = fcs
        true_energies[name] = eng

    return true_forces, true_energies

def find_spline_type_deliminating_indices(worker):
    """Finds the indices in the parameter vector that correspond to start/end
    (inclusive/exclusive respectively) for each spline group

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
        # TODO: verify this isn't messing up like with the np version

        scaled = np.array(population)

        fitnesses = np.zeros(scaled.shape[0])

        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(scaled)/w.natoms - true_forces[name]
            eng_err = w.compute_energy(scaled)/w.natoms - true_energies[name]

            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitnesses += fcs_err*fcs_err*weights[name]
            fitnesses += eng_err*eng_err*weights[name]

        return fitnesses

    return fxn

def build_stats_and_log():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "size", "min", "max", "avg", "std"

    return stats, logbook

def print_statistics(pop, gen_num, stats, logbook):

    record = stats.compile(pop)
    logbook.record(gen=gen_num, size=len(pop), **record)

    pickle.dump(logbook, open(LOG_FILE_NAME, 'wb'))

    print(logbook.stream, flush=True)

def run_powell_on_best(guess, toolbox, is_master_node, comm, num_steps=2):
    def cb(x):
        val = toolbox.evaluate_population([x])
        print("MASTER: Rank", comm.Get_rank(), "powell step: ", val, flush=True)
        # pass

    def parallel_wrapper(x, stop):
        stop[0] = comm.bcast(stop[0], root=0)
        x = comm.bcast(x, root=0)

        value = 0

        if stop[0] == 0: # i.e. DON'T stop
            cost = toolbox.evaluate_population([x])
            all_costs = comm.gather(cost, root=0)

            if is_master_node:
                value = np.sum(all_costs)

        return value

    if is_master_node:
        stop = [0]
        locally_optimized_pot = fmin_powell(parallel_wrapper, guess,
                args=(stop,), maxiter=num_steps, callback=cb, disp=0)
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

################################################################################

if __name__ == "__main__":
    main()
