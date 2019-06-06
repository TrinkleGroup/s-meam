"""A genetic algorithm module for use in potential fitting and database
optimization. In terms of the original paper, this module would be intended to
run a GA over the fitting databases as well as a GA to find the theta_MLE.

Authors: Josh Vita (UIUC), Dallas Trinkle (UIUC)
"""

import os
import sys
import numpy as np
import random
import shutil

np.set_printoptions(precision=8, suppress=True)

import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from mpi4py import MPI

from deap import base, creator, tools

import src.partools as partools
from src.database import Database
from src.manager import Manager


################################################################################

def ga(parameters, template):
    # Record MPI settings
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
        # Prepare directories and files
        print_settings(parameters)

        print("MASTER: Preparing save directory/files ... ", flush=True)
        prepare_save_directory(parameters)

        # GA tools
        stats, logbook = build_stats_and_log()

        potential_template = template

        master_database = Database(
            parameters['STRUCTURE_DIRECTORY'], parameters['INFO_DIRECTORY'],
            "Ti48Mo80_type1_c18"
        )

        master_database.load_structures(parameters['NUM_STRUCTS'])

        # all_struct_names  , structures = zip(*master_database.structures.items())
        all_struct_names = [s.encode('utf-8').strip().decode('utf-8') for s in
                            master_database.unique_structs]

        struct_natoms = master_database.unique_natoms
        num_structs = len(all_struct_names)

        print(all_struct_names)

        old_copy_names = list(all_struct_names)

        worker_ranks = partools.compute_procs_per_subset(
            struct_natoms, world_size
        )

        print("worker_ranks:", worker_ranks)
    else:
        potential_template = None
        master_database = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

    potential_template = world_comm.bcast(potential_template, root=0)

    # each Manager is in charge of a single structure
    world_group = world_comm.Get_group()

    all_rank_lists = world_comm.bcast(worker_ranks, root=0)
    all_struct_names = world_comm.bcast(all_struct_names, root=0)

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

        struct_name = manager_comm.scatter(list(all_struct_names), root=0)

        print(
            "Manager", manager_rank, "received structure", struct_name, "plus",
            len(worker_ranks), "processors for evaluation", flush=True
        )

    else:
        struct_name = None
        manager_rank = None

    if is_master:
        all_struct_names = list(old_copy_names)

    worker_group = world_group.Incl(worker_ranks)
    worker_comm = world_comm.Create(worker_group)

    struct_name = worker_comm.bcast(struct_name, root=0)
    manager_rank = worker_comm.bcast(manager_rank, root=0)

    manager = Manager(manager_rank, worker_comm, potential_template)

    manager.struct_name = struct_name
    manager.struct = manager.load_structure(
        manager.struct_name, parameters['STRUCTURE_DIRECTORY'] + "/"
    )

    manager.struct = manager.broadcast_struct(manager.struct)

    fxn_wrap, grad_wrap = partools.build_evaluation_functions(
        potential_template, master_database, all_struct_names, manager,
        is_master, is_manager, manager_comm, "Ti48Mo80_type1_c18"
    )

    # Have every process build the toolbox
    toolbox, creator = build_ga_toolbox(potential_template)

    toolbox.register("evaluate_population", fxn_wrap)
    toolbox.register("gradient", grad_wrap)

    # Create the original population
    if is_master:
        master_pop = toolbox.population(n=parameters['POP_SIZE'])
        master_pop = np.array(master_pop)
        print("master_pop.shape", master_pop.shape)

        weights = np.ones(len(master_database.entries))
    else:
        master_pop = 0
        weights = None

    master_pop = np.array(master_pop)
    weights = world_comm.bcast(weights, root=0)

    init_fit, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
        master_pop, weights, return_ni=True, penalty=True
    )

    if is_master:
        init_fit = np.sum(init_fit, axis=1)

        print('max_ni:', max_ni)
        print('min_ni:', min_ni)

        master_pop = partools.rescale_ni(
            master_pop, min_ni, max_ni,
            potential_template
        )

        subset = master_pop[:10]
    else:
        subset = None

    new_fit, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
            master_pop, weights, return_ni=True, penalty=True)

    if is_master:
        new_fit = np.sum(new_fit, axis=1)

        print('after max_ni:', max_ni)
        print('after min_ni:', min_ni)

    # Have master gather fitnesses and update individuals
    if is_master:

        pop_copy = []
        for ind in master_pop:
            pop_copy.append(creator.Individual(ind))

        master_pop = pop_copy

        for ind, fit in zip(master_pop, new_fit):
            ind.fitness.values = fit,

        # Sort population; best on top
        master_pop = tools.selBest(master_pop, len(master_pop))

        print_statistics(master_pop, 0, stats, logbook)

        partools.checkpoint(
            master_pop, new_fit, max_ni, min_ni, avg_ni, 0, parameters,
            potential_template
        )

        ga_start = time.time()

    # Begin GA
    if parameters['RUN_NEW_GA']:
        generation_number = 1
        while (generation_number < parameters['GA_NSTEPS']):

            if is_master:

                # Preserve top 50%, breed survivors
                for pot_num in range(len(master_pop) // 2, len(master_pop)):
                    mom_idx = np.random.randint(len(master_pop) // 2)

                    # TODO: add a check to make sure GA popsize is large enough

                    dad_idx = mom_idx
                    while dad_idx == mom_idx:
                        dad_idx = np.random.randint(len(master_pop) // 2)

                    mom = master_pop[mom_idx]
                    dad = master_pop[dad_idx]

                    kid, _ = toolbox.mate(toolbox.clone(mom),
                                          toolbox.clone(dad))
                    master_pop[pot_num] = kid

                # Mutate randomly everyone except top 10% (or top 2)
                for mut_ind in master_pop[
                               max(2, int(parameters['POP_SIZE'] / 10)):]:
                    if np.random.random() >= parameters[
                        'MUT_PB']: toolbox.mutate(mut_ind)

            # Run local minimization on best individual if desired
            if parameters['DO_LMIN'] and (
                    generation_number % parameters['LMIN_FREQ'] == 0):
                if is_master:
                    print("Performing local minimization ...", flush=True)
                    subset = master_pop[:10]

                subset = local_minimization(
                    subset, toolbox, weights, world_comm, is_master,
                    nsteps=parameters['LMIN_NSTEPS']
                )

                if is_master:
                    master_pop[:10] = subset

            fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
                master_pop, weights, return_ni=True, penalty=True
            )

            if parameters['DO_RESCALE'] and \
                    (generation_number % parameters['RESCALE_FREQ'] == 0):
                if is_master:
                    if generation_number < parameters['RESCALE_STOP_STEP']:
                        print("Rescaling ...")

                        master_pop = partools.rescale_ni(
                            master_pop, min_ni, max_ni,
                            potential_template
                        )
                else:
                    new_u_domains = None

            fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
                master_pop, weights, return_ni=True, penalty=True
            )

            if is_master:
                # only plotting the range of the 1st potential
                new_fit = np.sum(fitnesses, axis=1)

                tmp_min_ni = min_ni[np.argsort(new_fit)]
                tmp_max_ni = max_ni[np.argsort(new_fit)]
                tmp_avg_ni = avg_ni[np.argsort(new_fit)]

                pop_copy = []
                for ind in master_pop:
                    pop_copy.append(creator.Individual(ind))

                master_pop = pop_copy

                for ind, fit in zip(master_pop, new_fit):
                    ind.fitness.values = fit,

                # Sort
                master_pop = tools.selBest(master_pop, len(master_pop))

                # Print statistics to screen and checkpoint
                print_statistics(master_pop, generation_number, stats, logbook)

                if (generation_number % parameters['CHECKPOINT_FREQ'] == 0):
                    best = np.array(tools.selBest(master_pop, 1)[0])

                    partools.checkpoint(
                        master_pop, new_fit, tmp_max_ni, tmp_min_ni, tmp_avg_ni,
                        generation_number, parameters, potential_template
                    )

                best_guess = master_pop[0]

            generation_number += 1

        # end of GA loop

        # Perform a final local optimization on the final results of the GA
        if parameters['DO_LMIN']:
            if is_master:
                print("Performing final local minimization ...", flush=True)
                subset = master_pop[:10]
            else:
                subset = None

            subset = local_minimization(
                subset, toolbox, weights, world_comm, is_master,
                nsteps=parameters['LMIN_NSTEPS']
            )

        if is_master:
            if parameters['DO_LMIN']:
                master_pop[:10] = subset


        fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
            master_pop, weights, return_ni=True, penalty=False
        )

        if is_master:
            final_fit = np.sum(fitnesses, axis=1)

            pop_copy = []
            for ind in master_pop:
                pop_copy.append(creator.Individual(ind))

            master_pop = pop_copy

            for ind, fit in zip(master_pop, new_fit):
                ind.fitness.values = fit,

            master_pop = tools.selBest(master_pop, len(master_pop))
            ga_runtime = time.time() - ga_start

            partools.checkpoint(
                master_pop, final_fit, max_ni, min_ni, avg_ni,
                generation_number + 1, parameters, potential_template
            )

            print("MASTER: GA runtime = {:.2f} (s)".format(ga_runtime),
                  flush=True)
            print("MASTER: Average time per step = {:.2f}"
                  " (s)".format(ga_runtime / parameters['GA_NSTEPS']),
                  flush=True)

            best_guess = master_pop[0]

        else:
            best_guess = None

    if is_master:
        print("Final best cost = ", final_fit[0])

################################################################################

def build_ga_toolbox(potential_template):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray,
                   fitness=creator.CostFunctionMinimizer)

    def ret_pvec(arr_fxn):
        tmp = arr_fxn(potential_template.generate_random_instance())
        return tmp[np.where(potential_template.active_mask)[0]]

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual, )
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.parameter_set, )

    print(potential_template.scales)

    toolbox.register(
        "mutate", tools.mutGaussian, mu=0,
        sigma=(0.05 * potential_template.scales).tolist(),
        indpb=0.1
    )
    # toolbox.register("mate", tools.cxBlend, alpha=MATING_ALPHA)
    toolbox.register("mate", tools.cxTwoPoint)

    return toolbox, creator


def plot_best_individual():
    """Builds an animated plot of the trace of the GA. The final frame should be
    the final results after local optimization
    """

    trace = np.genfromtxt(parameters['TRACE_FILE_NAME'])

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


def prepare_save_directory(parameters):
    """Creates directories to store results"""

    # print()
    # print("Save location:", parameters['SAVE_DIRECTORY'])
    # if os.path.isdir(parameters['SAVE_DIRECTORY']):
    #     print()
    #     print("/" + "*" * 30 + " WARNING " + "*" * 30 + "/")
    #     print("A folder already exists for these settings.\nPress Enter"
    #           " to ovewrite old data, or Ctrl-C to quit")
    #     input("/" + "*" * 30 + " WARNING " + "*" * 30 + "/\n")
    # print()

    if os.path.isdir(parameters['SAVE_DIRECTORY']):
        shutil.rmtree(parameters['SAVE_DIRECTORY'])

    os.mkdir(parameters['SAVE_DIRECTORY'])


def print_settings(parameters):
    """Prints settings to screen"""

    print("POP_SIZE:", parameters['POP_SIZE'], flush=True)
    print("NUM_GENS:", parameters['GA_NSTEPS'], flush=True)
    # print("CXPB:", parameters['CX_PB'], flush=True)
    # print("MUTPB:", parameters['MUT_PB'], flush=True)
    print("CHECKPOINT_FREQUENCY:", parameters['CHECKPOINT_FREQ'], flush=True)
    print()


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
def local_minimization(master_pop, toolbox, weights, world_comm, is_master,
                       nsteps=20):
    pad = 100

    def lm_fxn_wrap(raveled_pop, original_shape):
        # print(raveled_pop)
        val = toolbox.evaluate_population(
            raveled_pop.reshape(original_shape), weights, output=False
        )

        val = world_comm.bcast(val, root=0)

        # pad with zeros since num structs is less than num knots
        tmp = np.concatenate([val.ravel(), np.zeros(pad * original_shape[0])])

        return tmp

    def lm_grad_wrap(raveled_pop, original_shape):
        grads = toolbox.gradient(
            raveled_pop.reshape(original_shape), weights
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

        tmp = np.vstack([
            padded_grad,
            np.zeros((pad * num_pots, num_pots * num_params))]
        )

        return tmp

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

    org_fits = toolbox.evaluate_population(master_pop, weights)
    new_fits = toolbox.evaluate_population(new_pop, weights)

    if is_master:
        updated_master_pop = list(master_pop)

        for i, ind in enumerate(new_pop):
            if np.sum(new_fits[i]) < np.sum(org_fits[i]):
                updated_master_pop[i] = new_pop[i]
            else:
                updated_master_pop[i] = updated_master_pop[i]

        master_pop = np.array(updated_master_pop)

    master_pop = world_comm.bcast(master_pop, root=0)

    return master_pop


def checkpoint(population, logbook, trace_update, i, parameters):
    """Saves information to files for later use"""

    digits = np.ceil(np.log10(int(parameters['GA_NSTEPS'])))

    format_str = os.path.join(
        parameters['SAVE_DIRECTORY'],
        'pop_{0:0' + str(int(digits) + 1) + 'd}.dat'
    )

    np.savetxt(format_str.format(i), population)

    pickle.dump(
        logbook,
        open(os.path.join(parameters['SAVE_DIRECTORY'], 'log.pkl'), 'wb')
    )

    f = open(parameters['TRACE_FILE_NAME'], 'ab')
    np.savetxt(f, [np.array(trace_update)])
    f.close()


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

################################################################################
