"""A genetic algorithm module for use in potential fitting and database
optimization. In terms of the original paper, this module would be intended to
run a GA over the fitting databases as well as a GA to find the theta_MLE.

Authors: Josh Vita (UIUC), Dallas Trinkle (UIUC)
"""

import numpy as np


np.set_printoptions(precision=8, suppress=True)

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
from mpi4py import MPI

from deap import base, creator, tools

import src.partools as partools
from partools import local_minimization

################################################################################

def ga(parameters, database, potential_template, is_manager, manager,
        manager_comm):
# def ga(parameters, template):
    # Record MPI settings
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
        # Prepare directories and files
        print_settings(parameters)

        # GA tools
        stats, logbook = build_stats_and_log()

        all_struct_names = [s.encode('utf-8').strip().decode('utf-8') for s in
                            list(database.keys())]

        struct_natoms = database.unique_natoms
        num_structs = len(all_struct_names)

        print(all_struct_names)

        old_copy_names = list(all_struct_names)

        worker_ranks = partools.compute_procs_per_subset(
            struct_natoms, world_size
        )

        print("worker_ranks:", worker_ranks)
    else:
        database = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

    potential_template = world_comm.bcast(potential_template, root=0)

    fxn_wrap, grad_wrap = partools.build_evaluation_functions(
        potential_template, database, all_struct_names, manager,
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
        # master_pop = np.ones(master_pop.shape)

        ud = np.concatenate(potential_template.u_ranges)
        u_domains = np.tile(ud, (master_pop.shape[0], 1))

        weights = np.ones(len(database.entries))
    else:
        master_pop = np.zeros(1)
        u_domains = np.zeros(1)
        weights = None

    master_pop = np.array(master_pop)

    weights = world_comm.bcast(weights, root=0)

    init_fit, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
        np.hstack([master_pop, u_domains]), weights, return_ni=True,
        penalty=True
    )

    if is_master:
        init_fit = np.sum(init_fit, axis=1)

        master_pop = partools.rescale_ni(
            master_pop, min_ni, max_ni,
            potential_template
        )

        subset = master_pop[:10]
    else:
        subset = None

    new_fit, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
        np.hstack([master_pop, u_domains]), weights, return_ni=True, penalty=True
    )

    # Have master gather fitnesses and update individuals
    if is_master:
        new_fit = np.sum(new_fit, axis=1)

        pop_copy = []
        for ind in master_pop:
            pop_copy.append(creator.Individual(ind))

        master_pop = pop_copy

        for ind, fit in zip(master_pop, new_fit):
            ind.fitness.values = fit,

        # Sort population; best on top
        master_pop = tools.selBest(master_pop, len(master_pop))
        u_domains = u_domains[np.argsort(new_fit)]

        partools.checkpoint(
            master_pop, new_fit, max_ni,
            min_ni, avg_ni, 0, parameters, potential_template
        )

    if is_master:
        print_statistics(master_pop, 0, stats, logbook)

        ga_start = time.time()

    # Begin GA
    if parameters['RUN_NEW_GA']:
        generation_number = 1
        while (generation_number < parameters['GA_NSTEPS']):
            if is_master:
                # print("before:", np.min(new_fit), new_fit[0])

                master_pop = tools.selBest(master_pop, len(master_pop))
                # Preserve top 50%, breed survivors
                for pot_num in range(len(master_pop) // 2, len(master_pop)):
                    mom_idx = np.random.randint(1, len(master_pop) // 2)

                    # TODO: add a check to make sure GA popsize is large enough
                    dad_idx = mom_idx
                    while dad_idx == mom_idx:
                        dad_idx = np.random.randint(1, len(master_pop) // 2)

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

                subset = np.array(subset)
                subset = local_minimization(
                    subset, u_domains, toolbox.evaluate_population,
                    toolbox.gradient, weights, world_comm, is_master,
                    nsteps=parameters['LMIN_NSTEPS']
                )

                if is_master:
                    master_pop[:10] = subset

                fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
                    np.hstack([master_pop, u_domains]), weights, return_ni=True, penalty=True
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

            if is_master:
                u_domains = np.tile([-1, 1, -1, 1], (6, 1))

            master_pop = np.array(master_pop)
            fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
                np.hstack([master_pop, u_domains]), weights, return_ni=True, penalty=True
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
                u_domains = u_domains[np.argsort(new_fit)]
                new_fit = new_fit[np.argsort(new_fit)]

                if (generation_number % parameters['CHECKPOINT_FREQ'] == 0):
                    best = np.array(tools.selBest(master_pop, 1)[0])

                    partools.checkpoint(
                        master_pop,
                        new_fit, tmp_max_ni, tmp_min_ni, tmp_avg_ni,
                        generation_number, parameters, potential_template
                    )

                best_guess = master_pop[0]

                # Print statistics to screen and checkpoint
                # print(new_fit)
                print_statistics(master_pop, generation_number, stats, logbook)
                # print("after:", np.min(new_fit), new_fit[0])

            generation_number += 1

        # end of GA loop

        # Perform a final local optimization on the final results of the GA
        if parameters['DO_LMIN']:
            if is_master:
                print("Performing final local minimization ...", flush=True)
                subset = master_pop[:10]
            else:
                subset = None

            subset = np.array(subset)
            subset = local_minimization(
                subset, u_domains, toolbox, weights, world_comm, is_master,
                nsteps=parameters['LMIN_NSTEPS']
            )

        if is_master:
            if parameters['DO_LMIN']:
                master_pop[:10] = subset


        master_pop = np.array(master_pop)
        fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
            np.hstack([master_pop, u_domains]), weights, return_ni=True, penalty=False
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
            u_domains = u_domains[np.argsort(new_fit)]

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


def plot_best_individual(parameters):
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
    stats.register("0", lambda x: x[0])

    logbook = tools.Logbook()
    logbook.header = "gen", "size", "min", "max", "avg", "std", "0"

    return stats, logbook


def print_statistics(pop, gen_num, stats, logbook):
    """Use Statistics and Logbook objects to output results to screen"""

    record = stats.compile(pop)
    logbook.record(gen=gen_num, size=len(pop), **record)
    print(logbook.stream, flush=True)

################################################################################
