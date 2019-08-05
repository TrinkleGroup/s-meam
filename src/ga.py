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

def ga(parameters, database, template, node_manager,):
    # Record MPI settings
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()

    is_master = (world_rank == 0)

    if is_master:
        # Prepare directories and files
        print_settings(parameters)

        # GA tools
        stats, logbook = build_stats_and_log()

        if is_master:
            original_mask = template.active_mask.copy()

    template = world_comm.bcast(template, root=0)

    local_structs = world_comm.gather(node_manager.loaded_structures, root=0)

    if is_master:
        all_struct_names = []
        for slist in local_structs:
            all_struct_names += slist

        weights = np.ones(len(all_struct_names))
    else:
        database = None
        all_struct_names = None
        weights = None

    weights = world_comm.bcast(weights, root=0)
    all_struct_names = world_comm.bcast(all_struct_names, root=0)

    fxn_wrap, grad_wrap = partools.build_evaluation_functions(
        template, database, all_struct_names, node_manager,
        world_comm, is_master,  "Ti48Mo80_type1_c18"
    )

    # Have every process build the toolbox
    toolbox, creator = build_ga_toolbox(template)

    toolbox.register("evaluate_population", fxn_wrap)
    toolbox.register("gradient", grad_wrap)

    import random
    random.seed(42)
    np.random.seed(42)

    # Create the original population
    if is_master:

        # master_pop contains all of the parameters (un-masked)
        master_pop = toolbox.population(n=parameters['POP_SIZE'])
        master_pop = np.array(master_pop)

        # ga_pop contains only the active parameters (masked)
        ga_pop = master_pop[:, np.where(template.active_mask)[0]].copy()

    else:
        master_pop = np.empty((parameters['POP_SIZE'], template.pvec_len))

        ga_pop = np.zeros(
            (
                parameters['POP_SIZE'],
                len(np.where(template.active_mask.shape)[0])
            )
        )


    init_fit, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
        master_pop, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print('init min/max ni', min_ni[0], max_ni[0])

        master_pop = partools.rescale_ni(master_pop, min_ni, max_ni, template)
        ga_pop = master_pop[:, np.where(template.active_mask)[0]]

        subset = ga_pop[:10]
    else:
        subset = None

    new_fit, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
        master_pop, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    # Have master gather fitnesses and update individuals
    if is_master:
        print('rescaled min/max ni', min_ni[0], max_ni[0])

        new_fit = np.sum(new_fit, axis=1)

        pop_copy = []
        for ind in ga_pop:
            pop_copy.append(creator.Individual(ind))

        ga_pop = pop_copy

        for ind, fit in zip(ga_pop, new_fit):
            ind.fitness.values = fit,

        ga_pop = tools.selBest(ga_pop, len(ga_pop))

        # Sort population; best on top
        master_pop = master_pop[np.argsort(new_fit)]
        master_pop[:, np.where(template.active_mask)[0]] = np.array(ga_pop)

        partools.checkpoint(
            master_pop, new_fit, max_ni,
            min_ni, avg_ni, 0, parameters, template,
            parameters['NSTEPS']
        )

    if is_master:
        print_statistics(
            ga_pop, 0, stats, logbook, len(np.where(template.active_mask)[0])
        )

        ga_start = time.time()

    # TODO: only calculate these if specified

    import random
    random.seed(42)
    np.random.seed(42)

    toggle_time = 0
    lmin_time = 0
    resc_time = 0
    shift_time = 0
    mcmc_time = 0

    if parameters['DO_TOGGLE']:
        toggle_time = parameters['TOGGLE_FREQ']

    if parameters['DO_LMIN']:
        lmin_time = parameters['LMIN_FREQ']

    if parameters['DO_RESCALE']:
        resc_time = parameters['RESCALE_FREQ']

    if parameters['DO_SHIFT']:
        shift_time = parameters['SHIFT_FREQ']

    if parameters['DO_MCMC']:
        mcmc_time = parameters['MCMC_FREQ']

    mcmc_step = 0

    u_only_status = 'off'

    # begin GA
    generation_number = 1
    while (generation_number < parameters['NSTEPS']):
        if is_master:

            # Preserve top 20%, breed survivors
            for pot_num in range(len(ga_pop) // 2, len(ga_pop)):
                mom_idx = np.random.randint(1, len(ga_pop) // 2)

                # TODO: add a check to make sure GA popsize is large enough
                dad_idx = mom_idx
                while dad_idx == mom_idx:
                    dad_idx = np.random.randint(1, len(ga_pop) // 2)

                mom = ga_pop[mom_idx]
                dad = ga_pop[dad_idx]

                kid, _ = toolbox.mate(
                    toolbox.clone(mom), toolbox.clone(dad)
                )

                ga_pop[pot_num] = kid

            # probabalistically mutate everyone except top 10% (or top 2)
            for mut_ind in ga_pop[max(2, int(parameters['POP_SIZE']/10)):]:
                if np.random.random() >= parameters['MUT_PB']:
                    toolbox.mutate(mut_ind)

            master_pop[:, np.where(template.active_mask)[0]] = np.array(ga_pop)

        # optionally rescale potentials to put ni into [-1, 1]
        if parameters['DO_RESCALE']:
            if generation_number < parameters['RESCALE_STOP_STEP']:
                if resc_time == 0:
                    if is_master:
                        print("Rescaling ...")

                        new_fit = np.sum(fitnesses, axis=1)

                        master_pop = partools.rescale_ni(
                            master_pop, min_ni, max_ni, template
                        )

                    fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
                        master_pop, weights, return_ni=True,
                        penalty=parameters['PENALTY_ON']
                    )

                    resc_time = parameters['RESCALE_FREQ'] - 1
                else:
                    if u_only_status == 'off':  # don't decrement counter if U-only
                        resc_time -= 1

        # optionally shift the U domains to encompass the rescaled ni
        if parameters['DO_SHIFT']:
            if shift_time == 0:
                if is_master:
                    new_fit = np.sum(fitnesses, axis=1)

                    tmp_min_ni = min_ni[np.argsort(new_fit)]
                    tmp_max_ni = max_ni[np.argsort(new_fit)]

                    new_u_domains = partools.shift_u(tmp_min_ni, tmp_max_ni)
                    print("new_u_domains:", new_u_domains)
                else:
                    new_u_domains = None

                new_u_domains = world_comm.bcast(new_u_domains, root=0)
                template.u_ranges = new_u_domains

                shift_time = parameters['SHIFT_FREQ'] - 1
            else:
                if u_only_status == 'off':  # don't decrement counter if U-only
                    shift_time -= 1

        fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
            master_pop, weights, return_ni=True,
            penalty=parameters['PENALTY_ON']
        )

        if parameters['DO_MCMC'] and (mcmc_time == 0):
            if is_master: 
                print("Performing U-only MCMC...")

            master_pop = partools.mcmc(
                    master_pop, weights, toolbox.evaluate_population,
                    template, 100, parameters, [5, 6], is_master,
                    start_step=mcmc_step, max_nsteps=parameters['MCMC_NSTEPS'],
                    suffix="U"
            )

            fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
                master_pop, weights, return_ni=True,
                penalty=parameters['PENALTY_ON']
            )

            if is_master:
                new_fit = np.sum(fitnesses, axis=1)

                ga_pop = master_pop[:, np.where(template.active_mask)[0]].copy()

                pop_copy = []
                for ind in ga_pop:
                    pop_copy.append(creator.Individual(ind))

                ga_pop = pop_copy

                for ind, fit in zip(ga_pop, new_fit):
                    ind.fitness.values = fit,

            mcmc_step += parameters['MCMC_NSTEPS']
            mcmc_time = parameters['MCMC_FREQ'] - 1

        else:
            mcmc_time -= 1

        # optionally run local minimization on best individual
        if parameters['DO_LMIN']:
            if lmin_time == 0:
                if is_master:
                    print(
                        "Performing local minimization on top 10 potentials...",
                        flush=True
                    )

                    subset = ga_pop[:10]

                subset = np.array(subset)

                subset = local_minimization(
                    subset, master_pop, template, toolbox.evaluate_population,
                    toolbox.gradient, weights, world_comm, is_master,
                    nsteps=parameters['LMIN_NSTEPS'], lm_output=True
                )

                if is_master:
                    ga_pop[:10] = subset

                    master_pop[:, np.where(template.active_mask)[0]] = np.array(ga_pop)

                fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
                    master_pop, weights, return_ni=True,
                    penalty=parameters['PENALTY_ON']
                )

                lmin_time = parameters['LMIN_FREQ'] - 1
            else:
                if u_only_status == 'off':  # don't decrement counter if U-only
                    lmin_time -= 1

        # TODO: errors will occur if you try to resc/shift with U-only on
        if parameters['DO_TOGGLE']:
            if toggle_time == 0:
                # optionally toggle splines on/off to allow U-only optimization

                if u_only_status == 'off':
                    u_only_status = 'on'
                    toggle_time = parameters['TOGGLE_DURATION'] - 1
                else:
                    u_only_status = 'off'
                    toggle_time = parameters['TOGGLE_FREQ'] - 1

                if is_master:
                    print("Toggling U-only mode to:", u_only_status)

                    master_pop, ga_pop, template = toggle_u_only_optimization(
                        u_only_status, master_pop, ga_pop, template, [5, 6],
                        original_mask
                        # TODO: shouldn't hard-code [5,6]; use nphi to find tags
                    )

                    new_mask = template.active_mask
                else:
                    new_mask = None

                new_mask = world_comm.bcast(new_mask, root=0)

                template.active_mask = new_mask

            else:
                toggle_time -= 1

        # update GA Individuals with new costs; sort
        if is_master:
            new_fit = np.sum(fitnesses, axis=1)

            tmp_min_ni = min_ni[np.argsort(new_fit)]
            tmp_max_ni = max_ni[np.argsort(new_fit)]
            tmp_avg_ni = avg_ni[np.argsort(new_fit)]

            pop_copy = []
            for ind in ga_pop:
                pop_copy.append(creator.Individual(ind))

            ga_pop = pop_copy

            for ind, fit in zip(ga_pop, new_fit):
                ind.fitness.values = fit,

            # Sort
            ga_pop = tools.selBest(ga_pop, len(ga_pop))

            master_pop = master_pop[np.argsort(new_fit)]
            master_pop[:, np.where(template.active_mask)[0]] = np.array(ga_pop)

            new_fit = new_fit[np.argsort(new_fit)]

            # checkpoint; save population, cost, and ni trace
            if generation_number % parameters['CHECKPOINT_FREQ'] == 0:
                partools.checkpoint(
                    master_pop, new_fit, tmp_max_ni, tmp_min_ni, tmp_avg_ni,
                    generation_number, parameters, template,
                    parameters['NSTEPS']
                )

            print_statistics(
                ga_pop, generation_number, stats, logbook,
                len(np.where(template.active_mask)[0])
            )

        generation_number += 1

    # end of GA loop

    # make sure that U-only mode is turned off before final LM
    if u_only_status == 'on':
        u_only_status = 'off'

        if is_master:
            print(
                "Toggling U-only mode off for final LM",
                flush=True
            )

            master_pop, ga_pop, template = toggle_u_only_optimization(
                u_only_status, master_pop, ga_pop, template, [5, 6],
                original_mask
                # TODO: shouldn't hard-code [5,6]; use nphi to find tags
            )

            new_mask = template.active_mask
        else:
            new_mask = None

        new_mask = world_comm.bcast(new_mask, root=0)

        template.active_mask = new_mask
    else:
        if is_master:
            print("U-only mode is off; ready for final LM", flush=True)

    # Perform a final local optimization on the final results of the GA
    if is_master:
        ga_runtime = time.time() - ga_start

        print(
            "Performing final local minimization on top 10 potentials...",
            flush=True
        )

        subset = ga_pop[:10]
    else:
        subset = None

    subset = np.array(subset)
    subset = local_minimization(
        subset, master_pop, template, toolbox.evaluate_population,
        toolbox.gradient, weights, world_comm, is_master,
        nsteps=parameters['LMIN_NSTEPS']
    )

    if is_master:
        ga_pop[:10] = subset
        master_pop[:, np.where(template.active_mask)[0]] = np.array(ga_pop)

    # compute final fitness
    fitnesses, max_ni, min_ni, avg_ni = toolbox.evaluate_population(
        master_pop, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    # update Individuals with final fitnesses; checkpoint and print stats
    if is_master:
        final_fit = np.sum(fitnesses, axis=1)

        pop_copy = []
        for ind in ga_pop:
            pop_copy.append(creator.Individual(ind))

        ga_pop = pop_copy

        for ind, fit in zip(ga_pop, new_fit):
            ind.fitness.values = fit,

        ga_pop = tools.selBest(ga_pop, len(ga_pop))

        master_pop = master_pop[np.argsort(final_fit)]
        master_pop[:, np.where(template.active_mask)[0]] = np.array(ga_pop)

        partools.checkpoint(
            master_pop, final_fit, max_ni, min_ni, avg_ni,
            generation_number + 1, parameters, template,
            parameters['NSTEPS']
        )

        print()
        print("Final best cost = ", final_fit[0])

        print()
        print(
            "MASTER: GA runtime = {:.2f} (s)".format(ga_runtime), flush=True
        )

        print(
            "MASTER: Average time per step = {:.2f}"
            " (s)".format(ga_runtime / parameters['NSTEPS']), flush=True
        )

################################################################################

def build_ga_toolbox(template):
    """Initializes GA toolbox with desired functions"""

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray,
                   fitness=creator.CostFunctionMinimizer)

    def ret_pvec(arr_fxn):

        tmp = arr_fxn(template.pvec)

        # TODO: for some reason, this messes up the ni rescale??
        # for ind_tup, rng_tup in zip(
        #         template.spline_indices, template.spline_ranges
        # ):
        #
        #     start, stop = ind_tup
        #     low, high = rng_tup
        #
        #     scale = (high - low) / 5
        #     size = stop - start
        #
        #     tmp[start:stop] += np.random.normal(scale=scale, size=size)

        tmp[np.where(template.active_mask)[0]] = arr_fxn(
            template.generate_random_instance()[
                np.where(template.active_mask)[0]
            ]
        )

        return tmp
        # return tmp[np.where(template.active_mask)[0]]

    toolbox = base.Toolbox()
    toolbox.register("parameter_set", ret_pvec, creator.Individual, )
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.parameter_set, )

    toolbox.register(
        "mutate", tools.mutGaussian, mu=0,
        sigma=(0.05 * template.scales).tolist(),
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
    print("NUM_GENS:", parameters['NSTEPS'], flush=True)
    print()


def build_stats_and_log():
    """Initialize DEAP Statistics and Logbook objects"""

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "size", "min", "max", "avg", "std", "num_active"

    return stats, logbook


def print_statistics(pop, gen_num, stats, logbook, active_knots):
    """Use Statistics and Logbook objects to output results to screen"""

    record = stats.compile(pop)

    logbook.record(
        gen=gen_num, size=len(pop), **record, num_active=active_knots
    )

    print(logbook.stream, flush=True)

def toggle_u_only_optimization(toggle_type, master_pop, ga_pop, template,
                               spline_tags, original_mask):
    """

    Args:
        toggle_type: (str)
            'on' or 'off'

        master_pop: (np.arr)
            array of full potentials (un-masked)

        ga_pop: (list[Individual])
            list of Individual objects (masked)

        template: (Template)
            potential template object

        spline_tags: (list[int])
            which splines to toggle on/off

        original_mask: (np.arr)
            the original mask used before U-only was toggled on

    Returns:

        master_pop: (np.arr)
            the updated master potentials

        ga_pop: (list[individual])
            the updated list of individuals after toggling the U splines on/off

    """

    # save the current state of the population into master_pop
    master_pop[:, np.where(template.active_mask)[0]] = np.array(ga_pop)

    if toggle_type == 'on':
        # find the indices that should be toggled
        active_indices = []

        for tag in spline_tags:
            active_indices.append(np.where(template.spline_tags == tag)[0])

        active_indices = np.concatenate(active_indices)

        # TODO: in the future, make this able to toggle arbitrary splines on/off

        # toggle all indices off
        template.active_mask[:] = 0

        # toggle the specified indices on
        template.active_mask[active_indices] = np.bitwise_not(
            template.active_mask[active_indices]
        )

        template.active_mask = template.active_mask.astype(int)
    else:
        template.active_mask = original_mask

    # extract the new population
    ga_pop = master_pop[:, np.where(template.active_mask)[0]]

    return master_pop, ga_pop, template


################################################################################
