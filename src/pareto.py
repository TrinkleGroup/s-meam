"""
Implements GrEA, a grid-based evolutionary algorithm for estimating the pareto
front. Consolut the original reference paper for descriptions of terms.

Ref:
    "A Grid-Based Evolutionary Algorithm for Many-Objective Optimization" by
    Shengxiang Yang et. al
"""

import time
import random
import itertools
import numpy as np
from mpi4py import MPI
from deap import tools

import src.partools
import src.pareto


def GrEA(parameters, template, node_manager,):
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()

    is_master = (world_rank == 0)


    template = world_comm.bcast(template, root=0)

    # have to set a "weight" because NodeManager was meant to change weights
    node_manager.weights = dict(zip(
        node_manager.loaded_structures,
        np.ones(len(node_manager.loaded_structures))
    ))

    # figure out what structures exist on the node managers
    all_struct_names = collect_structure_names(
        node_manager, world_comm, is_master
    )

    # gather the true values of each objective target
    true_values = prepare_true_values(node_manager, world_comm, is_master)

    # run the function constructor to build the objective function
    objective_fxn = build_objective_function(
        template, all_struct_names, node_manager,
        world_comm, is_master, true_values
    )

    # initialize population and perform initial rescaling
    if is_master:
        # master_pop contains all of the parameters (un-masked)
        master_pop = initialize_population(template, parameters['POP_SIZE'])
        archive = master_pop.copy()

        # ga_pop contains only the active parameters (masked)
        ga_pop = master_pop[:, np.where(template.active_mask)[0]].copy()
    else:
        master_pop = np.empty((parameters['POP_SIZE'], template.pvec_len))
        archive = np.empty((parameters['POP_SIZE'], template.pvec_len))

        ga_pop = np.zeros(
            (
                parameters['POP_SIZE'],
                len(np.where(template.active_mask.shape)[0])
            )
        )

    population_costs, max_ni, min_ni, avg_ni = objective_fxn(
        master_pop, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Initial min/max ni:", min_ni[0], max_ni[0])

        master_pop = src.partools.rescale_ni(
            master_pop, min_ni, max_ni, template
        )

        ga_pop = master_pop[:, np.where(template.active_mask)[0]]

    population_costs, max_ni, min_ni, avg_ni = objective_fxn(
        master_pop, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Rescaled initial min/max ni:", min_ni[0], max_ni[0])

        new_obj = np.sum(population_costs, axis=1)

        # sort populations, best on top
        sorting_indices = np.argsort(new_obj)

        ga_pop = ga_pop[sorting_indices]
        master_pop = master_pop[sorting_indices]
        master_pop[:, np.where(template.active_mask)[0]] = ga_pop.copy()

        # log and print initial statistics
        src.partools.checkpoint(
            master_pop, new_obj, max_ni,
            min_ni, avg_ni, 0, parameters, template,
            parameters['NSTEPS']
        )

        # stats, logbook = build_stats_and_log()
        #
        # print_statistics(
        #     ga_pop, 0, stats, logbook, len(np.where(template.active_mask)[0])
        # )

        print(
            "{}\t{}\t{}\t{}".format(
                0, np.min(new_obj), np.max(new_obj), np.average(new_obj)
            ),
            flush=True
        )

        # track the original mask so that you can toggle things on/off later
        original_mask = template.active_mask.copy()

        ga_start = time.time()

    # prepare timers
    resc_time = 0
    shift_time = 0
    mcmc_time = 0

    if parameters['DO_RESCALE']:
        resc_time = parameters['RESCALE_FREQ']

    if parameters['DO_SHIFT']:
        shift_time = parameters['SHIFT_FREQ']

    if parameters['DO_MCMC']:
        mcmc_time = parameters['MCMC_FREQ']

    mcmc_step = 0

    # begin GA
    generation_number = 1
    while generation_number < parameters['NSTEPS']:

        population_costs, max_ni, min_ni, avg_ni = objective_fxn(
            master_pop, return_ni=True, penalty=parameters['PENALTY_ON']
        )

        archive_costs = objective_fxn(
            master_pop, return_ni=False, penalty=parameters['PENALTY_ON']
        )

        if is_master:

            generation_number += 1

            # checkpoint; save population, cost, and ni trace
            if generation_number % parameters['CHECKPOINT_FREQ'] == 0:
                src.partools.checkpoint(
                    archive, np.sum(archive_costs, axis=1),
                    max_ni, min_ni, avg_ni,
                    generation_number,
                    parameters, template, parameters['NSTEPS']
                )

            # print_statistics(
            #     archive, generation_number, stats, logbook,
            #     len(np.where(template.active_mask)[0])
            # )

            costs = np.sum(archive_costs, axis=1)

            print(
                "{}\t{}\t{}\t{}".format(
                    generation_number,
                    np.min(costs), np.max(costs), np.average(costs)
                ),
                flush=True
            )

            # TODO: compute these locally, then gather on master

            # determine grid settings
            grid_widths, grid_lower_bounds = src.pareto.grid_settings(
                population_costs, parameters['GRID_DIVS']
            )

            grid_coords = src.pareto.grid_coordinates(
                population_costs, grid_widths, grid_lower_bounds
            )

            # compute fitness metrics
            grid_differences = src.pareto.grid_difference(grid_coords)

            grid_crowding = src.pareto.grid_crowding_distance(
                grid_differences, population_costs.shape[1]
            )

            # breed individuals in the archive, then mutate their offspring
            master_pop = breed(
                archive, archive_costs, grid_coords, grid_crowding,
                parameters['POP_SIZE']
            )

            ga_pop = master_pop[:, np.where(template.active_mask)[0]]

            mutate(
                ga_pop,
                mu=0, sigma=template.scales[np.where(template.active_mask)[0]],
                mutProb=parameters['MUT_PB']
            )

            # TODO: is this necessary? np.arrays should be by reference
            master_pop[:, np.where(template.active_mask)[0]] = ga_pop.copy()

        population_costs, max_ni, min_ni, avg_ni = objective_fxn(
            master_pop, return_ni=True, penalty=parameters['PENALTY_ON']
        )

        # TODO: probabalistically kick pots out of archive based on frac_in?

        if is_master:
            archive = environmental_selection(
                parameters['ARCHIVE_SIZE'],
                np.vstack([archive, master_pop]),
                np.vstack([archive_costs, population_costs]),
                parameters['GRID_DIVS']
            )

def grid_settings(objective_values, num_div):
    """
    Separates each dimension into 'num_div' number of equally-sized divisions.
    The lower/upper bounds of the grid in each dimensions are taken as the
    max/min +/- half the division size.

    Args:
        objective_values: (np.arr)
            an NxM matrix corresponding to the objective values for each N
            individual in each M objective dimension
        num_div: (int)
            the number of boxes to split each dimension into

    Returns:
        grid_widths: (np.arr)
            grid widths in each M dimension
        lower_bound: (np.arr)
            lower_bounds in each M dimension

    """

    M = objective_values.shape[1]

    grid_widths = np.zeros(M)
    lower_bounds = np.zeros(M)

    for k in range(M):
        mn = np.min(objective_values[:, k])
        mx = np.max(objective_values[:, k])

        extra = (mx - mn)/(2*num_div)
        lower_bounds[k] = mn - extra
        upper_bound = mx + extra
        diff = upper_bound - lower_bounds[k]

        grid_widths[k] = diff/num_div

    return grid_widths, lower_bounds


def grid_coordinates(objective_values, grid_widths, lower_bounds):
    """
    Calculates the grid coordinates of each individual in the population.

    Args:
        objective_values: (np.arr)
            an NxM matrix corresponding to the objective values for each N
            individual in each M objective dimension
        grid_widths: (np.arr)
            grid widths in each M dimension
        lower_bounds: (np.arr)
            lower bounds of grid in each M dimension

    Returns:
        grid_coords: (np.arr)
            NxM matrix of coordinates in objective space

    """

    grid_coords = np.copy(objective_values)
    grid_coords -= lower_bounds
    # grid_coords /= grid_widths
    grid_coords = np.divide(
        grid_coords, grid_widths, out=np.zeros_like(grid_coords),
        where=grid_widths!=0
    )

    return np.floor(grid_coords)


def grid_difference(grid_coords):
    """
    Computes the "grid difference" between each individual. The grid difference
    is simply the sum of the differences between the grid coordinates of two
    individuals.

    Args:
        grid_coords: (np.arr)

    Returns:
        grid_difference: (np.arr)
            grid_difference[i, j] = sum(abs(grid_coords[i] - grid_coords[j]))
    """

    n = grid_coords.shape[0]
    grid_difference = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                grid_difference[i, j] = sum(abs(grid_coords[i] - grid_coords[j]))

    return grid_difference


def grid_ranks(grid_coords):
    return np.sum(grid_coords, axis=1)


def grid_crowding_distance(grid_difference, m):
    """
    Computes the "grid crowding distance" (GCD), which is a measure of how
    densely-packed the individuals are around a point in objective space.

    Note: the GCD is defined based on the "neighbors" of an individual. A point
    is considered to be a neighbor of another point if their grid_difference
    metric is less than the dimensionality of the objective space.

    Args:
        grid_difference: (np.arr)
            NxN array, the grid distance between each individual
        m: (int)
            the dimensionality of objective space

    Returns:
        crowding_distances: (np.arr)
            length N array, the density estimation around each individual
    """

    n = grid_difference.shape[0]

    crowding_distances = np.zeros(n)

    for i in range(n):
        neighbors = np.where(grid_difference[i] < m)[0]

        crowding_distances[i] = np.sum(m - grid_difference[i, neighbors])
        crowding_distances[i] -= m  # avoid self-counting

    return crowding_distances


def grid_gcpd(objective_values, grid_coords, lower_bounds, grid_widths):
    """
    Computes the GCPD (unsure what this stands for) for each point. This helps
    to distinguish fitnesses between two points that have the same grid
    coordinates.
    """

    box_edges = lower_bounds + grid_coords*grid_widths
    inner = objective_values - box_edges
    safe_divide = np.divide(
        inner, grid_widths, out=np.zeros_like(inner), where=grid_widths!=0
    )

    return np.sqrt(np.sum(safe_divide**2, axis=1))


def dominates(p_cost, q_cost):
    """
    Determines if individual p dominates (grid or pareto) individual q.

    Args:
        p_cost: (np.arr)
            cost vector for individual p
        q_cost: (np.arr)
            cost vector for individual q
    """

    # make sure p isn't worse in any of the dimensions
    not_worse = (len(np.where(p_cost > q_cost)[0]) == 0)

    # and that it's better in at least one dimension
    better_somewhere = (len(np.where(p_cost < q_cost)[0]) > 0)

    return not_worse and better_somewhere


def tournament_selection(p_idx, q_idx, pareto_costs, grid_coords, gcd):
    """
    Performs a modified tournament selection between individuals p and q based
    on their dominance relations.

    Args:
        p_idx: (int) integer index of p individual
        q_idx: (int) integer index of q individual
        pareto_costs: (np.arr) matrix of objective function values
        grid_coords: (np.arr) matrix of grid coordinates
        gcd: (np.arr) array of grid crowding distance metrics
    """
    p_pareto = pareto_costs[p_idx]
    q_pareto = pareto_costs[q_idx]

    p_grid = grid_coords[p_idx]
    q_grid = grid_coords[q_idx]

    if dominates(p_pareto, q_pareto) or dominates(p_grid, q_grid):
        return p_idx
    elif dominates(q_pareto, p_pareto) or dominates(q_grid, p_grid):
        return q_idx
    elif gcd[p_idx] < gcd[q_idx]:
        return p_idx
    elif gcd[q_idx] < gcd[p_idx]:
        return q_idx
    elif np.random.random() < 0.5:
        return p_idx
    else:
        return q_idx


def gr_adjustment(q, grid_coords, ranks, grid_diffs, M):
    """
    Modifies the grid ranks relative to a reference individual q. This is used
    to help with choosing if an individual should be placed in the archive,
    assuming that q is already in the archive. The goal is to avoid adding
    individuals that are too similar to q.

    q - the original individual
    E(q) - anyone in the same grid location as q
    G(q) - anyone that is grid-dominated by q
    NG(q) - anyone that is NOT grid-dominated by q
    N(q) - all neighbors of q (grid distance < objective space dimensionality)
    """

    modified_grid_ranks = ranks.copy()

    N = grid_coords.shape[0]
    punishment_degree = np.zeros(N)

    # warning: includes q
    same_grid_location = np.where(
            np.sum(abs(grid_diffs - grid_diffs[q]), axis=1) == 0
        )[0]

    # penalize individuals that share the same grid as q
    for p in same_grid_location:
        if p != q:
            modified_grid_ranks[p] += M + 2
            punishment_degree[p] += M + 2

    # TODO: if you ever compute this twice you should probably just store it

    grid_dominated_by_q = []
    not_dominated_by_q = []

    # penalize individuals that are grid-dominated by q
    for p in range(N):
        if dominates(grid_coords[q], grid_coords[p]):
            grid_dominated_by_q.append(p)
            modified_grid_ranks[p] += M
            punishment_degree[p] += M
        else:
            if p not in same_grid_location:
                not_dominated_by_q.append(p)
                punishment_degree[p] = 0

    neighbors = np.where(grid_diffs[q] < M)[0]

    grid_dominated_or_same_grid = set(grid_dominated_by_q).union(same_grid_location)

    # penalize individuals 
    for p in set(neighbors).intersection(set(not_dominated_by_q)):
        if p not in same_grid_location:
            desired_punishment = M - grid_diffs[p, q]

            if  punishment_degree[p] < desired_punishment:
                punishment_degree[p] = desired_punishment

                for r in range(N):
                    if dominates(grid_coords[p], grid_coords[r]):
                        if r not in grid_dominated_or_same_grid:
                            if  punishment_degree[r] < punishment_degree[p]:
                                punishment_degree[r] = punishment_degree[p]

    # update ranks with punishments
    for p in not_dominated_by_q:
        if p not in same_grid_location:
            modified_grid_ranks[p] += punishment_degree[p]

    return modified_grid_ranks


def fast_non_dominated_sort(population_costs):
    """
    Groups the population into "levels" of non-dominated fronts, where the first
    front (theoretically) converges to the pareto front.

    This is an implementation of the sorting algorithm from "A Fast and Elitist
    Multiobjective Genetic Algorithm: NSGA-II".
    """

    n = population_costs.shape[0]

    # track number of dominators for each individual
    domination_count = np.zeros(n)
    dominators = [list() for _ in range(n)]
    dominatees = [list() for _ in range(n)]

    fronts = [[]]

    for p_idx, p in enumerate(population_costs):
        pareto_dominated = []

        for q_idx, q in enumerate(population_costs):
            if p_idx != q_idx:
                if dominates(p, q):
                    # dominators[q_idx].append(p_idx)
                    dominatees[p_idx].append(q_idx)
                elif dominates(q, p):
                    # dominators[p_idx].append(q_idx)
                    # dominatees[q_idx].append(p_idx)
                    domination_count[p_idx] += 1

        if domination_count[p_idx] == 0:
            # TODO: may need to store individual, not just index?
            fronts[0].append(p_idx)

    i = 0
    num_added = len(fronts[0])
    # while (len(fronts[i]) > 1):
    while num_added < population_costs.shape[0]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in dominatees[p_idx]:
                domination_count[q_idx] -= 1

                if domination_count[q_idx] == 0:
                    next_front.append(q_idx)

        i += 1
        fronts.append(sorted(list(next_front)))
        num_added += len(next_front)

    return fronts


def environmental_selection(N, population, population_costs, num_div):
    """
    Builds the archive.

    Args:
        N: (int) archive size

    Returns:
        archive: (np.arr) the N individuals selected for the archive

    """

    fronts = fast_non_dominated_sort(population_costs)

    # add the fronts to the archive; for the last front ("critical front"),
    # only add the individuals that best increase the diverstity of the archive

    archive = np.empty((N, population.shape[1]))

    i = 0
    num_added = 0
    front_size = len(fronts[i])

    # add everything up to (but not including) the critical front
    while (num_added + front_size < N):
        archive[num_added : num_added + front_size] = population[fronts[i]]
        num_added += front_size

        i += 1
        front_size = len(fronts[i])

    # initialize grid settings for critical front
    critical_front = fronts[i]

    widths, lower_bounds = grid_settings(
            population_costs, num_div
        )

    coords = grid_coordinates(
            population_costs, widths, lower_bounds
        )

    # compute grid metrics
    ranks = grid_ranks(coords)
    differences = grid_difference(coords)

    gcpd = grid_gcpd(
            population_costs, coords, lower_bounds, widths
        )

    gcd = np.zeros(population.shape[0])
    num_obj_dimensions = population.shape[1]

    while(num_added < N):
        q = find_best(critical_front, ranks, gcd, gcpd)

        archive[num_added] = population[critical_front[q]]
        del critical_front[q]

        gcd = grid_crowding_distance(differences, num_obj_dimensions)

        ranks = gr_adjustment(
                q, coords, ranks, differences, num_obj_dimensions
            )

        num_added += 1

    return archive


def find_best(population, ranks, gcd, gcpd):
    """
    Uses the grid metrics to determine the "best" individual in the population.
    Intended for use in environment selection for finding the next best
    individual in the critical front.
    """

    pop_arr = np.array(population)

    q = 0
    for p in range(1, pop_arr.shape[0]):
        # first check ranks
        if ranks[p] < ranks[q]:
            q = p
        elif ranks[p] == ranks[q]:
            # use GCD to break a tie in ranks
            if gcd[p] < gcd[q]:
                q = p
            elif gcd[p] == gcd[q]:
                # if necessary, use GCPD as the final decider
                if gcpd[p] < gcpd[q]:
                    q = p

    return q


def breed(archive, archive_costs, coords, gcd, N):
    """
    Builds a population of size N by breeding individuals in the archive.

    Args:
        archive: (np.arr) the archived individuals
        archive_costs: (np.arr) the costs of the individuals in the archive
        coords: (np.arr) grid coordinates of the archived individuals
        gcd: (np.arr) grid crowding density values of archived individuals
        N: (int) number of individuals that should be in the GA population

    Returns:
        population: (np.arr) the population

    """

    population = np.empty((N, archive.shape[1]))

    archive_indices = list(range(archive.shape[0]))
    gene_indices = list(range(archive.shape[1]))

    for i in range(N):
        # generate candidates for mom/dad
        c1, c2, c3, c4 = random.sample(archive_indices, 4)

        # tournament selection to select mom/dad from each pair of candidates
        mom_idx = tournament_selection(c1, c2, archive_costs, coords, gcd)
        dad_idx = tournament_selection(c3, c4, archive_costs, coords, gcd)

        mom = archive[mom_idx]
        dad = archive[dad_idx]

        # do two-point crossover to produce a child
        cx_points = sorted(random.sample(gene_indices, 2))

        child = mom.copy()
        child[cx_points[0]:cx_points[1]] = dad[cx_points[0]:cx_points[1]].copy()

        population[i] = child

    return population


def mutate(population, mu, sigma, mutProb):
    """
    Probabalistically adds gaussian noise to each gene of the individual,
    where the noise is sampled from a normal distribution of mean=mu, std=sigma.
    If mu and/or sigma is array of values, it specifies the mean/std for each
    gene of the individual independently.

    Note: changes population in place, doesn't use a copy

    Args:
        population: (np.arr)
           the original population of individuals

        mu: (float or np.arr)
            the means of the gaussian noise for each gene

        sigma:  (float or np.arr)
            the std of the gaussian noise for each gene

        mutProb: (float)
            the probability of mutating a given gene of a given individual

    Returns:
        a mutated version of the population
    """

    mutations = np.random.normal(loc=mu, scale=sigma, size=population.shape)
    mutated_indices = np.where(np.random.random(population.shape) < mutProb)
    population[mutated_indices] += mutations[mutated_indices]


def build_stats_and_log():
    """Initialize DEAP Statistics and Logbook objects"""

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "size", "min", "max", "avg", "std",# "num_active"

    return stats, logbook


def prepare_true_values(node_manager, world_comm, is_master):
    local_true_eng = {}
    local_true_ref = {}

    for key in node_manager.loaded_structures:
        local_true_eng[key] = node_manager.get_true_value('energy', key)
        local_true_ref[key] = node_manager.get_true_value('ref_struct', key)

    local_true_values = {
            'energy': local_true_eng,
            'ref_struct': local_true_ref,
        }

    all_true_values = world_comm.gather(local_true_values, root=0)

    if is_master:
        all_eng = {}
        # all_fcs = {}
        all_ref = {}
        ref_names = {}

        for d in all_true_values:
            all_eng.update(d['energy'])
            # all_fcs.update(d['forces'])
            all_ref.update(d['ref_struct'])

        true_values = {
                'energy': all_eng,
                # 'forces': all_fcs,
                'ref_struct': all_ref,
            }

    else:
        true_values = None


    return true_values


def collect_structure_names(node_manager, world_comm, is_master):
    local_structs = world_comm.gather(node_manager.loaded_structures, root=0)

    if is_master:
        all_struct_names = []
        for slist in local_structs:
            all_struct_names += slist
    else:
        all_struct_names = None

    all_struct_names = world_comm.bcast(all_struct_names, root=0)

    return all_struct_names


def initialize_population(template, N):

    where = np.where(template.active_mask)[0]

    population = np.empty((N, len(template.active_mask)))

    tmp = template.pvec.copy()

    for i in range(N):
        tmp[where] = np.array(template.generate_random_instance()[where])

        population[i] = tmp.copy()

    return population


def print_statistics(pop, gen_num, stats, logbook, active_knots):
    """Use Statistics and Logbook objects to output results to screen"""

    record = stats.compile(pop)

    logbook.record(
        gen=gen_num, size=len(pop), **record, num_active=active_knots
    )

    print(logbook.stream, flush=True)


def build_objective_function(template, all_struct_names, node_manager,
                             world_comm, is_master, true_values):

    def fxn_wrap(population, return_ni=False, output=False, penalty=False):

        pop = world_comm.bcast(np.atleast_2d(population), root=0)

        manager_energies = node_manager.compute(
            'energy', node_manager.loaded_structures, pop, template.u_ranges
        )

        force_costs = np.array(list(node_manager.compute(
            'forces', node_manager.loaded_structures, pop, template.u_ranges
        ).values()))

        eng = np.vstack([retval[0] for retval in manager_energies.values()])

        ni = [retval[1] for retval in manager_energies.values()]
        ni_stats = src.partools.calculate_ni_stats(ni, template)

        c_min_ni = ni_stats[0]
        c_max_ni = ni_stats[1]
        c_avg_ni = ni_stats[2]
        # c_ni_var = ni_stats[3]
        c_frac_in = ni_stats[4]

        mgr_eng = world_comm.gather(eng, root=0)
        mgr_force_costs = world_comm.gather(force_costs, root=0)

        mgr_min_ni = world_comm.gather(c_min_ni, root=0)
        mgr_max_ni = world_comm.gather(c_max_ni, root=0)
        mgr_avg_ni = world_comm.gather(c_avg_ni, root=0)
        # mgr_ni_var = world_comm.gather(c_ni_var, root=0)
        mgr_frac_in = world_comm.gather(c_frac_in, root=0)

        objective_values = 0
        max_ni = 0
        min_ni = 0
        avg_ni = 0

        if is_master:
            all_eng = np.vstack(mgr_eng)
            all_force_costs = np.vstack(mgr_force_costs)

            min_ni = np.min(np.dstack(mgr_min_ni), axis=2).T
            max_ni = np.max(np.dstack(mgr_max_ni), axis=2).T
            avg_ni = np.average(np.dstack(mgr_avg_ni), axis=2).T
            # ni_var = np.min(np.dstack(mgr_ni_var), axis=2).T
            frac_in = np.sum(np.dstack(mgr_frac_in), axis=2).T

            objective_values = np.zeros(
                (len(pop), len(all_struct_names)*2 + 2)
            )

            lambda_pen = 1000

            ns = len(all_struct_names)

            objective_values[:, -frac_in.shape[1]:] = lambda_pen*abs(ns-frac_in)

            for fit_id, name in enumerate(all_struct_names):
                # force "costs" are the RMSE between the calculated and expected
                objective_values[:, 2*fit_id + 1] = all_force_costs[fit_id]

                # energy "costs" are the difference in energy differences

                ref_name = true_values['ref_struct'][name]

                s_id = all_struct_names.index(name)
                r_id = all_struct_names.index(ref_name)

                true_ediff = true_values['energy'][name]
                comp_ediff = all_eng[s_id] - all_eng[r_id]

                objective_values[:, 2*fit_id] = abs(true_ediff - comp_ediff)

        if is_master:
            if not penalty:
                objective_values = objective_values[:, :-2]

        if output:
            if is_master:
                tmp = np.sum(objective_values, axis=1)
                print("{} {} {}".format(
                        np.min(tmp), np.max(tmp), np.average(tmp)
                    ),
                )

        if return_ni:
            return objective_values, max_ni, min_ni, avg_ni
        else:
            return objective_values

    return fxn_wrap

