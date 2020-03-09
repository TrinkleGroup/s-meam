import os
import numpy as np
from scipy.sparse import diags
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline

def build_evaluation_functions(
        template, all_struct_names, node_manager, world_comm, is_master, true_values,
        parameters, manager_comm
):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    # @profile
    def fxn_wrap(master_pop, weights, return_ni=False, output=False,
            penalty=False):
        """Master: returns all potentials for all structures.

        NOTE: the order of 'weights' should match the order of the database
        entries

        """

        # Node heads need the population, since they //-ize over structs
        if node_manager.is_node_master:
            pop = manager_comm.bcast(master_pop, root=0)
            pop = np.atleast_2d(pop)

            only_eval_on_head = (pop.shape[0] < parameters['PROCS_PER_NODE_MANAGER'])
        else:
            pop = None
            only_eval_on_head = None

        only_eval_on_head = world_comm.bcast(only_eval_on_head, root=0)

        # TODO: shouldn't update these every time, only when dbOpt needs to
        node_manager.weights = dict(zip(
            node_manager.loaded_structures,
            weights[:len(node_manager.loaded_structures)]
        ))

        manager_energies = node_manager.compute(
            'energy', node_manager.loaded_structures, pop, template.u_ranges,
            stress=True
        )

        manager_force_costs = node_manager.compute(
            'forces', node_manager.loaded_structures, pop, template.u_ranges
        )

        fitnesses = np.ones((parameters['POP_SIZE'], template.ntypes))
        min_ni = np.ones_like(fitnesses)*-1
        max_ni = np.ones_like(fitnesses)
        avg_ni = np.zeros_like(fitnesses)

        # world_comm.Barrier()

        # min_ni, max_ni, avg_ni, ni_var, frac_in = calculate_ni_stats(
        #     node_manager, is_master, manager_comm, only_eval_on_head
        # )

        if manager_energies is None:
        # if only_eval_on_head and not node_manager.is_node_head:
            # this indicates that you were an MPI rank that didn't have to
            # evaluate anything

            if return_ni:
                return fitnesses, max_ni, min_ni, avg_ni
            else:
                return fitnesses

        force_costs = np.array(list(manager_force_costs.values()))

        unsorted_energies = [retval[0] for retval in manager_energies.values()]
        eng = [  # sort according to structure name
            x for _, x in sorted(
                zip(node_manager.loaded_structures, unsorted_energies)
            )
        ]

        unsorted_stresses = [retval[1] for retval in manager_energies.values()]

        stresses = [  # sort according to structure name
            x for _, x in sorted(
                zip(node_manager.loaded_structures, unsorted_stresses)
            )
        ]

        sorted_names = sorted(node_manager.loaded_structures)

        if node_manager.is_node_master:
            mgr_eng = manager_comm.gather(eng, root=0)
            mgr_stress = manager_comm.gather(stresses, root=0)
            mgr_force_costs = manager_comm.gather(force_costs, root=0)

            mgr_names_list = manager_comm.gather(sorted_names, root=0)

        if is_master:
            all_names = np.concatenate(mgr_names_list)

            # note: can't stack mgr_fcs b/c different dimensions per struct
            all_eng = np.vstack(mgr_eng)

            all_stress_costs = np.vstack(mgr_stress)

            # TODO: why are all_eng sorted, but forces aren't?

            all_eng = all_eng[np.argsort(all_names), :]
            all_force_costs = np.vstack(mgr_force_costs)

            # note that ni stats should be in the shape (ntypes, popsize)

            fitnesses = np.zeros(
                (
                    len(pop),
                    # len()*3 for energy/force/stress, 3*ntypes for
                    # frac_in/small_var/big_var, +1 for LHS phi deriv

                    len(all_struct_names)*3 \
                    + 3*template.ntypes \
                    + 1
                )
            )

            # assumes that 'weights' has the same order as all_struct_names
            for fit_id, (name, weight) in enumerate(zip(
                all_struct_names, weights
                )):

                ref_name = true_values['ref_struct'][name]

                s_id = all_struct_names.index(name)
                r_id = all_struct_names.index(ref_name)

                true_ediff = true_values['energy'][name]
                comp_ediff = all_eng[s_id] - all_eng[r_id]

                # NOTE: error weights are applied AFTER logging

                # tmp = (comp_ediff - true_ediff) ** 2
                tmp = abs(comp_ediff - true_ediff)

                fitnesses[:, 3*fit_id] = tmp#*parameters['ENERGY_WEIGHT']

                fitnesses[:, 3*fit_id+1] = \
                    all_force_costs[fit_id]#*parameters['FORCES_WEIGHT']

                # fitnesses[:, 8*fit_id+2:8*fit_id+8] = \
                #     all_stress_costs[fit_id]*parameters['STRESS_WEIGHT']
                fitnesses[:, 3*fit_id+2] = \
                    all_stress_costs[fit_id]#*parameters['STRESS_WEIGHT']

            lambda_pen = parameters['NI_PENALTY']

            # penalize fraction outside of U domains
            # fitnesses[:, -template.ntypes*3-1:-template.ntypes*2-1] = lambda_pen*abs(1-frac_in)

            fitnesses[:, -1] = 10*np.linalg.norm(master_pop, axis=1)

            # # penalize too small variance
            # fitnesses[:, -template.ntypes*2-1:-template.ntypes-1] = \
            #         lambda_pen*np.clip(0.05-ni_var, 0, None)

            # # penalize too big variance
            # fitnesses[:, -template.ntypes-1:-1] = lambda_pen*np.clip(
            #     ni_var-1, 0, None)

            # # penalize non-negative LHS phi derivatives; this is done to make
            # # sure that the potential has repulsive forces for small pair
            # # distances, even if the database doesn't have data like this.

            # fitnesses[:, -1] = lambda_pen*np.clip(
            #     pop[:, template.phi_lhs_deriv_indices], 0, None
            # ).sum(axis=1)*0

        if is_master:
            if not penalty:
                fitnesses = fitnesses[:, :-template.ntypes*2-1]

        if output:
            if is_master:
                tmp = np.sum(fitnesses, axis=1)
                print("{} {} {}".format(
                        np.min(tmp), np.max(tmp), np.average(tmp)
                    ),
                )

        if return_ni:
            return fitnesses, max_ni, min_ni, avg_ni
        else:
            return fitnesses

    def grad_wrap(master_pop, weights):
        """Evalautes the gradient for all potentials in the population"""

        pop = world_comm.bcast(master_pop, root=0)
        pop = np.atleast_2d(pop)

        node_manager.weights = dict(zip(
            node_manager.loaded_structures,
            weights[:len(node_manager.loaded_structures)]
        ))

        manager_energies = node_manager.compute(
            'energy', node_manager.loaded_structures, pop, template.u_ranges,
            stress=True
        )

        manager_eng_grads = node_manager.compute(
            'energy_grad', node_manager.loaded_structures, pop, template.u_ranges
        )

        manager_fcs_grads = node_manager.compute(
            'forces_grad', node_manager.loaded_structures, pop, template.u_ranges
        )

        gradient = 0

        eng = np.vstack([retval[0] for retval in manager_energies.values()])
        eng_grads = np.vstack(list(manager_eng_grads.values()))
        fcs_grads = np.vstack(list(manager_fcs_grads.values()))

        mgr_eng = world_comm.gather(eng, root=0)
        mgr_eng_grad = world_comm.gather(eng_grads, root=0)
        mgr_fcs_grad = world_comm.gather(fcs_grads, root=0)

        if is_master:
            all_eng = np.vstack(mgr_eng)
            all_eng_grad = np.dstack(mgr_eng_grad).T
            all_fcs_grad = np.dstack(mgr_fcs_grad).T

            gradient = np.zeros((
                len(pop),
                template.pvec_len,
                len(all_struct_names)*2
            ))


            for fit_id, (name, weight) in enumerate(zip(
                    all_struct_names, weights
                )):

                # ref_name = database[name].attrs['ref_struct']
                # ref_name = true_values['ref_struct'][name]

                # find index of structures to know which energies to use
                s_id = all_struct_names.index(name)
                # r_id = all_struct_names.index(ref_name)


                gradient[:, :, 2*fit_id + 1] += all_fcs_grad[:, :, s_id]

                # true_ediff = database[name]['true_values']['energy']
                true_ediff = true_values['energy'][name]
                comp_ediff = all_eng[s_id] # - all_eng[r_id]

                eng_err = comp_ediff - true_ediff
                s_grad = all_eng_grad[:, :, s_id]
                # r_grad = all_eng_grad[:, :, r_id]

                gradient[:, :, 2*fit_id] += \
                    (eng_err[:, np.newaxis]*(s_grad)*2)*weight
                    # (eng_err[:, np.newaxis]*(s_grad - r_grad)*2)*weight

            # indices = np.where(template.active_mask)[0]
            # gradient = gradient[:, indices, :].swapaxes(1, 2)
            gradient = gradient.swapaxes(1, 2)

        return gradient

    return fxn_wrap, grad_wrap


def compute_relative_weights(database):
    work_weights = []

    name_list = sorted(list(database.structures.keys()))

    for name in name_list:
        work_weights.append(database.structures[name].natoms)

    work_weights = np.array(work_weights)
    work_weights = work_weights / np.min(work_weights)
    work_weights = work_weights * work_weights  # cost assumed to scale as N^2

    return work_weights, name_list


# def group_database_subsets(database, num_managers):
#     """Groups workers based on evaluation time to help with load balancing.
# 
#     Returns:
#         distributed_work (list): the partitioned work
#         num_managers (int): desired number of managers to be used
#     """
#     # TODO: record scaling on first run, then redistribute to load balance
#     # TODO: managers should just split Database; determine cost later
# 
#     # Number of managers should not exceed the number of structures
#     num_managers = min(num_managers, len(database.structures))
# 
#     unassigned_structs = sorted(list(database.structures.keys()))
# 
#     work_weights, name_list = compute_relative_weights(database)
# 
#     work_per_proc = np.sum(work_weights)  # / num_managers
# 
#     # work_weights = work_weights.tolist()
# 
#     # Splitting code taken from SO: https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
#     cum_arr = np.cumsum(work_weights) / np.sum(work_weights)
# 
#     idx = np.searchsorted(cum_arr,
#                           np.linspace(0, 1, num_managers, endpoint=False)[1:])
#     name_chunks = np.split(unassigned_structs, idx)
#     weight_chunks = np.split(work_weights, idx)
# 
#     subsets = []
#     grouped_work = []
# 
#     for name_chunk, work_chunk in zip(name_chunks, weight_chunks):
#         cumulated_work = 0
# 
#         names = []
# 
#         # while unassigned_structs and (cumulated_work < work_per_proc):
#         for n, w in zip(name_chunk, work_chunk):
#             names.append(n)
#             cumulated_work += w
# 
#         mini_database = Database.manual_init(
#             {name: database.structures[name] for name in names},
#             {name: database.true_energies[name] for name in names},
#             {name: database.true_forces[name] for name in names},
#             {name: database.weights[name] for name in names},
#             database.reference_struct,
#             database.reference_energy
#         )
# 
#         subsets.append(mini_database)
#         grouped_work.append(cumulated_work)
# 
#     return subsets, grouped_work


def compute_procs_per_subset(struct_natoms, total_num_procs, method='natoms'):
    """
    Should split the number of processors into groups based on the relative
    evaluation costs of each Database subset.

    Possible methods for computing weights:
        'natoms': number of atoms in cell
        'time': running tests to see how long each one takes

    Args:
        struct_natoms (list[int]): number of atoms in each structure
        total_num_procs (int): total number of processors available
        method (str): 'natoms' or 'time' (default is 'natoms')

    Returns:
        ranks_per_struct (list[list]): processor ranks for each struct
    """

    structures = list(struct_natoms)
    num_structs = len(struct_natoms)

    # Note: ideally, should have at least as many cores as struct   s
    supported_methods = ['natoms', 'time']

    # Error checking on  desired method
    if method == 'natoms':
        weights = [n * n for n in struct_natoms]
        weights /= np.sum(weights)
    elif method == 'time':
        raise NotImplementedError("Oops ...")
    else:
        raise ValueError(
            "Invalid method. Must be either", ' or '.join(supported_methods)
        )

    # TODO: possible problem where too many large structs get passed to one node
    # TODO: dimer got more workers than a trimer

    # Sort weights and structs s.t. least expensive is last
    sort = np.argsort(weights)[::-1]

    weights = weights[sort]
    struct_natoms = [s for _, s in sorted(zip(sort, struct_natoms))]

    # Compute how many processors would "ideally" be needed for each struct
    num_procs_needed_per_struct = weights * total_num_procs

    # Every struct gets at least once processor
    num_procs_needed_per_struct = np.clip(num_procs_needed_per_struct, 1, None)
    num_procs_needed_per_struct = np.ceil(num_procs_needed_per_struct)

    # if too many procs assigned
    if np.sum(num_procs_needed_per_struct) > total_num_procs:
        while np.sum(num_procs_needed_per_struct) != total_num_procs:
            max_idx = np.argmax(num_procs_needed_per_struct)
            num_procs_needed_per_struct[max_idx] -= 1
    else:  # not enough procs assigned
        while np.sum(num_procs_needed_per_struct) != total_num_procs:
            min_idx = np.argmin(num_procs_needed_per_struct)
            num_procs_needed_per_struct[min_idx] += 1

    # Create list of indices for np.split()
    num_procs_needed_per_struct = num_procs_needed_per_struct.astype(int)
    split_indices = np.cumsum(num_procs_needed_per_struct)

    return np.split(np.arange(total_num_procs), split_indices)[:-1]


def build_objective_function(testing_database, error_fxn, is_master):
    """In the first paper, this was the log of the Bayesian estimate of the
    mean. In the second paper, it was the logistic function C(x, eps) of the
    Bayesian estimate.

    C(x, eps) = 1 / (1 + exp(-m*(x/eps - 1)))

    Here, x is the error in the estimate of the property
    eps is the 'decision boundary' (see second paper), and
    m is the 'stiffness' (chosen as 2 in Pinchao's code)

    Currently, this code applies the logistic function, but replaces the
    Bayesian estimate of the mean simply with the MLE value.
    """

    m = 2
    logistic = lambda x, eps: 1 / (1 + np.exp(-m * (x / eps - 1)))

    def objective_fxn(mle, weights):
        """Only designed for one MLE at a time (currently)"""
        errors = error_fxn(mle, weights)

        values = 0
        if is_master:
            errors = errors.ravel()
            values = np.zeros(len(errors))

            for entry_id, entry in enumerate(testing_database.entries):
                if entry.type == 'energy':
                    values[entry_id] = logistic(errors[entry_id], 0.2)
                elif entry.type == 'forces':
                    values[entry_id] = logistic(errors[entry_id], 0.5)
                else:
                    raise ValueError("Not a valid entry type")

        return np.sum(values)

    return objective_fxn


def rescale_ni(pots, min_ni, max_ni, template):
    """
    Rescales the rho/f/g splines to try to fit it all ni into the U domain of
    [-1, 1].

    Note:
        - order of min_ni and max_ni is assumed to be the same as pots
        - pots is assumed to be only the active parameters of the potentials

    Args:
        pots: (NxP array) the N starting potentials of P parameters
        min_ni: (Nx1 array) the minimum ni sampled for each of the N potential
        max_ni: (Nx1 array) the maximum ni sampled for each of the N potential
        template: (Template) template object, used for indexing

    Returns:
        updated_pots: (NxP array) the potentials with rescaled rho/f/g splines
    """

    pots = np.array(pots)

    # ni row format: [max_A, max_B, ..., min_A, min_B, ...]
    ni = np.hstack([max_ni, min_ni])

    nt = template.ntypes

    per_type_slice = [ni[:, i::nt] for i in range(nt)]

    per_type_argmax = [
        np.argmax(abs(ni_slice), axis=1) for ni_slice in per_type_slice
    ]

    per_type_extracted = np.vstack([
        np.choose(argmax, ni_slice.T) for (argmax, ni_slice) in
        zip(per_type_argmax, per_type_slice)
    ])

    scale = np.min(per_type_extracted, axis=0)

    scale /= 1.2  # 1.2 to try to get U domain overlap
    signs = np.sign(scale)

    pots[:, template.rho_indices] /= \
        signs[:, np.newaxis]*abs(scale[:, np.newaxis])

    pots[:, template.f_indices] /= \
        abs(scale[:, np.newaxis]) ** (1. / 3)

    pots[:, template.g_indices] /= \
        signs[:, np.newaxis]*(abs(scale[:, np.newaxis]) ** (1. / 3))

    return pots


def shift_u(min_ni, max_ni):
    """
    Computes the new U domains using the min/max ni from the best potential.
    Assumes that 'min_ni' and 'max_ni' are sorted so that the best potentials
    are first.

    Args:
        min_ni: (Nx1 array) the minimum ni sampled for each of the N potential
        max_ni: (Nx1 array) the maximum ni sampled for each of the N potential

    Returns:
        new_u_domains: (num_atom_types length list) each element in the list
        is a length-2 tuple of the min/max values for the U spline of one
        atom type
    """

    tmp_min_ni = min_ni[0]
    tmp_max_ni = max_ni[0]

    new_u_domains = [
        (tmp_min_ni[i], tmp_max_ni[i]) for i in range(2)]

    for k, tup in enumerate(new_u_domains):
        tmp_tup = []

        size = abs(tup[1] - tup[0])

        for kk, lim in enumerate(tup):
            if kk == 0:  # lower bound
                # add some to the lower bound
                tmp = lim + 0.1*size

                if tmp > 0: tmp = 0

                tmp_tup.append(tmp)
            elif kk == 1:
                # subtract some from the upper bound
                tmp = lim - 0.1*size

                if tmp < 1: tmp = 1

                tmp_tup.append(tmp)

        new_u_domains[k] = tuple(tmp_tup)

    return new_u_domains


def mcmc(population, weights, cost_fxn, template, T,
         parameters, active_tags, is_master, start_step=0,
         cooling_rate=1, T_min=0, suffix="", max_nsteps=None, penalty=False):
    """
    Runs an MCMC optimization on the given subset of the parameter vectors.
    Stopping criterion is either 20 steps without any improvement,
    or max_nsteps reached.

    Args:
        max_nsteps: (int) maximum number of steps to run
        population: (np.arr) 2d array where each row is a potential
        weights: (np.arr) weights for each term in cost function
        cost_fxn: (callable) cost funciton
        template: (Template) template containing potential information
        T: (float) normalization factor
        move_prob: (float) probability that any given knot will be changed
        move_scale: (float) stdev for normal dist for each knot
        active_tags: (list) integer tags specifying active splines
        is_master: (bool) used for MPI communication
        start_step: (int) current step number; used for outputting
        cooling_rate: (float) cooling rate if performing simulated annealing
        T_min: (float) minimum allowed normalization constant

    Returns:
        final: (np.arr) the final parameter vectors

    """

    if max_nsteps is None:
        max_nsteps = parameters['MCMC_NSTEPS']

    move_prob = parameters['MOVE_PROB']
    move_scale = parameters['MOVE_SCALE']
    checkpoint_freq = parameters['CHECKPOINT_FREQ']

    if is_master:
        population = np.array(population)

        active_indices = []

        for tag in active_tags:
            active_indices.append(
                np.where(template.spline_tags == tag)[0]
            )

        active_indices = np.concatenate(active_indices)

        current = population[:, active_indices]

        tmp = population.copy()
        tmp_trial = tmp.copy()

        num_without_improvement = 0

        current_best = np.infty

    else:
        current = None
        tmp = None
        tmp_trial = None

    if suffix in ['rws', 'rho', 'U', 'f', 'g']:
        def move_proposal(inp, move_scale):
            # choose a single knot from each potential
            rnd_indices = np.random.randint(
                inp.shape[1], size=inp.shape[0]
            )

            trial_position = inp.copy()
            trial_position[:, rnd_indices] += np.random.normal(
                scale=move_scale, size=inp.shape[0]
            )

            return trial_position
    else:
        def move_proposal(inp, move_scale):
            # choose a random collection of knots from each potential
            mask = np.random.choice(
                [True, False], size=(inp.shape[0], inp.shape[1]),
                p=[move_prob, 1 - move_prob]
            )

            trial = inp.copy()
            trial[mask] = trial[mask] + np.random.normal(scale=move_scale)

            return trial

    for step_num in range(max_nsteps):

        if is_master:
            trial = move_proposal(current, move_scale)
            tmp_trial[:, active_indices] = trial
        else:
            trial = None

        current_cost, c_max_ni, c_min_ni, c_avg_ni = cost_fxn(
            tmp, weights, return_ni=True, penalty=parameters['PENALTY_ON']
        )

        trial_cost, t_max_ni, t_min_ni, t_avg_ni = cost_fxn(
            tmp_trial, weights, return_ni=True, penalty=parameters['PENALTY_ON']
        )

        if is_master:
            prev_best = current_best

            current_cost = np.sum(current_cost, axis=1)
            trial_cost = np.sum(trial_cost, axis=1)

            ratio = np.exp((current_cost - trial_cost) / T)
            where_auto_accept = np.where(ratio >= 1)[0]

            where_cond_accept = np.where(
                np.random.random(ratio.shape[0]) < ratio
            )[0]

            # update population
            current[where_auto_accept] = trial[where_auto_accept]
            current[where_cond_accept] = trial[where_cond_accept]

            # update costs
            current_cost[where_auto_accept] = trial_cost[where_auto_accept]
            current_cost[where_cond_accept] = trial_cost[where_cond_accept]

            # update ni
            c_max_ni[where_auto_accept] = t_max_ni[where_auto_accept]
            c_max_ni[where_cond_accept] = t_max_ni[where_cond_accept]

            c_min_ni[where_auto_accept] = t_min_ni[where_auto_accept]
            c_min_ni[where_cond_accept] = t_min_ni[where_cond_accept]

            c_avg_ni[where_auto_accept] = t_avg_ni[where_auto_accept]
            c_avg_ni[where_cond_accept] = t_avg_ni[where_cond_accept]

            current_best = current_cost[np.argmin(current_cost)]

            print(
                start_step + step_num, "{:.3f}".format(T),
                np.min(current_cost), np.max(current_cost),
                np.average(current_cost), suffix,
                # num_accepted / (step_num + 1) / parameters['POP_SIZE'],
                flush=True
            )

            T = np.max([T_min, T*cooling_rate])

            if is_master:
                tmp[:, active_indices] = current

                if (start_step + step_num) % checkpoint_freq == 0:
                    checkpoint(
                        tmp, current_cost, c_max_ni, c_min_ni, c_avg_ni,
                        start_step + step_num, parameters, template,
                        max_nsteps
                    )

    if is_master:
        population[:, active_indices] = current

    return population

def checkpoint(population, costs, max_ni, min_ni, avg_ni, i, parameters,
    template, max_nsteps, suffix=""):
    """Saves information to files for later use"""

    # save population
    digits = np.floor(np.log10(max_nsteps))

    format_str = os.path.join(
        parameters['SAVE_DIRECTORY'],
        'pop_{0:0' + str(int(digits) + 1)+ 'd}.dat' + suffix
    )

    np.savetxt(format_str.format(i), population)

    # save costs
    format_str = os.path.join(
        parameters['SAVE_DIRECTORY'],
        'fitnesses_{0:0' + str(int(digits) + 1) + 'd}.dat' + suffix
    ).format(i)

    with open(format_str, 'w') as f:
        np.savetxt(f, np.atleast_2d(costs))

    # output ni to file
    with open(parameters['NI_TRACE_FILE_NAME'] + suffix, 'ab') as f:
        np.savetxt(
            f,
            np.atleast_2d(np.concatenate(
                [
                    min_ni.ravel(),
                    max_ni.ravel(),
                    avg_ni.ravel(),
                    np.concatenate(template.u_ranges)
                ]
            ))
        )

def build_M(num_x, dx, bc_type):
    """Builds the A and B matrices that are needed to find the function
    derivatives at all knot points. A and B come from the system of equations
    that comes from matching second derivatives at internal spline knots
    (using Hermitian cubic splines) and specifying boundary conditions

        Ap' = Bk

    where p' is the vector of derivatives for the interpolant at each knot
    point and k is the vector of parameters for the spline (y-coordinates of
    knots and second derivatives at endpoints).

    Let N be the number of knot points

    In addition to N equations from internal knots and 2 equations from boundary
    conditions, there are an additional 2 equations for requiring linear
    extrapolation outside of the spline range. Linear extrapolation is
    achieved by specifying a spline who's first derivatives match at each end
    and whose endpoints lie in a line with that derivative.

    With these specifications, A and B are both (N+2, N+2) matrices

    A's core is a tridiagonal matrix with [h''_10(1), h''_11(1)-h''_10(0),
    -h''_11(0)] on the diagonal which is dx*[2, 8, 2] based on their definitions

    B's core is tridiagonal matrix with [-h''_00(1), h''_00(0)-h''_01(1),
    h''_01(0)] on the diagonal which is [-6, 0, 6] based on their definitions

    Note that the dx is a scaling factor defined as dx = x_k+1 - x_k, assuming
    uniform grid points and is needed to correct for the change into the
    variable t, defined below.

    and functions h_ij are defined as:

        h_00 = (1+2t)(1-t)^2
        h_10 = t (1-t)^2
        h_01 = t^2 (3-2t)
        h_11 = t^2 (t-1)

        with t = (x-x_k)/dx

    which means that the h''_ij functions are:

        h''_00 = 12t - 6
        h''_10 = 6t - 4
        h''_01 = -12t + 6
        h''_11 = 6t - 2

    Args:
        num_x (int): the total number of knots

        dx (float): knot spacing (assuming uniform spacing)

        bc_type (tuple): tuple of 'natural' or 'fixed'

    Returns:
        M (np.arr):
            A^(-1)B
    """

    n = num_x - 2

    if n <= 0:
        raise ValueError("the number of knots must be greater than 2")

    # note that values for h''_ij(0) and h''_ij(1) are substituted in
    # TODO: add checks for non-grid x-coordinates

    bc_lhs, bc_rhs = bc_type
    bc_lhs = bc_lhs.lower()
    bc_rhs = bc_rhs.lower()

    A = np.zeros((n + 2, n + 2))
    B = np.zeros((n + 2, n + 4))

    # match 2nd deriv for internal knots
    fillA = diags(np.array([2, 8, 2]), [0, 1, 2], (n, n + 2))
    fillB = diags([-6, 0, 6], [0, 1, 2], (n, n + 2))
    A[1:n+1, :n+2] = fillA.toarray()
    B[1:n+1, :n+2] = fillB.toarray()

    # equation accounting for lhs bc
    if bc_lhs == 'natural':
        A[0,0] = -4; A[0,1] = -2
        B[0,0] = 6; B[0,1] = -6; B[0,-2] = 1
    elif bc_lhs == 'fixed':
        A[0,0] = 1/dx;
        B[0,-2] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    # equation accounting for rhs bc
    if bc_rhs == 'natural':
        A[-1,-2] = 2; A[-1,-1] = 4
        B[-1,-4] = -6; B[-1,-3] = 6; B[-1,-1] = 1
    elif bc_rhs == 'fixed':
        A[-1,-1] = 1/dx
        B[-1,-1] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    A *= dx

    # M = A^(-1)B
    return np.dot(np.linalg.inv(A), B)

def local_minimization(
        pop_to_opt, master_pop, template, fxn, grad, weights, world_comm,
        is_master, nsteps=20, lm_output=False, penalty=False
    ):

    # TODO: remove master_pop as argument; it's unused

    # NOTE: if LM throws size errors, you probaly need to add more padding
    pad = 100

    def lm_fxn_wrap(raveled_pop, original_shape):
        if is_master:
            if lm_output:
                print('LM step: ', end="", flush=True)

            # full = template.insert_active_splines(
            #     raveled_pop.reshape(original_shape)
            # )
            full = master_pop.copy()
            full[:, np.where(template.active_mask)[0]] = raveled_pop.reshape(original_shape)
        else:
            full = None

        val = fxn(full, weights, output=lm_output, penalty=penalty)

        if is_master:
            if penalty:
                val = val[:, :-3]

        val = world_comm.bcast(val, root=0)

        # pad with zeros since num structs is less than num knots
        tmp = np.concatenate([val.ravel(), np.zeros(pad * original_shape[0])])

        return tmp

    def lm_grad_wrap(raveled_pop, original_shape):

        if is_master:
            # full = template.insert_active_splines(
            #     raveled_pop.reshape(original_shape)
            # )
            full = master_pop.copy()
            full[:, np.where(template.active_mask)[0]] = raveled_pop.reshape(original_shape)
            # full = raveled_pop.reshape(original_shape)
        else:
            full = None

        grads = grad(full, weights)

        if is_master:
            grads = grads[:, :, np.where(template.active_mask)[0]]
            # gradient = gradient[:, indices, :].swapaxes(1, 2)

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

    # uncomment if want to use centered differences for gradient approximation
    # lm_grad_wrap = '2-point'

    pop_to_opt = world_comm.bcast(pop_to_opt, root=0)
    pop_to_opt = np.array(pop_to_opt)

    opt_results = least_squares(
        lm_fxn_wrap, pop_to_opt.ravel(), lm_grad_wrap,
        method='lm', max_nfev=nsteps, args=(pop_to_opt.shape,)
    )

    if is_master:
        new_pop = opt_results['x'].reshape(pop_to_opt.shape)

        tmp = master_pop.copy()
        tmp[:, np.where(template.active_mask)[0]] = pop_to_opt

        new_tmp = master_pop.copy()
        new_tmp[:, np.where(template.active_mask)[0]] = new_pop

        tmp = np.atleast_2d(tmp)
        new_tmp = np.atleast_2d(new_tmp)

    else:
        tmp = None
        new_tmp = None
        new_pop = None

    org_fits = fxn(tmp, weights, penalty=penalty)
    new_fits = fxn(new_tmp, weights, penalty=penalty)

    if is_master:
        updated_pop = []

        for i, ind in enumerate(new_pop):
            if np.sum(new_fits[i]) < np.sum(org_fits[i]):
                updated_pop.append(new_pop[i])
            else:
                updated_pop.append(pop_to_opt[i])
                updated_pop[i] = updated_pop[i]

        final_pop = np.array(updated_pop)
    else:
        final_pop = None

    return final_pop

# @profile
def calculate_ni_stats(node_manager, is_master, manager_comm, only_eval_on_head):

    if is_master:
        min_ni = np.zeros(
            (node_manager.full_popsize, node_manager.template.ntypes)
        )

        max_ni = np.zeros_like(min_ni)
        avg_ni = np.zeros_like(min_ni)
        ni_var = np.zeros_like(min_ni)

    if only_eval_on_head:
        global_ni = [node_manager.my_ni]
    else:
        physical_node_ni = node_manager.my_head_comm.gather(node_manager.my_ni, root=0)

        if node_manager.is_node_head:

            physical_node_ni = [
                # vstack because shape is (P, natoms), and each worker has a pop
                np.vstack(worker_pop_ni) for worker_pop_ni in zip(*physical_node_ni)
            ]

            node_ni = node_manager.master_to_heads_comm.gather(
                physical_node_ni, root=0
            )

        if node_manager.is_node_master:

            node_ni = [
                # vstack because physical nodes are also parallel over pop
                np.vstack(pnode_pop_ni) for pnode_pop_ni in zip(*node_ni)
            ]

            global_ni = manager_comm.gather(node_ni, root=0)

    if is_master:

        global_ni = [
            # hstack because each NodeManager parallelizes over structures
            np.hstack(nm_struct_ni) for nm_struct_ni in zip(*global_ni)
        ]

        # recall: each entry is for a different atomic type
        min_ni = np.array([np.min(ni, axis=1) for ni in global_ni]).T
        max_ni = np.array([np.max(ni, axis=1) for ni in global_ni]).T
        avg_ni = np.array([np.average(ni, axis=1) for ni in global_ni]).T
        ni_var = np.array([np.std(ni, axis=1)**2 for ni in global_ni]).T

        biggest_min = max(
            [el[0] for el in node_manager.template.u_ranges]
        )

        biggest_max = max(
            [el[1] for el in node_manager.template.u_ranges]
        )

        frac_in = np.zeros_like(min_ni)
        for ii, ni in enumerate(global_ni):  # loop over atom types

            num_in = np.logical_and(
                ni >= biggest_min - 0.1,  # +/- 0.1 for some wiggle room
                ni <= biggest_max + 0.1
            ).sum(axis=1)

            frac_in[:, ii] = num_in / ni.shape[1]

    else:
        min_ni = None
        max_ni = None
        avg_ni = None
        ni_var = None
        frac_in = None

    return min_ni, max_ni, avg_ni, ni_var, frac_in

def calculate_ni_stats2(node_manager, is_master, manager_comm):
    """
    Once everyone has finished their calculations, we can start computing ni
    statistics. First, the master node must gather all of the averages to
    compute the overall average. Second, the master must broadcast the
    overall average to all node heads. The node heads can then compute the
    sum of the squared differences to the mean value, which the master can
    then gather to compute the final variance.

    Args:
        node_manager: NodeManager object on each process
        is_master: True if the process is the global master rank
        manager_comm: communicator between the master rank and the Node masters

    Returns:
        global_variances: (P, ntypes) array of ni variances

    """

    # have each node master gather the statistics from their physical nodes
    node_ni_stats = node_manager.master_to_heads_comm.gather(
        node_manager.ni_shmem_arr, root=0
    )

    # TODO: seems like it would be better to just NOT compute any statistics
    # locally, then you only need a single group operation

    if node_manager.is_node_master:
        node_ni_stats = np.stack(node_ni_stats, axis=1)

        # now gather the ni stats from all NodeManager masters
        node_ni_stats = manager_comm.gather(node_ni_stats, root=0)

    # master now has all of the min/max/avg ni for all structures and pots

    if is_master:
        global_ni_stats = np.stack(node_ni_stats, axis=0)

        # compute the overall average ni for each element type
        overall_average = np.average(global_ni_stats[:, :, :, -1], axis=(0, 1))
    else:
        overall_average = None

    # send to all node masters
    if node_manager.is_node_master:
        overall_average = manager_comm.bcast(overall_average, root=0)

    # send from node masters to all physical nodes
    if node_manager.is_node_head:
        node_manager.master_to_heads_comm.bcast(overall_average, root=0)

    # and finally, from physical node heads down to individual workers since
    # they're the ones that stored the actual ni values

    overall_average = node_manager.my_head_comm.bcast(overall_average, root=0)

    # then compute the variances on each physical node

    # recall that node_manager.my_ni is now a list of length ntypes where each
    # entry is a (P, natoms_per_struct*nstructs) array of ni values

    my_variances = [
        (node_manager.my_ni[ii] - overall_average[ii])**2
        for ii in range(node_manager.template.ntypes)
    ]

    my_variances = [np.sum(var, axis=1) for var in my_variances]

    # my_variances is now a list of length ntypes where each entry is a
    # length-P array of squared deviations

    # now start gather back up to the top
    # also, gather the min/max/avg ni for each potential

    if node_manager.is_node_head:
        # stack along population dimension

        # when we gather, we get a (num_workers, P, ntypes) array of values
        physical_node_variances = np.stack(node_manager.my_head_comm.gather(
            my_variances, root=0
        ), axis=1).sum(axis=0)

    if node_manager.is_node_master:
        node_manager_variances = np.stack(
            node_manager.master_to_heads_comm.gather(
                physical_node_variances, root=0
            )
        ).sum(axis=0)  # stack along a new dimension, then sum along it

    if is_master:
        global_variances = np.stack(
            manager_comm.gather(node_manager_variances, root=0),
        ).sum(axis=0)  # same; stack along new dimension, then sum

        global_min_ni = np.min(global_ni_stats[:, :, :, 0])
        global_max_ni = np.min(global_ni_stats[:, :, :, 1])
        global_avg_ni = np.min(global_ni_stats[:, :, :, 2])

        global_variances /= node_manager.global_num_atoms
    else:
        global_min_ni = None
        global_max_ni = None
        global_avg_ni = None
        global_variances = None

    return global_min_ni, global_max_ni, global_avg_ni, global_variances


def cs_convert_domains(old_u_knots, new_type):

    new_u_knots = []

    for old in old_u_knots:
        if new_type == 0:
            old_x = np.linspace(-1, 1, len(old) - 2)
            new_x = np.linspace(0, 1, len(old) - 2)

            old_cs = CubicSpline(
                old_x, old[:-2],
                bc_type=((1, old[-2]), (1, old[-1]))
            )

            new_y = old_cs(new_x)
            new_bc = [old_cs(new_x[0], 1), old_cs(new_x[-1], 1)]

            new = np.concatenate([new_y, new_bc])

        if new_type == 1:
            old_x = np.linspace(0, 1, len(old) - 2)
            new_x = np.linspace(-1, 1, len(old) - 2)

            old_cs = CubicSpline(
                old_x, old[:-2],
                bc_type=((2, 0), (1, old[-1]))
            )

            new_y = old_cs(new_x)

            old_spacing = old_x[1] - old_x[0]
            new_spacing = new_x[1] - new_x[0]

            midpoint = len(new_y)//2

            if len(new_y) % 2 == 0:
                shift = new_spacing
            else:
                shift = 2*new_spacing

            new_y[:midpoint] = np.linspace(
                old[0] - (midpoint+1)*new_spacing*old[-2],
                old[0] + shift, midpoint
            )

            new_bc = old[-2:]

            new = np.concatenate([new_y, new_bc])

        new_u_knots.append(new)

    return new_u_knots
