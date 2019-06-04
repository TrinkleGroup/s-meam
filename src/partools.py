import os
import numpy as np
from src.database import Database
from src.potential_templates import Template

def build_evaluation_functions(
        potential_template, master_database, all_struct_names, manager,
        is_master,
        is_manager, manager_comm, ref_name
):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn_wrap(master_pop, weights, return_ni=False, output=False,
            penalty=False):
        """Master: returns all potentials for all structures.

        NOTE: the order of 'weights' should match the order of the database
        entries

        """
        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
            pop = np.atleast_2d(pop)
        else:
            pop = None

        # eng = manager.compute_energy(pop)
        # eng, c_max_ni, c_min_ni = manager.compute_energy(pop)
        # eng, c_ni = manager.compute_energy(pop)
        # eng, c_min_ni, c_max_ni, c_avg_ni, c_ni_var = manager.compute_energy(
        #     pop)

        eng, c_min_ni, c_max_ni, c_avg_ni, c_ni_var, c_frac_in = manager.compute_energy(
            pop)
        fcs = manager.compute_forces(pop)

        fitnesses = 0
        max_ni = 0
        min_ni = 0
        avg_ni = 0

        if is_manager:
            # gathers everything on master process

            mgr_eng = manager_comm.gather(eng, root=0)
            mgr_fcs = manager_comm.gather(fcs, root=0)

            mgr_min_ni = manager_comm.gather(c_min_ni, root=0)
            mgr_max_ni = manager_comm.gather(c_max_ni, root=0)
            mgr_avg_ni = manager_comm.gather(c_avg_ni, root=0)
            mgr_ni_var = manager_comm.gather(c_ni_var, root=0)
            mgr_frac_in = manager_comm.gather(c_frac_in, root=0)

            if is_master:
                # note: can't stack mgr_fcs b/c different dimensions per struct
                all_eng = np.vstack(mgr_eng)
                all_fcs = mgr_fcs

                # do operations so that the final shape is (2, num_pots)
                min_ni = np.min(np.dstack(mgr_min_ni), axis=2).T
                max_ni = np.max(np.dstack(mgr_max_ni), axis=2).T
                avg_ni = np.average(np.dstack(mgr_avg_ni), axis=2).T
                ni_var = np.min(np.dstack(mgr_ni_var), axis=2).T
                frac_in = np.sum(np.dstack(mgr_frac_in), axis=2).T

                fitnesses = np.zeros(
                    (len(pop), len(master_database.entries) + 4)
                )

                # maybe make the error U[] - var?

                lambda_pen = 10000

                ns = len(master_database.unique_structs)

                fitnesses[:, -frac_in.shape[1]:] = lambda_pen*abs(ns - frac_in)

                fitnesses[:, -3] = lambda_pen*np.sum(np.abs(avg_ni), axis=1)
                fitnesses[:, -4] = lambda_pen*np.sum(np.abs(1 - ni_var), axis=1)


                for fit_id, (entry, weight) in enumerate(
                        zip(master_database.entries, weights)):

                    name = entry.struct_name
                    s_id = all_struct_names.index(name)

                    if entry.type == 'forces':

                        w_fcs = all_fcs[s_id]
                        true_fcs = entry.value

                        diff = w_fcs - true_fcs

                        # zero out interactions outside of range of O atom
                        epsilon = np.linalg.norm(diff, 'fro',
                                                 axis=(1, 2)) / np.sqrt(10)
                        fitnesses[:, fit_id] = epsilon * epsilon * weight

                    elif entry.type == 'energy':

                        r_name = entry.ref_struct
                        true_ediff = entry.value

                        # find index of structures to know which energies to use
                        r_id = all_struct_names.index(r_name)

                        comp_ediff = all_eng[s_id, :] - all_eng[r_id, :]

                        tmp = (comp_ediff - true_ediff) ** 2
                        fitnesses[:, fit_id] = tmp * weight

        if is_master:
            if not penalty:
                fitnesses = fitnesses[:, :-4]

        if return_ni:
            return fitnesses, max_ni, min_ni, avg_ni
        else:
            if output:
                if is_master:
                    fitnesses = np.sum(fitnesses, axis=1)

                    print(
                        np.min(fitnesses), np.max(fitnesses), np.average(fitnesses),
                        flush=True
                    )

            return fitnesses

    def grad_wrap(master_pop, weights):
        """Evalautes the gradient for all potentials in the population"""

        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
            pop = np.atleast_2d(pop)
        else:
            pop = None

        # eng = manager.compute_energy(pop)
        eng, _, _, _, _, _ = manager.compute_energy(pop)
        fcs = manager.compute_forces(pop)

        eng_grad = manager.compute_energy_grad(pop)
        fcs_grad = manager.compute_forces_grad(pop)

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

                gradient = np.zeros((
                    len(pop), potential_template.pvec_len,
                    len(master_database.entries)
                ))

                for fit_id, (entry, weight) in enumerate(
                        zip(master_database.entries, weights)):

                    name = entry.struct_name

                    if entry.type == 'forces':

                        s_id = all_struct_names.index(name)

                        w_fcs = all_fcs[s_id]
                        true_fcs = entry.value

                        diff = w_fcs - true_fcs

                        # zero out interactions outside of range of O atom
                        fcs_grad = mgr_fcs_grad[s_id]

                        scaled = np.einsum('pna,pnak->pnak', diff, fcs_grad)
                        summed = scaled.sum(axis=1).sum(axis=1)

                        gradient[:, :, fit_id] += (2 * summed / 10) * weight

                    elif entry.type == 'energy':
                        r_name = entry.ref_struct
                        true_ediff = entry.value

                        # find index of structures to know which energies to use
                        s_id = all_struct_names.index(name)
                        r_id = all_struct_names.index(r_name)

                        comp_ediff = all_eng[s_id, :] - all_eng[r_id, :]

                        eng_err = comp_ediff - true_ediff
                        s_grad = mgr_eng_grad[s_id]
                        r_grad = mgr_eng_grad[r_id]

                        gradient[:, :, fit_id] += \
                            (eng_err[:, np.newaxis]*(s_grad - r_grad)*2)*weight

                indices = np.where(potential_template.active_mask)[0]
                gradient = gradient[:, indices, :].swapaxes(1, 2)

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


def group_database_subsets(database, num_managers):
    """Groups workers based on evaluation time to help with load balancing.

    Returns:
        distributed_work (list): the partitioned work
        num_managers (int): desired number of managers to be used
    """
    # TODO: record scaling on first run, then redistribute to load balance
    # TODO: managers should just split Database; determine cost later

    # Number of managers should not exceed the number of structures
    num_managers = min(num_managers, len(database.structures))

    unassigned_structs = sorted(list(database.structures.keys()))

    work_weights, name_list = compute_relative_weights(database)

    work_per_proc = np.sum(work_weights)  # / num_managers

    # work_weights = work_weights.tolist()

    # Splitting code taken from SO: https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
    cum_arr = np.cumsum(work_weights) / np.sum(work_weights)

    idx = np.searchsorted(cum_arr,
                          np.linspace(0, 1, num_managers, endpoint=False)[1:])
    name_chunks = np.split(unassigned_structs, idx)
    weight_chunks = np.split(work_weights, idx)

    subsets = []
    grouped_work = []

    for name_chunk, work_chunk in zip(name_chunks, weight_chunks):
        cumulated_work = 0

        names = []

        # while unassigned_structs and (cumulated_work < work_per_proc):
        for n, w in zip(name_chunk, work_chunk):
            names.append(n)
            cumulated_work += w

        mini_database = Database.manual_init(
            {name: database.structures[name] for name in names},
            {name: database.true_energies[name] for name in names},
            {name: database.true_forces[name] for name in names},
            {name: database.weights[name] for name in names},
            database.reference_struct,
            database.reference_energy
        )

        subsets.append(mini_database)
        grouped_work.append(cumulated_work)

    return subsets, grouped_work


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

# def initialize_potential_template(load_path):
#     # TODO: BW settings
#     inner_cutoff = 1.5
#     outer_cutoff = 5.5
#
#     points_per_spline = 7
#
#     x_pvec = np.concatenate([
#         np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 5),
#         np.tile(np.linspace(-1, 1, points_per_spline), 2),
#         np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 2),
#         np.tile(np.linspace(-1, 1, points_per_spline), 3)]
#     )
#
#     x_indices = range(0, points_per_spline * 12, points_per_spline)
#     types = ["Ti", "Mo"]
#
#     potential_template = Template(
#         pvec_len=108,
#         u_ranges=[(-1, 1), (-1, 1)],
#         # Ranges taken from Lou Ti-Mo (phis) or from old TiO (other)
#         spline_ranges=[(-1, 1), (-1, 1), (-1, 1), (-5, 5), (-5, 5),
#                        (-1, 1), (-1, 1), (-2.5, 2.5), (-2.5, 2.5),
#                        (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)],
#         spline_indices=[(0, 9), (9, 18), (18, 27), (27, 36), (36, 45),
#                         (45, 54), (54, 63), (63, 72), (72, 81),
#                         (81, 90), (90, 99), (99, 108)]
#     )
#
#     mask = np.ones(potential_template.pvec_len)
#
#     potential_template.pvec[6] = 0
#     mask[6] = 0  # rhs value phi_Ti
#     potential_template.pvec[8] = 0
#     mask[8] = 0  # rhs deriv phi_Ti
#
#     potential_template.pvec[15] = 0
#     mask[15] = 0  # rhs value phi_TiMo
#     potential_template.pvec[17] = 0
#     mask[17] = 0  # rhs deriv phi_TiMo
#
#     potential_template.pvec[24] = 0
#     mask[24] = 0  # rhs value phi_Mo
#     potential_template.pvec[26] = 0
#     mask[26] = 0  # rhs deriv phi_Mo
#
#     potential_template.pvec[33] = 0
#     mask[33] = 0  # rhs value rho_Ti
#     potential_template.pvec[35] = 0
#     mask[35] = 0  # rhs deriv rho_Ti
#
#     potential_template.pvec[42] = 0
#     mask[42] = 0  # rhs value rho_Mo
#     potential_template.pvec[44] = 0
#     mask[44] = 0  # rhs deriv rho_Mo
#
#     potential_template.pvec[69] = 0
#     mask[69] = 0  # rhs value f_Ti
#     potential_template.pvec[71] = 0
#     mask[71] = 0  # rhs deriv f_Ti
#
#     potential_template.pvec[78] = 0
#     mask[78] = 0  # rhs value f_Mo
#     potential_template.pvec[80] = 0
#     mask[80] = 0  # rhs deriv f_Mo
#
#     potential_template.active_mask = mask
#
#     return potential_template


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


def rescale_ni(pots, min_ni, max_ni, potential_template):
    """
    Rescales the rho/f/g splines to try to fit it all ni into the U domain of
    [-1, 1].

    Note:
        order of min_ni and max_ni is assumed to be the same as pots

    Args:
        pots: (NxP array) the N starting potentials of P parameters
        min_ni: (Nx1 array) the minimum ni sampled for each of the N potential
        max_ni: (Nx1 array) the maximum ni sampled for each of the N potential
        potential_template: (Template) template object, used for indexing

    Returns:
        updated_pots: (NxP array) the potentials with rescaled rho/f/g splines
    """

    pots = np.array(pots)

    ni = np.hstack([max_ni, min_ni])
    indices = np.argmax(abs(ni), axis=1)
    scale = np.choose(indices, ni.T)
    signs = np.sign(scale)

    pots[:, potential_template.rho_indices] /= \
        scale[:, np.newaxis]

    pots[:, potential_template.f_indices] /= \
        abs(scale[:, np.newaxis]) ** (1. / 3)

    pots[:, potential_template.g_indices] /= \
        signs[:, np.newaxis]*(abs(scale[:, np.newaxis]) ** (1. / 3))

    return pots


def shift_u(min_ni, max_ni):
    """
    Computes the new U domains for all N potentials

    Args:
        min_ni: (Nx1 array) the minimum ni sampled for each of the N potential
        max_ni: (Nx1 array) the maximum ni sampled for each of the N potential

    Returns:
        new_u_domains: (num_atom_types length list) each element in the list
        is a length-2 tuple of the min/max values for the U spline of one
        atom type
    """

    scale = np.max(np.abs(np.hstack([min_ni, max_ni])), axis=1)
    scale = np.average(scale, axis=0)

    tmp_min_ni = np.average(min_ni, axis=0)
    tmp_max_ni = np.average(max_ni, axis=0)

    new_u_domains = [
        (tmp_min_ni[i] / scale, tmp_max_ni[i] / scale) for i in range(2)]

    for k, tup in enumerate(new_u_domains):
        tmp_tup = []

        size = abs(tup[1] - tup[0])

        for kk, lim in enumerate(tup):
            if kk == 0:  # lower bound
                # add some to the lower bound
                tmp_tup.append(lim + 0.1*size)
            elif kk == 1:
                # subtract some from the upper bound
                tmp_tup.append(lim - 0.1*size)

        new_u_domains[k] = tuple(tmp_tup)

    return new_u_domains


def mcmc(population, weights, cost_fxn, potential_template, T,
         parameters, active_tags, checkpoint_fxn, is_master, start_step=0,
         cooling_rate=1, T_min=0, suffix="", max_nsteps=None):
    """
    Runs an MCMC optimization on the given subset of the parameter vectors.
    Stopping criterion is either 20 steps without any improvement,
    or max_nsteps reached.

    Args:
        max_nsteps: (int) maximum number of steps to run
        population: (np.arr) 2d array where each row is a potential
        weights: (np.arr) weights for each term in cost function
        cost_fxn: (callable) cost funciton
        potential_template: (Template) template containing potential information
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
        max_nsteps = parameters['MCMC_BLOCK_SIZE']

    move_prob = parameters['SA_MOVE_PROB']
    move_scale = parameters['SA_MOVE_SCALE']
    checkpoint_freq = parameters['CHECKPOINT_FREQ']

    if is_master:
        population = np.array(population)

        active_indices = []

        for tag in active_tags:
            active_indices.append(
                np.where(potential_template.spline_tags == tag)[0]
            )

        active_indices = np.concatenate(active_indices)

        # new_mask = potential_template.active_mask.copy()
        # new_mask[:] = 0
        # new_mask[active_indices] = 1

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
            tmp, weights, return_ni=True, penalty=True
        )

        trial_cost, t_max_ni, t_min_ni, t_avg_ni = cost_fxn(
            tmp_trial, weights, return_ni=True, penalty=True
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
                    checkpoint_fxn(
                        tmp, current_cost, c_max_ni, c_min_ni, c_avg_ni,
                        start_step + step_num, parameters, potential_template
                    )

            # if current_best == prev_best:
            #     num_without_improvement += 1
            #
            #     if num_without_improvement == 20:
            #         break
            #
            # else:
            #     num_without_improvement = 0

    if is_master:
        population[:, active_indices] = current

    return population

def checkpoint(population, costs, max_ni, min_ni, avg_ni, i, parameters,
    potential_template):
    """Saves information to files for later use"""

    # save costs -- assume file is being appended to
    with open(parameters['COST_FILE_NAME'], 'ab') as f:
        np.savetxt(f, np.atleast_2d(costs))

    # save population
    digits = np.floor(np.log10(parameters['SA_NSTEPS']))

    format_str = os.path.join(
        parameters['SAVE_DIRECTORY'],
        'pop_{0:0' + str(int(digits) + 1)+ 'd}.dat'
    )

    np.savetxt(format_str.format(i), population)

    # output ni to file
    with open(parameters['NI_TRACE_FILE_NAME'], 'ab') as f:
        np.savetxt(
            f,
            np.concatenate(
                [
                    min_ni.ravel(),
                    max_ni.ravel(),
                    avg_ni.ravel(),
                    [potential_template.u_ranges[0][0],
                    potential_template.u_ranges[0][1],
                    potential_template.u_ranges[1][0],
                    potential_template.u_ranges[1][1]],
                ]
            )
        )

def build_M(self, num_x, dx, bc_type):
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
