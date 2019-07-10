import os
import sys
import shutil
import random
import numpy as np
from mpi4py import MPI
import partools
import datetime
from scipy.optimize import least_squares
from collections import namedtuple

from src.database import Database
from src.manager import Manager

np.set_printoptions(precision=3)

################################################################################
def mcmc(parameters, database, template, is_manager, manager, manager_comm):
    """
    Runs a simulated annealing run. The Metropolis-Hastings acception/rejection
    criterion sampling from a normally-distributed P-dimensional vector for move
    proposals (where P is the number of parameters in the parameter vector)

    Note: cost_fxn should be parallelized, which is why you the processors need
    to know if they are the master

    Args:
        cost_fxn (callable): function for evaluating the costs
        is_master (bool): True if processor's world rank is 0
        cooling_rate (float): T = T0 - a*t (linear) or T = T0*a^t (exponential)

    Returns:
        chain (np.arr): the chain of 'nsteps' number of parameter vectors
        trace (np.arr): costs of each vector in the chain

    "cost" = "fitness"
    "likelihood", L() = np.exp(-cost / W) -- W = cost of MLE
    acceptance ratio = L(new) / L(old) -- see Givens/Hoeting section 7.1.1
    """
    # Record MPI settings
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
        # identify unique structures for creating managers
        all_struct_names = [s.encode('utf-8').strip().decode('utf-8') for s in
                database.unique_structs]

        struct_natoms = database.unique_natoms

        if is_master:
            original_mask = template.active_mask.copy()
    else:
        database = None
        all_struct_names = None

    # send setup info to all workers
    template = world_comm.bcast(template, root=0)

    # prepare evaluation functions
    cost_fxn, grad_wrap = partools.build_evaluation_functions(
        template, database, all_struct_names, manager,
        is_master, is_manager, manager_comm, "Ti48Mo80_type1_c18"
    )

    # set up starting population
    if is_master:
        chain = np.zeros(
            (parameters['NSTEPS']+1, parameters['POP_SIZE'], template.pvec_len)
        )

        for i in range(parameters['POP_SIZE']):
            chain[0, i, :] = template.generate_random_instance() 

        current = chain[0]

        weights = np.ones(len(database.entries))
    else:
        current = np.zeros(1)
        chain = None
        weights = None
        trace = None

    weights = world_comm.bcast(weights, root=0)

    current_cost, max_ni, min_ni, avg_ni = cost_fxn(
        current, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    # perform initial rescaling
    if is_master:
        current = partools.rescale_ni(current, min_ni, max_ni, template)

    current_cost, max_ni, min_ni, avg_ni = cost_fxn(
        current, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        new_costs = np.sum(current_cost, axis=1)

        trace = np.zeros((parameters['NSTEPS'] + 1, parameters['POP_SIZE']))

        trace[0] = new_costs

        T = 10.
        # T_min = parameters['TMIN']

        print("step T min_cost avg_cost avg_accepted")
        print(
            0, "{:.2}".format(T), np.min(new_costs),
            np.average(new_costs), "--", flush=True
        )

        partools.checkpoint(
            current, new_costs, max_ni,
            min_ni, avg_ni, 0, parameters, template,
            parameters['NSTEPS']
        )

        num_accepted = 0

    # define the function for proposing moves
    if parameters['PROP_TYPE'] == 'single':
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
                p=[parameters['MOVE_PROB'], 1 - parameters['MOVE_PROB']]
            )

            trial = inp.copy()
            trial[mask] = trial[mask] + np.random.normal(scale=move_scale)

            return trial

    num_without_improvement = 0

    lmin_time = parameters['LMIN_FREQ']
    shift_time = parameters['SHIFT_FREQ']
    resc_time = parameters['RESCALE_FREQ']
    toggle_time = parameters['TOGGLE_FREQ']

    u_only_status = 'off'

    # start run
    for step_num in range(parameters['NSTEPS']):

        # optionally run local minimization on best individual
        if parameters['DO_LMIN']:
            if lmin_time == 0:
                if is_master:
                    print(
                        "Performing local minimization on top 10 potentials...",
                        flush=True
                    )

                    subset = current[:10, np.where(template.active_mask)[0]]
                else:
                    subset = None

                subset = local_minimization(
                    subset, current, template, cost_fxn, grad_wrap, weights,
                    world_comm, is_master, nsteps=parameters['LMIN_NSTEPS']
                )

                if is_master:
                    current[:10, np.where(template.active_mask)[0]] = subset

                current_cost, max_ni, min_ni, avg_ni = cost_fxn(
                    current, weights, return_ni=True,
                    penalty=parameters['PENALTY_ON']
                )

                lmin_time = parameters['LMIN_FREQ'] - 1
            else:
                if u_only_status == 'off':  # don't decrement counter if U-only
                    lmin_time -= 1

        # optionally rescale potentials to put ni into [-1, 1]
        if parameters['DO_RESCALE']:
            if step_num < parameters['RESCALE_STOP_STEP']:
                if resc_time == 0:
                    if is_master:
                        print("Rescaling ...")

                        new_costs = np.sum(current_cost, axis=1)

                        current = partools.rescale_ni(
                            current, min_ni, max_ni, template
                        )

                    # re-compute the ni data for use with shifting U domains
                    current_cost, max_ni, min_ni, avg_ni = cost_fxn(
                        current, weights, return_ni=True,
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
                    new_costs = np.sum(current_cost, axis=1)

                    tmp_min_ni = min_ni[np.argsort(new_costs)]
                    tmp_max_ni = max_ni[np.argsort(new_costs)]

                    new_u_domains = partools.shift_u(tmp_min_ni, tmp_max_ni)
                    print("New U domains:", new_u_domains)
                else:
                    new_u_domains = None

                if is_manager:
                    new_u_domains = manager_comm.bcast(new_u_domains, root=0)
                    template.u_ranges = new_u_domains

                    manager.pot_template.u_ranges = template.u_ranges

                shift_time = parameters['SHIFT_FREQ'] - 1
            else:
                if u_only_status == 'off':  # don't decrement counter if U-only
                    shift_time -= 1

        # TODO: errors will occur if you try to resc/shift with U-only on
        if parameters['DO_TOGGLE'] and (toggle_time == 0):
            # optionally toggle splines on/off to allow U-only optimization

            if u_only_status == 'off':
                u_only_status = 'on'
                toggle_time = parameters['TOGGLE_DURATION'] - 1
            else:
                u_only_status = 'off'
                toggle_time = parameters['TOGGLE_FREQ'] - 1

            if is_master:
                print("Toggling U-only mode to:", u_only_status)

                # TODO: shouldn't hard-code [5,6]; use nphi to find tags
                current, template = toggle_u_only_optimization(
                    u_only_status, current, template, [5, 6], original_mask
                )

                new_mask = template.active_mask
            else:
                new_mask = None

            if is_manager:
                new_mask = manager_comm.bcast(new_mask, root=0)

                template.active_mask = new_mask
                manager.pot_template.active_mask = template.active_mask

        else:
            toggle_time -= 1

        # propose a move
        if is_master:
            current_active = current[:, np.where(template.active_mask)[0]]

            trial_move = move_proposal(current_active, parameters['MOVE_SCALE'])
            trial = current.copy()
            print(type(trial_move))
            print(type(trial))
            print(type(np.where(template.active_mask)[0]))
            print(trial_move.dtype)
            print(trial.dtype)
            print(np.where(template.active_mask)[0].dtype)
            # print(trial.shape)
            # print(np.where(template.active_mask)[0])
            # print(type(trial))
            # print(type(np.where(template.active_mask)[0]))
            # print('boop:', trial[:, [0, 1, 3, 6, 8]], flush=True)
            trial[: np.where(template.active_mask)[0]] = trial_move
            # trial[: np.array([0, 1, 2, 3,4])] = trial_move
        else:
            trial = None

        # compute costs of current population and proposed moves
        current_cost, c_max_ni, c_min_ni, c_avg_ni = cost_fxn(
            current, weights, return_ni=True, penalty=parameters['PENALTY_ON']
        )

        trial_cost, t_max_ni, t_min_ni, t_avg_ni = cost_fxn(
            trial, weights, return_ni=True, penalty=parameters['PENALTY_ON']
        )

        if is_master:
            new_cost_c = np.sum(current_cost, axis=1)
            new_cost_t = np.sum(trial_cost, axis=1)

            ratio = np.exp((new_cost_c - new_cost_t) / T)
            where_auto_accept = np.where(ratio >= 1)[0]

            where_cond_accept = np.where(
                np.random.random(ratio.shape[0]) < ratio
            )[0]

            # update population
            current[where_auto_accept] = trial[where_auto_accept]
            current[where_cond_accept] = trial[where_cond_accept]

            # update costs
            new_cost_c[where_auto_accept] = new_cost_t[where_auto_accept]
            new_cost_c[where_cond_accept] = new_cost_t[where_cond_accept]

            # update ni
            c_max_ni[where_auto_accept] = t_max_ni[where_auto_accept]
            c_max_ni[where_cond_accept] = t_max_ni[where_cond_accept]

            c_min_ni[where_auto_accept] = t_min_ni[where_auto_accept]
            c_min_ni[where_cond_accept] = t_min_ni[where_cond_accept]

            c_avg_ni[where_auto_accept] = t_avg_ni[where_auto_accept]
            c_avg_ni[where_cond_accept] = t_avg_ni[where_cond_accept]

            num_accepted += len(where_auto_accept) + len(where_cond_accept)

            current_best = new_cost_c[np.argmin(new_cost_c)]

            print(
                step_num, "{:.3f}".format(T),
                np.min(new_cost_c), 
                np.average(new_cost_c),
                num_accepted / (step_num + 1) / parameters['POP_SIZE'],
                np.average(u_domains, axis=0),
                flush=True
            )

            # T = np.max([T_min, T*parameters['COOLING_RATE']])

            if is_master:
                if (step_num) % parameters['CHECKPOINT_FREQ'] == 0:

                    partools.checkpoint(
                        current, new_cost_c, c_max_ni, c_min_ni, c_avg_ni,
                        step_num, parameters, template, parameters['NSTEPS']
                    )
    # end MCMC loop

    if parameters['DO_LMIN' ]:
        if is_master:
            print("Performing final local minimization ...", flush=True)
            print("Before:", np.sum(current_cost, axis=1)[:10])

            subset = current[:10, np.where(template.active_mask)[0]]
        else:
            subset = None

        subset = local_minimization(
            subset, current, template, cost_fxn, grad_wrap, weights,
            world_comm, is_master, nsteps=parameters['LMIN_NSTEPS']
        )

        if is_master:
            current[:10, np.where(template.active_mask)[0]] = subset

        lm_cost, max_ni, min_ni, avg_ni = cost_fxn(
            current, weights, return_ni=True,
            penalty=parameters['PENALTY_ON']
        )

        lm_cost = cost_fxn(minimized, weights)

        if is_master:
            new_cost = np.sum(lm_cost, axis=1)
            print("After:", new_cost[:10])

            partools.checkpoint(
                current, new_cost, max_ni, min_ni, avg_ni,
                step_num, parameters, template, parameters['NSTEPS']
            )

    return chain, trace

def prepare_save_directory(parameters):
    """Creates directories to store results"""

    if os.path.isdir(parameters['SAVE_DIRECTORY']):
        shutil.rmtree(parameters['SAVE_DIRECTORY'])

    os.mkdir(parameters['SAVE_DIRECTORY'])

def local_minimization(current, u_domains, fxn, grad, weights, world_comm, is_master, nsteps=20):
    pad = 100

    def lm_fxn_wrap(raveled_pop, original_shape):
        # print(raveled_pop)
        val = fxn(
            np.hstack([raveled_pop.reshape(original_shape), u_domains]),
            weights, output=False
        )

        val = world_comm.bcast(val, root=0)

        # pad with zeros since num structs is less than num knots
        tmp = np.concatenate([val.ravel(), np.zeros(pad*original_shape[0])])

        return tmp

    def lm_grad_wrap(raveled_pop, original_shape):
        # shape: (num_pots, num_structs*2, num_params)

        # if is_master:
        #     print('org_pop', raveled_pop.reshape(original_shape))

        grads = grad(
            np.hstack([raveled_pop.reshape(original_shape), u_domains]), weights
        )

        # if is_master:
        #     print("grads.shape\n", grads.shape)

        grads = world_comm.bcast(grads, root=0)

        # if is_master:
        #     print('grad', grads[0].astype(int))

        num_pots, num_structs_2, num_params = grads.shape

        padded_grad = np.zeros(
            (num_pots, num_structs_2, num_pots, num_params)
        )

        for pot_id, g in enumerate(grads):
            padded_grad[pot_id, :, pot_id, :] = g

        padded_grad = padded_grad.reshape(
            (num_pots * num_structs_2, num_pots * num_params)
        )

        # if is_master:
        #     print('padded_grad', padded_grad.shape, padded_grad.astype(int))


        # also pad with zeros since num structs is less than num knots

        tmp = np.vstack([
            padded_grad,
            np.zeros((pad*num_pots, num_pots * num_params))]
        )

        return tmp

    # lm_grad_wrap = '2-point'

    current = world_comm.bcast(current, root=0)
    current = np.array(current)

    opt_results = least_squares(
        lm_fxn_wrap, current.ravel(), lm_grad_wrap,
        method='lm', max_nfev=nsteps, args=(current.shape,)
    )

    if is_master:
        new_pop = opt_results['x'].reshape(current.shape)
    else:
        new_pop = None

    org_fits = fxn(np.hstack([current, u_domains]), weights)
    new_fits = fxn(np.hstack([new_pop, u_domains]), weights)

    if is_master:
        updated_current = list(current)

        for i, ind in enumerate(new_pop):
            if np.sum(new_fits[i]) < np.sum(org_fits[i]):
                updated_current[i] = new_pop[i]
            else:
                updated_current[i] = updated_current[i]

        current = np.array(updated_current)

    current = world_comm.bcast(current, root=0)

    return current

def toggle_u_only_optimization(toggle_type, master_pop, template,
                               spline_tags, original_mask):
    """

    Args:
        toggle_type: (str)
            'on' or 'off'

        master_pop: (np.arr)
            array of full potentials (un-masked)

        template: (Template)
            potential template object

        spline_tags: (list[int])
            which splines to toggle on/off

        original_mask: (np.arr)
            the original mask used before U-only was toggled on

    Returns:

        master_pop: (np.arr)
            the updated master population

    """

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

    return master_pop, template

if __name__ == "__main__":
    mcmc(SA_NSTEPS, COOLING_RATE)
