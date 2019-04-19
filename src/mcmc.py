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
def mcmc(parameters, template):
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

        # set up logging
        prepare_save_directory(parameters)

        # prepare database and template
        potential_template = template

        master_database = Database(
            parameters['STRUCTURE_DIRECTORY'],
            parameters['INFO_DIRECTORY'],
            parameters['REFERENCE_STRUCT']
        )

        master_database.load_structures(parameters['NUM_STRUCTS'])

        # identify unique structures for creating managers
        all_struct_names = [s.encode('utf-8').strip().decode('utf-8') for s in
                master_database.unique_structs]

        struct_natoms = master_database.unique_natoms
        num_structs = len(all_struct_names)

        print(all_struct_names)

        old_copy_names = list(all_struct_names)

        worker_ranks = partools.compute_procs_per_subset(
            struct_natoms, world_size
        )

        weights = np.ones(len(master_database.entries))
        print("worker_ranks:", worker_ranks)
    else:
        potential_template = None
        master_database = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

        weights = None

    # send setup info to all workers
    weights = world_comm.bcast(weights, root=0)
    potential_template = world_comm.bcast(potential_template, root=0)
    num_structs = world_comm.bcast(num_structs, root=0)

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

    # set up comms between managers and their workers
    worker_group = world_group.Incl(worker_ranks)
    worker_comm = world_comm.Create(worker_group)

    struct_name = worker_comm.bcast(struct_name, root=0)
    manager_rank = worker_comm.bcast(manager_rank, root=0)

    manager = Manager(manager_rank, worker_comm, potential_template)

    # load structures on managers and their workers
    manager.struct_name = struct_name
    manager.struct = manager.load_structure(
        manager.struct_name, parameters['STRUCTURE_DIRECTORY'] + "/"
    )

    manager.struct = manager.broadcast_struct(manager.struct)

    # prepare evaluation functions
    cost_fxn, grad_wrap = partools.build_evaluation_functions(
        potential_template, master_database, all_struct_names, manager,
        is_master, is_manager, manager_comm, "Ti48Mo80_type1_c18"
    )

    # set up starting population
    if is_master:
        chain = np.zeros(
            (parameters['SA_NSTEPS'] + 1, parameters['POP_SIZE'],
                np.where(potential_template.active_mask)[0].shape[0])
        )

        for i in range(parameters['POP_SIZE']):
            tmp = potential_template.generate_random_instance()
            chain[0, i, :] = tmp[np.where(potential_template.active_mask)[0]]

        current = chain[0]

        ud = np.concatenate(potential_template.u_ranges)
        u_domains = np.tile(ud, (current.shape[0], 1))

    else:
        current = np.zeros(1)
        u_domains = np.zeros(1)
        chain = None
        trace = None

    current_cost, max_ni, min_ni, avg_ni = cost_fxn(
        np.hstack([current, u_domains]), weights,
        return_ni=True
    )

    # perform initial rescaling
    if is_master:
        current = partools.rescale_ni(
            current, min_ni, max_ni,
            potential_template
        )

    current_cost, max_ni, min_ni, avg_ni = cost_fxn(
        np.hstack([current, u_domains]), weights,
        return_ni=True
    )

    if is_master:
        current_cost = np.sum(current_cost, axis=1)

        trace = np.zeros((parameters['SA_NSTEPS'] + 1, parameters['POP_SIZE']))

        trace[0] = current_cost

        T = parameters['TSTART']
        T_min = parameters['TMIN']

        print("step T min_cost avg_cost avg_accepted")
        print(
            0, "{:.2}".format(T), np.min(current_cost),
            np.average(current_cost), "--", flush=True
        )

        tmp_min_ni = min_ni[np.argsort(current_cost)]
        tmp_max_ni = max_ni[np.argsort(current_cost)]
        tmp_avg_ni = avg_ni[np.argsort(current_cost)]

        partools.checkpoint(
            current, current_cost, tmp_max_ni, tmp_min_ni, tmp_avg_ni, -1,
            parameters, potential_template
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
                p=[parameters['SA_MOVE_PROB'], 1 - parameters['SA_MOVE_PROB']]
            )

            trial = inp.copy()
            trial[mask] = trial[mask] + np.random.normal(scale=move_scale)

            return trial

    num_without_improvement = 0

    # start run
    for step_num in range(parameters['SA_NSTEPS']):

        # check if shifting should be done
        if step_num % parameters['SHIFT_FREQ'] == 0:
            if is_master:
                print("Shifting U[] ...")
                u_domains = np.concatenate(
                        partools.shift_u(tmp_min_ni, tmp_max_ni)
                    )
                u_domains = np.tile(u_domains, (current.shape[0], 1))

        # check if rescaling should be done
        if step_num % parameters['RESCALE_FREQ'] == 0:
            if is_master:
                print("Rescaling rho/f/g ...")
                current = partools.rescale_ni(
                    current, min_ni, max_ni,
                    potential_template
                )

        if is_master:
            trial = move_proposal(current, parameters['SA_MOVE_SCALE'])
        else:
            trial = np.zeros(1)

        # compute costs of current population and proposed moves
        current_cost, c_max_ni, c_min_ni, c_avg_ni = cost_fxn(
            np.hstack([current, u_domains]),
            weights, return_ni=True, penalty=True
        )

        trial_cost, t_max_ni, t_min_ni, t_avg_ni = cost_fxn(
            np.hstack([trial, u_domains]),
            weights, return_ni=True, penalty=True
        )

        if is_master:
            # prev_best = current_best

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

            num_accepted += len(where_auto_accept) + len(where_cond_accept)

            current_best = current_cost[np.argmin(current_cost)]

            print(
                step_num, "{:.3f}".format(T),
                np.min(current_cost), 
                np.average(current_cost),
                num_accepted / (step_num + 1) / parameters['POP_SIZE'],
                np.average(u_domains, axis=0),
                flush=True
            )

            T = np.max([T_min, T*parameters['COOLING_RATE']])

            if is_master:
                if (step_num) % parameters['CHECKPOINT_FREQ'] == 0:

                    partools.checkpoint(
                        current, current_cost, c_max_ni, c_min_ni, c_avg_ni, -1,
                        parameters, potential_template
                    )

            # if current_best == prev_best:
            #     num_without_improvement += 1
            #
            #     if num_without_improvement == 20:
            #         break
            #
            # else:
            #     num_without_improvement = 0

    if parameters['DO_LMIN' ]:
        if is_master:
            print("Performing final local minimization ...", flush=True)
            print("Before", np.sum(current_cost, axis=1))

            minimized = current.copy()
        else:
            minimized = None

        minimized = local_minimization(
            minimized, u_domains, cost_fxn, grad_wrap, weights, world_comm, is_master,
            nsteps=parameters['LMIN_NSTEPS']
        )

        lm_cost = cost_fxn(minimized, weights)

        if is_master:
            current = minimized.copy()
            current_cost = np.sum(lm_cost, axis=1)
            print("After", current_cost)

            partools.checkpoint(
                current, tmp_max_ni, tmp_min_ni, tmp_avg_ni,
                parameters['SA_NSTEPS'], parameters, potential_template
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

if __name__ == "__main__":
    sa(SA_NSTEPS, COOLING_RATE)
