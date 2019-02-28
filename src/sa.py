import os
import sys

import random
import numpy as np
from mpi4py import MPI
import partools
import datetime
from scipy.optimize import least_squares

from src.database import Database
from src.manager import Manager

################################################################################

def sa(parameters, template):
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

    if os.path.isdir(parameters['SAVE_DIRECTORY']):
        parameters['SAVE_DIRECTORY'] = parameters['SAVE_DIRECTORY'] + '-' + \
            str(np.random.randint(100000))

    if is_master:
        prepare_save_directory(parameters)

        potential_template = template

        master_database = Database(
            parameters['STRUCTURE_DIRECTORY'],
            parameters['INFO_DIRECTORY'],
            parameters['REFERENCE_STRUCT']
        )

        # Trace file to be appended to later
        f = open(parameters['NI_TRACE_FILE_NAME'], 'ab')
        f.close()

        master_database.load_structures(parameters['NUM_STRUCTS'])

        # master_database.print_metadata()

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


        weights = np.ones(len(master_database.entries))
        print("worker_ranks:", worker_ranks)
    else:
        potential_template = None
        master_database = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

        weights = None

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

    cost_fxn, grad_wrap = partools.build_evaluation_functions(
        potential_template, master_database, all_struct_names, manager,
        is_master, is_manager, manager_comm, "Ti48Mo80_type1_c18"
    )

    if is_master:
        chain = np.zeros(
            (parameters['SA_NSTEPS'] + 1, parameters['POP_SIZE'],
                np.where(potential_template.active_mask)[0].shape[0])
        )

        for i in range(parameters['POP_SIZE']):
            tmp = potential_template.generate_random_instance()
            chain[0, i, :] = tmp[np.where(potential_template.active_mask)[0]]

        current = chain[0]
    else:
        current = None
        chain = None
        trace = None

    current_cost = cost_fxn(current, weights)

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

        num_accepted = 0

    # run simulated annealing; stop cooling once T = Tmin
    step_num = 0
    while step_num < parameters['SA_NSTEPS']:
        # do U rescaling
        current_cost, max_ni, min_ni = cost_fxn(
            current, weights, return_ni=True
        )

        if is_master:
            # only plotting the range of the 1st potential
            current_cost = np.sum(current_cost, axis=1)

            tmp_min_ni = min_ni[np.argsort(current_cost)]
            tmp_max_ni = max_ni[np.argsort(current_cost)]

            # output to ile
            with open(parameters['NI_TRACE_FILE_NAME'], 'ab') as f:
                np.savetxt(
                    f,
                    np.atleast_2d(
                        [tmp_min_ni[0], tmp_max_ni[0]]
                        # np.concatenate(
                            # [[tmp_min_ni[i], tmp_max_ni[i]] for i in
                            # range(2)]
                            # )
                        )
                )


        if parameters['DO_RESCALE'] and \
            (step_num < parameters['RESCALE_STOP_STEP']) and \
                (step_num % parameters['RESCALE_FREQ'] == 0):
                    if is_master:
                        tmp_min_ni = min_ni[np.argsort(current_cost)]
                        tmp_max_ni = max_ni[np.argsort(current_cost)]

                        # new_u_domains = [(tmp_min_ni[i], tmp_max_ni[i]) for
                        #         i in range(2)]
                        new_u_domains = [(tmp_min_ni[0], tmp_max_ni[0])]

                        print(
                            "Rescaling ... new ranges =", new_u_domains,
                            flush=True
                        )
                    else:
                        new_u_domains = None

                    # if is_manager:
                    #     new_u_domains = manager_comm.bcast(new_u_domains, root=0)
                    #     potential_template.u_ranges = new_u_domains
                    # 
                    # current_cost = cost_fxn(current, weights)

        if parameters['DO_LMIN'] and (step_num % parameters['LMIN_FREQ'] == 0):
            current_cost = cost_fxn(current, weights)

            if is_master:
                print("Performing local minimization ...", flush=True)
                print("Before", np.sum(current_cost, axis=1))

                minimized = current.copy()
            else:
                minimized = None

            minimized = local_minimization(
                minimized , cost_fxn, grad_wrap, weights, world_comm, is_master,
                nsteps=parameters['LMIN_NSTEPS']
            )

            lm_cost = cost_fxn(minimized, weights)

            if is_master:
                current = minimized.copy()
                current_cost = np.sum(lm_cost, axis=1)
                print("After", current_cost)

        # propose a move
        if is_master:

            # choose a single knot from each potential
            # rnd_indices = np.random.randint(
            #     current.shape[1], size=current.shape[0]
            # )
            # 
            # trial_position = current.copy()
            # trial_position[:, rnd_indices] += np.random.normal(
            #     scale=0.01, size=current.shape[0]
            # )

            # choose a random collection of knots from each potential
            mask = np.random.choice(
                [True, False],
                size = (current.shape[0], current.shape[1]),
                p = [parameters['SA_MOVE_PROB'], 1 - parameters['SA_MOVE_PROB']]
            )

            trial_position = current.copy()
            trial_position[mask] = trial_position[mask] +\
                np.random.normal(scale=parameters['SA_MOVE_SCALE'])

        else:
            trial_position = None

        # compute the Metropolis-Hastings ratio
        trial_cost = cost_fxn(trial_position, weights)

        # temperature is used as a multiplicative factor on the MLE cost
        if is_master:
            trial_cost = np.sum(trial_cost, axis=1)

            tmp = current_cost

            T = np.max([T_min, T*parameters['COOLING_RATE']])

            ratio = np.exp((current_cost - trial_cost) / T)

            # automatically accept anythinig with a ratio >= 1
            where_auto_accept = np.where(ratio >= 1)[0]

            # conditionally accept everything else
            where_cond_accept = np.where(
                np.random.random(ratio.shape[0]) < ratio
            )[0]

            total_accepted = set(
                np.concatenate([where_auto_accept, where_cond_accept])
            )

            num_accepted += len(total_accepted)

            # accepted = False
            # if ratio > 1:
            #     accepted = True
            #     num_accepted += 1
            # else:
            #     if np.random.random() < ratio: # accept the move
            #         accepted = True
            #         num_accepted += 1


            # update accepted moves and costs
            current[where_auto_accept] = trial_position[where_auto_accept]
            current_cost[where_auto_accept] = trial_cost[where_auto_accept]

            current[where_cond_accept] = trial_position[where_cond_accept]
            current_cost[where_cond_accept] = trial_cost[where_cond_accept]

            # print statistics
            print(
                step_num + 1, "{:.3f}".format(T), np.min(current_cost),
                np.average(current_cost),
                num_accepted / (step_num + 1) / parameters['POP_SIZE'],
                flush=True
            )

            chain[step_num + 1] = current
            trace[step_num + 1] = current_cost

            checkpoint(current, step_num, parameters)

        step_num += 1
    return chain, trace

def prepare_save_directory(parameters):
    """Creates directories to store results"""

    print()
    if os.path.isdir(parameters['SAVE_DIRECTORY']):
        print()
        print("/" + "*" * 30 + " WARNING " + "*" * 30 + "/")
        print("A folder already exists for these settings.\nPress Enter"
              " to ovewrite old data, or Ctrl-C to quit")
        input("/" + "*" * 30 + " WARNING " + "*" * 30 + "/\n")
    print()

    os.mkdir(parameters['SAVE_DIRECTORY'])

def local_minimization(master_pop, fxn, grad, weights, world_comm, is_master, nsteps=20):
    pad = 100

    def lm_fxn_wrap(raveled_pop, original_shape):
        # print(raveled_pop)
        val = fxn(
            raveled_pop.reshape(original_shape), weights, output=False
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
            raveled_pop.reshape(original_shape), weights
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

    org_fits = fxn(master_pop, weights)
    new_fits = fxn(new_pop, weights)

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

def checkpoint(population, i, parameters):
    """Saves information to files for later use"""
    digits = np.floor(np.log10(parameters['SA_NSTEPS']))

    format_str = os.path.join(
        parameters['SAVE_DIRECTORY'],
        'pop_{0:0' + str(int(digits) + 1)+ 'd}.dat'
    )

    np.savetxt(format_str.format(i), population)

if __name__ == "__main__":
    sa(SA_NSTEPS, COOLING_RATE)
