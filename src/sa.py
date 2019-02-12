import os
import sys
sys.path.append('./')

import random
import numpy as np
from mpi4py import MPI
import partools
import datetime
from scipy.optimize import least_squares

from src.database import Database
from src.manager import Manager

################################################################################

NUM_STRUCTS = 30
POP_SIZE = int(sys.argv[1])
SA_STEPS = int(sys.argv[2])
COOLING_RATE = float(sys.argv[3])

################################################################################

DO_LMIN = True
LMIN_FREQ = 100
LMIN_STEPS = 15

################################################################################

BASE_PATH = "/home/jvita/scripts/s-meam/"
BASE_PATH = ""

LOAD_PATH = BASE_PATH + "data/fitting_databases/hyojung/"
LOAD_PATH = "/u/sciteam/vita/hyojung/"
SAVE_PATH = BASE_PATH + "data/results/"

date_str = datetime.datetime.now().strftime("%Y-%m-%d")

SAVE_DIRECTORY = SAVE_PATH + date_str + "-" + "meam" + "{}-{}".format(SA_STEPS,
                                                             COOLING_RATE)
if os.path.isdir(SAVE_DIRECTORY):
    SAVE_DIRECTORY = SAVE_DIRECTORY + '-' + str(np.random.randint(100000))

DB_PATH = LOAD_PATH + 'structures'
DB_INFO_FILE_NAME = LOAD_PATH + 'info_ref'

def simulated_annealing(nsteps, cooling_rate, cooling_type='linear'):
    """
    Runs a simulated annealing run. The Metropolis-Hastings acception/rejection
    criterion sampling from a normally-distributed P-dimensional vector for move
    proposals (where P is the number of parameters in the parameter vector)

    Note: cost_fxn should be parallelized, which is why you the processors need
    to know if they are the master

    Args:
        cost_fxn (callable): function for evaluating the costs
        nsteps (int): the number of monte carlo steps to take
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
        print("MASTER: Preparing save directory/files ... ", flush=True)
        prepare_save_directory()

        # Prepare database and potential template
        potential_template = partools.initialize_potential_template(LOAD_PATH)

        print(DB_PATH, DB_INFO_FILE_NAME)

        master_database = Database(
            DB_PATH, DB_INFO_FILE_NAME, "Ti48Mo80_type1_c18"
        )

        master_database.load_structures(NUM_STRUCTS)

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

        print("worker_ranks:", worker_ranks)
    else:
        potential_template = None
        master_database = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

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
        manager.struct_name, DB_PATH + "/"
    )

    manager.struct = manager.broadcast_struct(manager.struct)

    cost_fxn, grad_wrap = partools.build_evaluation_functions(
        potential_template, master_database, all_struct_names, manager,
        is_master, is_manager, manager_comm, "Ti48Mo80_type1_c18"
    )

    weights = np.ones(num_structs)

    if is_master:
        chain = np.zeros(
            (nsteps + 1, POP_SIZE, np.where(potential_template.active_mask)[0].shape[0])
        )

        for i in range(POP_SIZE):
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

        trace = np.zeros((nsteps + 1, POP_SIZE))

        trace[0] = current_cost

        T = 10.
        T_min = 0.01

        print("step T min_cost avg_cost avg_accepted")
        print(
            0, "{:.2}".format(T), np.min(current_cost),
            np.average(current_cost), "--", flush=True
        )

        num_accepted = 0

    # if cooling_type == 'linear':
    #     # T_schedule = np.linspace(1, 10, (2 - 0.02) / alpha)[::-1]
    #     T_min = 1
    # else:
    #     raise ValueError(
    #         "Invalid cooling schedule type; only 'linear' cooling is currently
    #         supported"
    #     )


    # run while cooling the system; stop cooling once minimum T reached
    step_num = 0
    while step_num < nsteps:

        if DO_LMIN and (step_num % LMIN_FREQ == 0):
            current_cost = cost_fxn(current, weights)

            if is_master:
                print("Performing local minimization ...", flush=True)
                print("Before", np.sum(current_cost, axis=1))

                minimized = current.copy()
            else:
                minimized = None

            minimized = local_minimization(
                minimized , cost_fxn, grad_wrap, weights, world_comm, is_master,
                nsteps=LMIN_STEPS
            )

            lm_cost = cost_fxn(minimized, weights)

            if is_master:
                current = minimized.copy()
                current_cost = np.sum(lm_cost, axis=1)
                print("After", current_cost)

        # propose a move
        if is_master:

            rnd_indices = np.random.randint(
                current.shape[1], size=current.shape[0]
            )

            trial_position = current.copy()
            trial_position[:, rnd_indices] += np.random.normal(
                scale=0.01, size=current.shape[0]
            )
        else:
            trial_position = None

        # compute the Metropolis-Hastings ratio
        trial_cost = cost_fxn(trial_position, weights)

        # temperature is used as a multiplicative factor on the MLE cost
        if is_master:
            trial_cost = np.sum(trial_cost, axis=1)

            tmp = current_cost

            T = np.max([T_min, T - cooling_rate])

            ratio = np.exp((current_cost - trial_cost) / T)

            # automatically accept anythinig with a ratio >= 1
            where_auto_accept = np.where(ratio >= 1)[0]
            num_accepted += len(where_auto_accept)

            # conditionally accept everything else
            where_cond_accept = np.where(
                np.random.random(ratio.shape[0]) < ratio
            )[0]
            num_accepted += len(where_cond_accept)

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
                step_num + 1, "{:.2f}".format(T), np.min(current_cost),
                np.average(current_cost),
                num_accepted / (step_num + 1) / POP_SIZE,
                flush=True
            )

            chain[step_num + 1] = current
            trace[step_num + 1] = current_cost

            checkpoint(current, step_num)

        step_num += 1


    return chain, trace

def prepare_save_directory():
    """Creates directories to store results"""

    print()
    print("Save location:", SAVE_DIRECTORY)
    if os.path.isdir(SAVE_DIRECTORY) and CHECK_BEFORE_OVERWRITE:
        print()
        print("/" + "*" * 30 + " WARNING " + "*" * 30 + "/")
        print("A folder already exists for these settings.\nPress Enter"
              " to ovewrite old data, or Ctrl-C to quit")
        input("/" + "*" * 30 + " WARNING " + "*" * 30 + "/\n")
    print()

    # os.rmdir(SAVE_DIRECTORY)
    os.mkdir(SAVE_DIRECTORY)

def local_minimization(master_pop, fxn, grad, weights, world_comm, is_master, nsteps=20):
    pad = 100

    def lm_fxn_wrap(raveled_pop, original_shape):
        val = fxn(raveled_pop.reshape(original_shape), weights, output=False)

        val = world_comm.bcast(val, root=0)

        # pad with zeros since num structs is less than num knots
        tmp = np.concatenate([val.ravel(), np.zeros(pad*original_shape[0])])
        return tmp

    def lm_grad_wrap(raveled_pop, original_shape):
        # shape: (num_pots, num_structs*2, num_params)

        grads = grad(raveled_pop.reshape(original_shape), weights)

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

        master_pop = updated_master_pop

    return np.array(master_pop)

def checkpoint(population, i):
    """Saves information to files for later use"""
    digits = np.floor(np.log10(SA_STEPS))

    format_str = os.path.join(
        SAVE_DIRECTORY,
        'pop_{0:0' + str(int(digits) + 1)+ 'd}.dat'
    )

    np.savetxt(format_str.format(i), population)

if __name__ == "__main__":
    simulated_annealing(SA_STEPS, COOLING_RATE)
