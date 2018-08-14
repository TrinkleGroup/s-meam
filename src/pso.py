import os
import sys
sys.path.append('./')

import numpy as np
import random
np.set_printoptions(precision=8, linewidth=np.inf, suppress=True)
np.random.seed(42)
random.seed(42)

import glob
import pickle
import datetime
from mpi4py import MPI
from scipy.optimize import least_squares

################################################################################

# TODO: BW settings

# LOAD_PATH = "data/fitting_databases/leno-redo/"
LOAD_PATH = "/projects/sciteam/baot/leno-redo/"

DB_PATH = LOAD_PATH + 'structures/'
DB_INFO_FILE_NAME = LOAD_PATH + 'rhophi/info'

date_str = datetime.datetime.now().strftime("%Y-%m-%d")

LOG_FILENAME = "log.dat"
BEST_TRACE_FILENAME = "best_trace.dat"
FINAL_FILENAME = "final_pot.dat"

################################################################################

MASTER_RANK = 0

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
rank = comm.Get_rank()

is_master_node = (rank == MASTER_RANK)

SWARM_SIZE = mpi_size

COGNITIVE_WEIGHT = 0.005  # relative importance of individual best
SOCIAL_WEIGHT = 0.003     # relative importance of global best
MOMENTUM_WEIGHT = 0.002   # relative importance of particle momentum

MAX_NUM_PSO_STEPS = 500
NUM_LMIN_STEPS = 30

FITNESS_THRESH = 1

################################################################################

def main():
    if is_master_node:

        # print("MASTER: Preparing save directory/files ... ", flush=True)
        # prepare_save_directory()

        for fname in [LOG_FILENAME, BEST_TRACE_FILENAME, FINAL_FILENAME]:
            if os.path.isfile(fname): os.remove(fname)

        print("MASTER: Loading structures ...", flush=True)
        structures, weights = load_structures_on_master()
        print(structures.keys())

        print("MASTER: Loading energy/forces database ... ", flush=True)
        true_forces, true_energies = load_true_values(structures.keys())

        print("MASTER: Determining potential information ...", flush=True)
        ex_struct = structures[list(structures.keys())[0]]
        type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
        pvec_len = ex_struct.len_param_vec
        print('PVEC=', pvec_len)

        print("MASTER: Preparing to send structures to slaves ... ", flush=True)

        # initialize swarm
        swarm_positions = init_positions(SWARM_SIZE)
        swarm_velocities = init_velocities(SWARM_SIZE)
        swarm_bests = np.copy(swarm_positions)
        global_best_pos = swarm_positions.shape[1]
    else:
        spline_indices = None
        structures = None
        weights = None
        true_forces = None
        true_energies = None
        pvec_len = None

        swarm_positions = None
        swarm_velocities = None
        swarm_bests = None
        global_best_pos = None

    # Send all information necessary to building evaluation functions
    spline_indices = comm.bcast(spline_indices, root=0)
    pvec_len = comm.bcast(pvec_len, root=0)
    structures = comm.bcast(structures, root=0)
    weights = comm.bcast(weights, root=0)
    true_forces = comm.bcast(true_forces, root=0)
    true_energies = comm.bcast(true_energies, root=0)

    print("SLAVE: Rank", rank, "received", len(structures), 'structures',
            flush=True)

    eval_fxn, grad_fxn = build_evaluation_functions(structures, weights,
                                                    true_forces, true_energies,
                                                    spline_indices)

    # send individual particle informations to each process
    print("SLAVE: Rank", rank, "performing initial LMIN ...", flush=True)
    position = comm.scatter(swarm_positions, root=0)
    velocity = comm.scatter(swarm_velocities, root=0)

    personal_best_pos = comm.scatter(swarm_bests, root=0)
    global_best_pos = comm.bcast(global_best_pos, root=0)

    opt_best = least_squares(eval_fxn, position, grad_fxn, method='lm',
            max_nfev=NUM_LMIN_STEPS)

    position = opt_best['x']

    personal_best_pos = position

    fitness = np.sum(eval_fxn(position))
    personal_best_fitness = fitness

    print("SLAVE: Rank", rank, "minimized fitness:", fitness, flush=True)

    new_positions = comm.gather(position, root=0)
    all_fitnesses = comm.gather(fitness, root=0)

    # record global best
    if is_master_node:
        all_fitnesses = np.vstack(all_fitnesses)

        min_idx = np.argmin(all_fitnesses)

        global_best_pos = new_positions[min_idx]
        global_best_fit = all_fitnesses[min_idx]
    else:
        global_best_pos = None
        global_best_fit = None

    global_best_pos = comm.bcast(global_best_pos, root=0)
    global_best_fit = comm.bcast(global_best_fit, root=0)

    if is_master_node:
        global_best_pos_trace = []
        global_best_fit_trace = []

        log_f = open(LOG_FILENAME, 'wb')
        np.savetxt(log_f, [np.concatenate([[0], all_fitnesses.ravel()])])
        log_f.close()

        print("MASTER: step g_best", flush=True)

    i = 0
    while (i < MAX_NUM_PSO_STEPS) and (global_best_fit > FITNESS_THRESH):
        if is_master_node:
            print("{} {}".format(i+1, global_best_fit[0]), flush=True)

        # generate new velocities; update positions
        rp = np.random.random(position.shape[0])
        rg = np.random.random(position.shape[0])

        new_velocity = \
            MOMENTUM_WEIGHT * velocity + \
            COGNITIVE_WEIGHT * rp * (personal_best_pos - position) + \
            SOCIAL_WEIGHT * rg * (global_best_pos - position)

        position += new_velocity
        velocity = new_velocity

        fitness = np.sum(eval_fxn(position))

        # update personal bests
        if fitness < personal_best_fitness:
            personal_best_fitness = fitness
            personal_best_pos = position

        # update global bests
        new_positions = comm.gather(position, root=0)
        all_fitnesses = comm.gather(fitness, root=0)

        if is_master_node:
            all_fitnesses = np.vstack(all_fitnesses)

            min_idx = np.argmin(all_fitnesses)

            if all_fitnesses[min_idx] < global_best_fit:
                global_best_pos = new_positions[min_idx]
                global_best_fit = all_fitnesses[min_idx]

        global_best_pos = comm.bcast(global_best_pos, root=0)
        global_best_fit = comm.bcast(global_best_fit, root=0)

        if is_master_node:
            log_f = open(LOG_FILENAME, 'ab')
            np.savetxt(log_f, [np.concatenate([[0], all_fitnesses.ravel()])])
            log_f.close()

            f = open(BEST_TRACE_FILENAME, 'ab')
            np.savetxt(f, [np.concatenate([global_best_fit, global_best_pos.ravel()])])
            f.close()

        i += 1

    if is_master_node:
        opt_best = least_squares(eval_fxn, position, grad_fxn, method='lm',
                max_nfev=NUM_LMIN_STEPS)

        final_pot = opt_best['x']
        final_val = np.sum(eval_fxn(final_pot))

        f = open(FINAL_FILENAME, 'wb')
        np.savetxt(f, [final_val])
        np.savetxt(f, [final_pot.ravel()])
        f.close()

################################################################################

def prepare_save_directory():
    """Creates directories to store results"""

    print()
    print("Save location:", SAVE_DIRECTORY)
    if os.path.isdir(SAVE_DIRECTORY) and CHECK_BEFORE_OVERWRITE:
        print()
        print("/" + "*"*30 + " WARNING " + "*"*30 + "/")
        print("A folder already exists for these settings.\nPress Enter"
                " to ovewrite old data, or Ctrl-C to quit")
        input("/" + "*"*30 + " WARNING " + "*"*30 + "/\n")
    print()

def load_structures_on_master():
    """Builds Worker objects from the HDF5 database of stored values. Note that
    database weights are determined HERE.
    """

    structures = {}
    weights = {}

    i = 0
    for name in glob.glob(DB_PATH + '*'):

        short_name = os.path.split(name)[-1]
        short_name = os.path.splitext(short_name)[0]
        weights[short_name] = 1

        if weights[short_name] > 0:
            i += 1
            structures[short_name] = pickle.load(open(name, 'rb'))

    return structures, weights

def load_true_values(all_names):
    """Loads the 'true' values according to the database provided"""

    true_forces = {}
    true_energies = {}

    for name in all_names:

        fcs = np.genfromtxt(open(DB_INFO_FILE_NAME + '/info.' + name, 'rb'),
                skip_header=1)
        eng = np.genfromtxt(open(DB_INFO_FILE_NAME + '/info.' + name, 'rb'),
                max_rows=1)

        true_forces[name] = fcs
        true_energies[name] = eng

    return true_forces, true_energies

def find_spline_type_deliminating_indices(worker):
    """Finds the indices in the parameter vector that correspond to start/end
    (inclusive/exclusive respectively) for each spline group. For example,
    phi_range[0] is the index of the first know of the phi splines, while
    phi_range[1] is the next knot that is NOT part of the phi splines

    Args:
        worker (WorkerSpline): example worker that holds all spline objects
    """

    ntypes = worker.ntypes
    nphi = worker.nphi

    splines = worker.phis + worker.rhos + worker.us + worker.fs + worker.gs
    indices = [s.index for s in splines]

    phi_range = (indices[0], indices[nphi])
    rho_range = (indices[nphi], indices[nphi + ntypes])
    u_range = (indices[nphi + ntypes], indices[nphi + 2*ntypes])
    f_range = (indices[nphi + 2*ntypes], indices[nphi + 3*ntypes])
    g_range = (indices[nphi + 3*ntypes], -1)

    return [phi_range, rho_range, u_range, f_range, g_range], indices

def init_positions(N):
    # TODO: confirm that potentials are being spawned uniformly throghout

    positions = np.zeros((N, 83))

    scale_mag = 0.3

    ind = np.zeros(83)

    ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            (-0.2, 0.4)]

    indices = [(0,13), (15,20), (22,35), (37,48), (50,55), (57,61), (63,68)]

    for i in range(N):
        for rng,ind_tup in zip(ranges, indices):
            r_lo, r_hi = rng
            i_lo, i_hi = ind_tup

            rand = np.random.random(i_hi-i_lo)
            positions[i,i_lo:i_hi] = rand*(r_hi-r_lo) + r_lo

    return positions

def init_velocities(N):
    velocities = np.zeros((N, 83))

    scale_mag = 0.3

    ind = np.zeros(83)

    ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            (-0.2, 0.4)]

    indices = [(0,13), (15,20), (22,35), (37,48), (50,55), (57,61), (63,68)]

    for i in range(N):
        for rng,ind_tup in zip(ranges, indices):
            r_lo, r_hi = rng
            i_lo, i_hi = ind_tup

            diff = i_hi - i_lo

            # velocities should cover the range [-diff, diff]
            velocities[i,i_lo:i_hi] = np.random.random(i_hi-i_lo)*(2*diff) - diff

    return velocities


def build_evaluation_functions(structures, weights, true_forces, true_energies,
        spline_indices):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn(pot):
        pot = np.atleast_2d(pot)

        # full = np.hstack([pot, np.zeros((pot.shape[0], 108))])
        full = np.hstack([pot, np.zeros((pot.shape[0], 54))])

        fitness = np.zeros(2*len(structures))

        # Compute error for each worker on MPI node
        i = 0
        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(full) - true_forces[name]
            eng_err = w.compute_energy(full) - true_energies[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitness[i] += eng_err*eng_err*weights[name]
            fitness[i+1] += fcs_err*fcs_err*weights[name]

            i += 2

        # print(np.sum(fitness), flush=True)

        return fitness

    def grad(pot):
        """Only evaluates for a single potential at once"""

        pot = np.atleast_2d(pot)

        # full = np.hstack([pot, np.zeros((pot.shape[0], 108))])
        full = np.hstack([pot, np.zeros((pot.shape[0], 54))])

        grad_vec = np.zeros((2*len(structures), full.shape[1]))

        i = 0
        for name in structures.keys():
            w = structures[name]

            eng_err = w.compute_energy(full) - true_energies[name]
            fcs_err = (w.compute_forces(full) - true_forces[name])

            # Scale force errors

            # compute gradients
            eng_grad = w.energy_gradient_wrt_pvec(full)
            fcs_grad = w.forces_gradient_wrt_pvec(full)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            grad_vec[i] += (eng_err[:, np.newaxis]*eng_grad*2).ravel()
            grad_vec[i+1] += (2*summed / 10).ravel()

            i += 2

        # return grad_vec[:,:36]
        return grad_vec[:,:83]

    return fxn, grad


################################################################################

if __name__ == "__main__":
    main()
