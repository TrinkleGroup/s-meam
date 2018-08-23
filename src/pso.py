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

COGNITIVE_WEIGHT = 0.5  # relative importance of individual best
SOCIAL_WEIGHT = 0.3     # relative importance of global best
MOMENTUM_WEIGHT = 0.2   # relative importance of particle momentum

MAX_NUM_PSO_STEPS = 50
NUM_LMIN_STEPS = 30

FITNESS_THRESH = 1

PARTICLES_PER_MPI_TASK = 1

################################################################################

# TODO: BW settings

LOAD_PATH = "/home/jvita/scripts/s-meam/project/data/fitting_databases/leno-redo/"
# LOAD_PATH = "/projects/sciteam/baot/leno-redo/"

# DB_PATH = './structures/'
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

SWARM_SIZE = mpi_size*PARTICLES_PER_MPI_TASK

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
        print('PVEC_SIZE =', pvec_len)

        print("MASTER: Preparing to send structures to slaves ... ", flush=True)

        # initialize swarm
        swarm_positions = np.array_split(init_positions(SWARM_SIZE), mpi_size)
        swarm_velocities = np.array_split(init_velocities(SWARM_SIZE), mpi_size)
    else:
        spline_indices = None
        structures = None
        weights = None
        true_forces = None
        true_energies = None

        swarm_positions = None
        swarm_velocities = None

    # Send all information necessary to building evaluation functions
    spline_indices = comm.bcast(spline_indices, root=0)
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
    positions = comm.scatter(swarm_positions, root=0)
    velocity = comm.scatter(swarm_velocities, root=0)

    fitnesses = np.zeros(positions.shape[0])
    personal_best_fitnesses = np.zeros(positions.shape[0])

    for p,particle in enumerate(positions):
        opt_best = least_squares(eval_fxn, particle, grad_fxn, method='lm',
                max_nfev=NUM_LMIN_STEPS)

        positions[p] = opt_best['x']

        fitnesses[p] = np.sum(eval_fxn(positions[p]))
        personal_best_fitnesses[p] = fitnesses[p]

    personal_best_positions = np.copy(positions)

    print("SLAVE: Rank", rank, "minimized fitnesses:", fitnesses, flush=True)

    new_positions = comm.gather(positions, root=0)
    all_fitnesses = comm.gather(fitnesses, root=0)

    # record global best
    if is_master_node:
        new_positions = np.vstack(new_positions)
        all_fitnesses = np.vstack(all_fitnesses)
        all_fitnesses = np.ravel(all_fitnesses)

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

        print("MASTER: step g_best max avg std", flush=True)

    i = 0
    while (i < MAX_NUM_PSO_STEPS) and (global_best_fit > FITNESS_THRESH):
        if is_master_node:
            print("STEP: {} {} {} {} {}".format(
                i+1, global_best_fit, np.max(all_fitnesses), np.average(all_fitnesses),
                np.std(all_fitnesses), flush=True))

            print(all_fitnesses)

        # generate new velocities; update positions
        for p,particle in enumerate(positions):
            rp = np.random.random(particle.shape)
            rg = np.random.random(particle.shape)
            # rp = random_velocity()
            # rg = random_velocity()

            new_velocity = \
                MOMENTUM_WEIGHT*velocity[p] + \
                COGNITIVE_WEIGHT*rp*(personal_best_positions[p]-particle) +\
                SOCIAL_WEIGHT * rg * (global_best_pos - particle)

            # print(new_velocity, flush=True)
            particle += new_velocity
            velocity = new_velocity

            particle = check_bounds(particle)

            fitnesses[p] = np.sum(eval_fxn(particle))

            # update personal bests
            if fitnesses[p] < personal_best_fitnesses[p]:
                personal_best_fitnesses[p] = fitnesses[p]
                personal_best_positions[p] = particle

        # update global bests
        new_positions = comm.gather(positions, root=0)
        all_fitnesses = comm.gather(fitnesses[p], root=0)

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
            np.savetxt(log_f, [np.concatenate([[i], all_fitnesses.ravel()])])
            log_f.close()

            f = open(BEST_TRACE_FILENAME, 'ab')
            np.savetxt(f, [np.concatenate([[global_best_fit],
                                           global_best_pos.ravel()])])
            f.close()

        i += 1

    if is_master_node:
        print("MASTER: performing final minimization ...", flush=True)
        opt_best = least_squares(eval_fxn, global_best_pos, grad_fxn,
                                 method='lm', max_nfev=NUM_LMIN_STEPS)

        final_pot = opt_best['x']
        final_val = np.sum(eval_fxn(final_pot))

        print("MASTER: final fitness =", final_val)

        f = open(FINAL_FILENAME, 'wb')
        np.savetxt(f, [final_val])
        np.savetxt(f, [final_pot.ravel()])
        f.close()

################################################################################

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

    ranges = [(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3), (-0.5, 1),
            (-0.5, 1)]

    indices = [(0,13), (15,20), (22,35), (37,48), (50,55), (57,61), (63,68)]

    for i in range(N):
        for rng,ind_tup in zip(ranges, indices):
            r_lo, r_hi = rng
            i_lo, i_hi = ind_tup

            rand = np.random.random(i_hi-i_lo)
            positions[i,i_lo:i_hi] = rand*(r_hi-r_lo) + r_lo

    return positions

def init_velocities(N):
#     velocities = np.zeros((N, 83))
# 
#     ranges = [(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3), (-0.5, 1),
#             (-0.5, 1)]
# 
#     indices = [(0,13), (15,20), (22,35), (37,48), (50,55), (57,61), (63,68)]
# 
#     for i in range(N):
#         for rng,ind_tup in zip(ranges, indices):
#             r_lo, r_hi = rng
#             i_lo, i_hi = ind_tup
# 
#             diff = r_hi - r_lo
# 
#             # velocities should cover the range [-diff, diff]
#             velocities[i,i_lo:i_hi] = np.random.random(i_hi-i_lo)*(2*diff) - diff

    velocities = np.random.random((N, 83)) - 0.5

    return velocities

def random_velocity():
    velocities = np.zeros((1, 83))

    ranges = [(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3), (-0.5, 1),
            (-0.5, 1)]

    indices = [(0,13), (15,20), (22,35), (37,48), (50,55), (57,61), (63,68)]

    for rng,ind_tup in zip(ranges, indices):
        r_lo, r_hi = rng
        i_lo, i_hi = ind_tup

        diff = r_hi - r_lo

        # velocities should cover the range [-diff, diff]
        velocities[:,i_lo:i_hi] = np.random.random(i_hi-i_lo)*(2*diff) - diff

    return velocities.ravel()

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


def check_bounds(positions):

    new_pos = positions.copy()

    ranges = [(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3), (-0.5, 1),
            (-0.5, 1)]

    indices = [(0,13), (15,20), (22,35), (37,48), (50,55), (57,61), (63,68)]

    for rng,ind_tup in zip(ranges, indices):
        r_lo, r_hi = rng
        i_lo, i_hi = ind_tup

        piece = positions[i_lo:i_hi]

        new_pos[i_lo:i_hi] = np.clip(positions[i_lo:i_hi],
                r_lo,r_hi)

    return new_pos

################################################################################

if __name__ == "__main__":
    main()
