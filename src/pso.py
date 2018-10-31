import os
import sys

sys.path.append('./')

import time
import random
import numpy as np

np.set_printoptions(precision=8, linewidth=np.inf, suppress=True)
np.random.seed(42)
random.seed(42)

import glob
import pickle
import datetime
from mpi4py import MPI
from scipy.optimize import least_squares

import src.meam
from src.meam import MEAM
from src.worker import Worker
from src.database import Database
from src.potential_templates import Template

################################################################################

COGNITIVE_WEIGHT = 0.75  # relative importance of individual best
SOCIAL_WEIGHT = 1     # relative importance of global best
MOMENTUM_WEIGHT = 0.5   # relative importance of particle momentum

MAX_NUM_PSO_STEPS = int(sys.argv[2])  # maximum number of PSO steps

DO_LOCAL_MIN = True
NUM_LMIN_STEPS = 30     # maximum number of LM steps
LMIN_FREQUENCY = 20

FITNESS_THRESH = 1

VELOCITY_SCALE = 0.01 # scale for starting velocity

# PARTICLES_PER_MPI_TASK = int(sys.argv[1])

################################################################################

# TODO: BW settings

# LOAD_PATH = "/home/jvita/scripts/s-meam/project/data/fitting_databases/leno-redo/"
LOAD_PATH = "/mnt/c/Users/jvita/scripts/s-meam/data/fitting_databases/fixU/"
#/ LOAD_PATH = "/projects/sciteam/baot/leno-redo/"

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

# SWARM_SIZE = mpi_size*PARTICLES_PER_MPI_TASK
SWARM_SIZE = int(sys.argv[1])

################################################################################

def main():
    if is_master_node:

        # print("MASTER: Preparing save directory/files ... ", flush=True)
        # prepare_save_directory()

        # load database of structures and true values
        database = Database(DB_PATH, DB_INFO_FILE_NAME, ['H', 'He'])
        # database.print_metadata()

        # build potential template
        potential = MEAM.from_file(LOAD_PATH + 'HHe.meam.spline')
        x_pvec, true_y_pvec, indices = src.meam.splines_to_pvec(potential.splines)

        # TODO: download updated Template from git
        # TODO: hard-coded for EAM
        true_y_pvec[83:] = 0

        x_pvec, seed_pvec, indices = src.meam.splines_to_pvec(
            potential.splines)

        mask = np.ones(seed_pvec.shape)

        seed_pvec[12] = 0; mask[12] = 0 # rhs phi_A knot
        seed_pvec[14] = 0; mask[14] = 0 # rhs phi_A deriv

        seed_pvec[27] = 0; mask[27] = 0 # rhs phi_B knot
        seed_pvec[29] = 0; mask[29] = 0 # rhs phi_B deriv

        seed_pvec[42] = 0; mask[42] = 0 # rhs phi_B knot
        seed_pvec[44] = 0; mask[44] = 0 # rhs phi_B deriv

        seed_pvec[55] = 0; mask[55] = 0 # rhs rho_A knot
        seed_pvec[57] = 0; mask[57] = 0 # rhs rho_A deriv

        seed_pvec[68] = 0; mask[68] = 0 # rhs rho_B knot
        seed_pvec[70] = 0; mask[70] = 0 # rhs rho_B deriv

        seed_pvec[92] = 0; mask[92] = 0 # rhs f_A knot
        seed_pvec[94] = 0; mask[94] = 0 # rhs f_A deriv

        seed_pvec[104] = 0; mask[104] = 0 # rhs f_B knot
        seed_pvec[106] = 0; mask[106] = 0 # rhs f_B deriv

        seed_pvec[83:] = 0; mask[83:] = 0 # EAM params only
        # seed_pvec[45:] = 0; mask[45:] = 0 # EAM params only

        potential_template = Template(
            seed=seed_pvec,
            active_mask=mask,
            spline_ranges=[(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3),
                           (-0.5, 1), (-0.5, 1), (-2, 3), (-2, 3), (-7, 2),
                           (-7, 2), (-7, 2)],
            spline_indices=[(0, 15), (15, 30), (30, 45), (45, 58), (58, 71),
                             (71, 77), (77, 83), (83, 95), (95, 107),
                             (107, 117), (117, 127), (127, 137)]
        )

        # potential_template.print_statistics()

        # initialize swarm
        swarm_positions = np.array_split(
            init_positions(SWARM_SIZE, potential_template),
            mpi_size
            )

        swarm_velocities = np.array_split(
            init_velocities(SWARM_SIZE, potential_template),
            mpi_size
            )
    else:
        potential_template = None
        database = None
        swarm_positions = None
        swarm_velocities = None

        # TODO: define fitness as minimized

    # Send all information necessary to building evaluation functions
    potential_template = comm.bcast(potential_template, root=0)
    database = comm.bcast(database, root=0)

    eval_fxn, grad_fxn = build_evaluation_functions(
        database,
        potential_template
    )

    # send individual particle information to each process
    positions = comm.scatter(swarm_positions, root=0)
    velocities = comm.scatter(swarm_velocities, root=0)

    fitnesses = np.zeros(positions.shape[0])
    personal_best_fitnesses = np.zeros(positions.shape[0])

    # print("SLAVE: Rank", rank, "performing initial LMIN ...", flush=True)
    for p,particle in enumerate(positions):
        if DO_LOCAL_MIN:
            opt_best = least_squares(eval_fxn, particle, grad_fxn, method='lm',
                    max_nfev=NUM_LMIN_STEPS)

            positions[p] = opt_best['x']

        fitnesses[p] = np.sum(eval_fxn(positions[p]))
        personal_best_fitnesses[p] = fitnesses[p]

    personal_best_positions = np.copy(positions)

    # print("SLAVE: Rank", rank, "minimized fitnesses:", fitnesses, flush=True)

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

        start = time.time()
        print("MASTER: step g_best max avg std avg_vel_mag", flush=True)

    i = 0
    while (i < MAX_NUM_PSO_STEPS) and (global_best_fit > FITNESS_THRESH):
        if is_master_node:

            avg_vel = np.average(np.linalg.norm(velocities, axis=1))

            print("STEP: {} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}".format(
                i,
                global_best_fit,
                np.max(all_fitnesses),
                np.average(all_fitnesses),
                np.std(all_fitnesses),
                avg_vel),
                flush=True
                )

        for p,(particle,velocity) in enumerate(zip(positions, velocities)):
            rp = np.random.random(particle.shape)
            rg = np.random.random(particle.shape)

            if (i % LMIN_FREQUENCY == 0) and DO_LOCAL_MIN:
                opt_best = least_squares(eval_fxn, particle, grad_fxn, method='lm',
                        max_nfev=NUM_LMIN_STEPS)

                positions[p] = opt_best['x']

                particle = positions[p]

            new_velocity = \
                MOMENTUM_WEIGHT*velocity + \
                COGNITIVE_WEIGHT*rp*(personal_best_positions[p] - particle) +\
                SOCIAL_WEIGHT * rg * (global_best_pos - particle)

            positions[p] = particle + new_velocity
            velocities[p] = new_velocity

            # particle = check_bounds(particle, potential_template)

            opt_best = least_squares(eval_fxn, particle, grad_fxn, method='lm',
                    max_nfev=10)

            fitnesses[p] = np.sum(eval_fxn(opt_best['x']))

            # update personal bests
            if fitnesses[p] < personal_best_fitnesses[p]:
                personal_best_fitnesses[p] = fitnesses[p]
                personal_best_positions[p] = particle

        # update global bests
        new_positions = comm.gather(positions, root=0)
        new_velocities = comm.gather(velocities, root=0)
        all_fitnesses = comm.gather(fitnesses[p], root=0)

        if is_master_node:
            new_positions = np.vstack(new_positions)
            new_velocities = np.vstack(new_velocities)
            all_fitnesses = np.vstack(all_fitnesses).ravel()

            min_idx = np.argmin(all_fitnesses)

            if all_fitnesses[min_idx] < global_best_fit:
                global_best_pos = new_positions[min_idx]
                global_best_fit = all_fitnesses[min_idx]

        global_best_pos = comm.bcast(global_best_pos, root=0)
        global_best_fit = comm.bcast(global_best_fit, root=0)

        # positions = np.array_split(new_positions, mpi_size)
        # velocities = np.array_split(new_velocities, mpi_size)
        #
        # positions = comm.scatter(positions, root=0)
        # velocities = comm.scatter(velocities, root=0)

        # if is_master_node:
        #     log_f = open(LOG_FILENAME, 'ab')
        #     np.savetxt(log_f, [np.concatenate([[i], all_fitnesses.ravel()])])
        #     log_f.close()
        #
        #     f = open(BEST_TRACE_FILENAME, 'ab')
        #     np.savetxt(f, [np.concatenate([[global_best_fit],
        #                                    global_best_pos.ravel()])])
        #     f.close()

        i += 1

    if is_master_node:
        runtime = time.time() - start
        print("MASTER: Average time per step:",
            runtime / MAX_NUM_PSO_STEPS, '(s)',
            flush=True
            )

        # print("MASTER: performing final minimization ...", flush=True)
        # opt_best = least_squares(eval_fxn, global_best_pos, grad_fxn,
        #                          method='lm', max_nfev=NUM_LMIN_STEPS)
        #
        # final_pot = opt_best['x']
        # final_val = np.sum(eval_fxn(final_pot))
        #
        # print("MASTER: final fitness =", final_val)
        #
        # f = open(FINAL_FILENAME, 'wb')
        # np.savetxt(f, [final_val])
        # np.savetxt(f, [final_pot.ravel()])
        # f.close()

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

def init_positions(N, template):
    swarm_positions = np.zeros((N, len(np.where(template.active_mask)[0])))

    for i in range(N):
        tmp = template.generate_random_instance()
        swarm_positions[i] = tmp[np.where(template.active_mask)[0]]

    return swarm_positions

def init_velocities(N, template):
    velocities = np.random.normal(
        scale=VELOCITY_SCALE,
        size=(N, len(np.where(template.active_mask)[0]))
        )

    for ind_tup,rng_tup in zip(template.spline_indices, template.spline_ranges):
        start, stop = ind_tup
        low, high = rng_tup

        velocities[:, start:stop] *= (high-low)

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


def build_evaluation_functions(database, potential_template):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn(pot):
        full = potential_template.insert_active_splines(pot)

        fitness = np.zeros(2 * len(database.structures))

    #     ref_energy = 0

        i = 0
        for name in database.structures.keys():

            w = database.structures[name]

            w_eng = w.compute_energy(full)
            w_fcs = w.compute_forces(full)

    #         if name == database.reference_struct:
    #             ref_energy = w_eng

            true_eng = database.true_energies[name]
            true_fcs = database.true_forces[name]

            eng_err = (w_eng - true_eng) ** 2
            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2

            fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

            fitness[i] = eng_err
            fitness[i+1] = fcs_err

            i += 2

    #     fitness[::2] -= ref_energy

        # print(np.sum(fitness), flush=True)

        return fitness

    def grad(pot):
        full = potential_template.insert_active_splines(pot)

        grad_vec = np.zeros((2 * len(database.structures), 137))

        i = 0
        for name in database.structures.keys():
            w = database.structures[name]

            w_eng = w.compute_energy(full)
            w_fcs = w.compute_forces(full)

            true_eng = database.true_energies[name]
            true_fcs = database.true_forces[name]

            eng_err = (w_eng - true_eng)
            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10))

            eng_grad = w.energy_gradient_wrt_pvec(full)
            fcs_grad = w.forces_gradient_wrt_pvec(full)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            grad_vec[i] += (eng_err[:, np.newaxis] * eng_grad * 2).ravel()
            grad_vec[i + 1] += (2 * summed / 10).ravel()

            i += 2

        tmp = grad_vec[:, np.where(potential_template.active_mask)[0]]
        return tmp

    # def fxn(pot):
    #     # "pot" should be a of potential Template objects
    #
    #     full = potential_template.insert_active_splines(pot)
    #     fitness = np.zeros(2 * len(database.structures))
    #
    #     all_worker_energies = []
    #     all_worker_forces = []
    #
    #     all_true_energies = []
    #     all_true_forces = []
    #
    #     ref_struct_idx = None
    #
    #     # Compute error for each worker on MPI node
    #     for j, name in enumerate(database.structures.keys()):
    #         w = database.structures[name]
    #
    #         # if name == database.reference_struct:
    #         #     ref_struct_idx = j
    #
    #         all_worker_energies.append(w.compute_energy(full))
    #         all_worker_forces.append(w.compute_forces(full))
    #
    #         all_true_energies.append(database.true_energies[name])
    #         all_true_forces.append(database.true_forces[name])
    #
    #     # subtract off reference energies
    #     # all_worker_energies = np.array(all_worker_energies)
    #     # all_worker_energies -= all_worker_energies[ref_struct_idx]
    #     #
    #     # all_true_energies = np.array(all_true_energies)
    #     # all_true_energies -= all_true_energies[ref_struct_idx]
    #
    #     i = 0
    #     for i in range(len(database.structures)):
    #         eng_err = all_worker_energies[i] - all_true_energies[i]
    #         fcs_err = all_worker_forces[i] - all_true_forces[i]
    #         fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)
    #
    #         fitness[i] += eng_err * eng_err
    #         fitness[i + 1] += fcs_err * fcs_err
    #
    #         i += 2
    #
    #     return fitness
    #
    # def grad(pot):
    #     full = potential_template.insert_active_splines(pot)
    #
    #     all_worker_energies = []
    #     all_worker_forces = []
    #
    #     all_true_energies = []
    #     all_true_forces = []
    #
    #     all_eng_grads = []
    #     all_fcs_grads = []
    #
    #     all_worker_energies = []
    #     all_worker_forces = []
    #
    #     all_true_energies = []
    #     all_true_forces = []
    #
    #     ref_struct_idx = None
    #
    #     grad_vec = np.zeros((2 * len(database.structures), 137))
    #
    #     # Compute error for each worker on MPI node
    #     for j, name in enumerate(database.structures.keys()):
    #         w = database.structures[name]
    #
    #         if name == database.reference_struct:
    #             ref_struct_idx = j
    #
    #         all_worker_energies.append(w.compute_energy(full))
    #         all_worker_forces.append(w.compute_forces(full))
    #
    #         all_true_energies.append(database.true_energies[name])
    #         all_true_forces.append(database.true_forces[name])
    #
    #     # subtract off reference energies
    #     # all_worker_energies = np.array(all_worker_energies)
    #     # all_worker_energies -= all_worker_energies[ref_struct_idx]
    #     #
    #     # all_true_energies = np.array(all_true_energies)
    #     # all_true_energies -= all_true_energies[ref_struct_idx]
    #
    #     for i, name in enumerate(database.structures.keys()):
    #         w = database.structures[name]
    #
    #         eng_err = (all_worker_energies[i] - all_true_energies[i]) ** 2
    #         fcs_err = ((all_worker_forces[i] - all_true_forces[i]) / np.sqrt(
    #             10)) ** 2
    #
    #         eng_grad = w.energy_gradient_wrt_pvec(full)
    #         fcs_grad = w.forces_gradient_wrt_pvec(full)
    #
    #         scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
    #         summed = scaled.sum(axis=1).sum(axis=1)
    #
    #         grad_vec[i] += (eng_err[:, np.newaxis] * eng_grad * 2).ravel()
    #         grad_vec[i + 1] += (2 * summed / 10).ravel()
    #
    #     tmp = grad_vec[:, np.where(potential_template.active_mask)[0]]
    #
    #     return tmp

    return fxn, grad

def check_bounds(positions, template):

    new_pos = positions.copy()
    reset = False

    for ind_tup,rng_tup in zip(template.spline_indices, template.spline_ranges):
        start, stop = ind_tup
        low, high = rng_tup

        piece = positions[start:stop]

        if not np.all(np.logical_and(piece < high, piece > low)):
            reset = True

    if reset:
        new_pos = template.generate_random_instance()
        new_pos = new_pos[np.where(template.active_mask)[0]]

    return new_pos

################################################################################

if __name__ == "__main__":
    main()
