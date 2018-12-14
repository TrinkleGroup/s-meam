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
import src.partools as partools
from src.meam import MEAM
from src.worker import Worker
from src.database import Database
from src.potential_templates import Template
from src.manager import Manager

################################################################################

COGNITIVE_WEIGHT = 0.75  # relative importance of individual best
SOCIAL_WEIGHT = 1     # relative importance of global best
MOMENTUM_WEIGHT = 0.5   # relative importance of particle momentum

SWARM_SIZE = int(sys.argv[1])  # number of particles to create
MAX_NUM_PSO_STEPS = int(sys.argv[2])  # maximum number of PSO steps

DO_LOCAL_MIN = False
FLATTEN_LANDSCAPE = False
NMIN = 1 # how many particles to do LM on during local_minimization() calls
NUM_LMIN_STEPS = 30     # maximum number of LM steps
LMIN_FREQUENCY = 20

FITNESS_THRESH = 1

VELOCITY_SCALE = 0.01 # scale for starting velocity

# PARTICLES_PER_MPI_TASK = int(sys.argv[1])

################################################################################

# TODO: BW settings
BASE_PATH = ""
BASE_PATH = "/home/jvita/scripts/s-meam/"

LOAD_PATH = "/projects/sciteam/baot/pz-unfx-cln/"
LOAD_PATH = BASE_PATH + "data/fitting_databases/pinchao/"

DB_PATH = LOAD_PATH + 'structures'
DB_INFO_FILE_NAME = LOAD_PATH + 'full/info'

date_str = datetime.datetime.now().strftime("%Y-%m-%d")

LOG_FILENAME = "log.dat"
BEST_TRACE_FILENAME = "best_trace.dat"
FINAL_FILENAME = "final_pot.dat"

################################################################################

def main():
    # Record MPI settings
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:

        # Prepare directories and files
        # print_settings()

        # print("MASTER: Preparing save directory/files ... ", flush=True)
        # prepare_save_directory()

        # Trace file to be appended to later
        f = open(BEST_TRACE_FILENAME, 'ab')
        f.close()

        # Prepare database and potential template
        potential_template = initialize_potential_template()
        potential_template.print_statistics()
        print()

        struct_files = glob.glob(DB_PATH + "/*")

        master_database = Database(DB_PATH, DB_INFO_FILE_NAME)

        master_database.read_pinchao_formatting(
            os.path.join(LOAD_PATH, 'Database-Structures')
        )

        master_database.print_metadata()

        # all_struct_names, structures = zip(*master_database.structures.items())
        all_struct_names, struct_natoms = zip(*master_database.force_weighting.items())
        num_structs = len(struct_natoms)

        worker_ranks = partools.compute_procs_per_subset(
            struct_natoms, world_size
        )

        print("worker_ranks:", worker_ranks)

    else:
        potential_template = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

    # Send all information necessary to building evaluation functions
    potential_template = world_comm.bcast(potential_template, root=0)
    num_structs = world_comm.bcast(num_structs, root=0)

    # each Manager is in charge of a single structure
    world_group = world_comm.Get_group()

    all_rank_lists = world_comm.bcast(worker_ranks, root=0)

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

        struct_name = manager_comm.scatter(all_struct_names, root=0)

        print(
            "Manager", manager_rank, "received structure", struct_name, "plus",
            len(worker_ranks), "processors for evaluation", flush=True
        )

    else:
        struct_name = None
        manager_rank = None

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

    """
    pseudo-code:

    master creates swarm
    run LM on a portion of the swarm
    loop:
        evaluate swarm:
            bcast pop to managers
            evaluate on workers
            gather results on master
            compute fitnesses
        master updates positions/velocities
        if desired, run LM again on portion of swarm

    """

    def fxn_wrap(master_pop):
        """Master: returns all potentials for all structures"""
        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
            # pop = np.ones((POP_SIZE, 5))
        else:
            pop = None

        if FLATTEN_LANDSCAPE:
            master_pop = local_minimization(
                master_pop, fxn_wrap, grad_wrap, world_comm, is_master,
                num_to_minimize=NMIN, nsteps=FLAT_NSTEPS
            )

            pop = manager_comm.bcast(master_pop, root=0)

        # eng, per_u_max_ni = manager.compute_energy(pop)
        eng = manager.compute_energy(pop)
        fcs = manager.compute_forces(pop)

        # fitnesses = csr_matrix((pop.shape[0], 2 * num_structs))
        fitnesses = 0

        # if return_ni:
        #     max_ni = 0

        if is_manager:
            mgr_eng = manager_comm.gather(eng, root=0)

            # if return_ni:
            #     mgr_ni = manager_comm.gather(per_u_max_ni, root=0)

            mgr_fcs = manager_comm.gather(fcs, root=0)

            if is_master:
                # note: can't stack mgr_fcs b/c different dimensions per struct
                all_eng = np.vstack(mgr_eng)
                all_fcs = mgr_fcs

                w_energies = np.zeros((len(pop), num_structs))
                t_energies = np.zeros(num_structs)

                fcs_fitnesses = np.zeros((len(pop), num_structs))

                for s_id, name in enumerate(all_struct_names):
                    w_energies[:, s_id] = all_eng[s_id]
                    t_energies[s_id] = master_database.true_energies[name]

                    # if name == master_database.reference_struct:
                    #     ref_energy = w_energies[:, s_id]

                    force_weights = master_database.force_weighting[struct_name]
                    w_fcs = all_fcs[s_id] * force_weights[:, np.newaxis]
                    true_fcs = master_database.true_forces[name]

                    fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
                    fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

                    fcs_fitnesses[:, s_id] = fcs_err

                # w_energies -= ref_energy
                # t_energies -= master_database.reference_energy

                eng_fitnesses = np.zeros((len(pop), num_structs))

                for s_id, (w_eng,t_eng) in enumerate(
                    zip(w_energies.T, t_energies)):
                    eng_fitnesses[:, s_id] = (w_eng - t_eng) ** 2

                # TODO: this only works for how Pinchao's DB is formatted

                # eng_fitnesses = np.zeros(
                #     (len(pop), len(master_database.reference_structs))
                # )
                #
                # for fit_id, (s_name, ref) in enumerate(
                #         master_database.reference_structs.items()):
                #     r_name = ref.ref_struct
                #     true_ediff = ref.energy_difference
                #
                #     # find index of structures to know which energies to use
                #     s_id = all_struct_names.index(s_name)
                #     r_id = all_struct_names.index(r_name)
                #
                #     comp_ediff = w_energies[:, s_id] - w_energies[:, r_id]
                #     # comp_ediff = 0
                #
                #     eng_fitnesses[:, fit_id] = (comp_ediff - true_ediff) ** 2

                fitnesses = np.hstack([eng_fitnesses, fcs_fitnesses])

                # print(np.sum(fitnesses, axis=1), flush=True)

        # if return_ni: return fitnesses, per_u_max_ni
        # else: return fitnesses
        return fitnesses

    # @profile
    def grad_wrap(master_pop):
        """Evalautes the gradient for all potentials in the population"""

        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
        else:
            pop = None

        # pop = worker_comm.bcast(pop, root=0)

        eng, _ = manager.compute_energy(pop)
        fcs = manager.compute_forces(pop)

        eng_grad = manager.compute_energy_grad(pop)
        fcs_grad = manager.compute_forces_grad(pop)

        # gradient = csr_matrix((pop.shape[0], 2 * num_structs, pop.shape[1]))
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

                w_energies = np.zeros((len(pop), num_structs))
                t_energies = np.zeros(num_structs)

                fcs_grad_vec = np.zeros(
                    (len(pop), potential_template.pvec_len, num_structs)
                )

                ref_energy = 0

                for s_id, name in enumerate(all_struct_names):
                    w_energies[:, s_id] = all_eng[s_id]
                    t_energies[s_id] = master_database.true_energies[name]

                    if name == master_database.reference_struct:
                        ref_energy = w_energies[:, s_id]

                    force_weights = master_database.force_weighting[struct_name]
                    w_fcs = all_fcs[s_id] * force_weights[:, np.newaxis]
                    true_fcs = master_database.true_forces[name]

                    fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2

                    fcs_grad = mgr_fcs_grad[s_id]

                    scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
                    summed = scaled.sum(axis=1).sum(axis=1)

                    fcs_grad_vec[:, :, s_id] += (2 * summed / 10)

                # w_energies -= ref_energy
                # t_energies -= database.reference_energy

                eng_grad_vec = np.zeros(
                    (len(pop), potential_template.pvec_len, num_structs)
                )

                for s_id, (name, w_eng, t_eng), in enumerate(
                    zip(all_struct_names, w_energies.T, t_energies)):

                    eng_err = (w_eng - t_eng)
                    eng_grad = mgr_eng_grad[s_id]

                    eng_grad_vec[:, :, s_id] += (
                        eng_err[:, np.newaxis] * eng_grad * 2
                    )

                # eng_grad_vec = np.zeros(
                #     (len(pop), potential_template.pvec_len,
                #      len(master_database.reference_structs))
                # )
                #
                # for fit_id, (s_name, ref) in enumerate(
                #         master_database.reference_structs.items()):
                #     r_name = ref.ref_struct
                #     true_ediff = ref.energy_difference
                #
                #     # find index of structures to know which energies to use
                #     s_id = all_struct_names.index(s_name)
                #     r_id = all_struct_names.index(r_name)
                #
                #     comp_ediff = w_energies[:, s_id] - w_energies[:, r_id]
                #
                #     eng_err = comp_ediff - true_ediff
                #     s_grad = mgr_eng_grad[s_id]
                #     r_grad = mgr_eng_grad[r_id]
                #
                #     eng_grad_vec[:, :, fit_id] += (
                #             eng_err[:, np.newaxis] * (s_grad - r_grad) * 2
                #     )

                indices = np.where(potential_template.active_mask)[0]
                tmp_eng = eng_grad_vec[:, indices, :]
                tmp_fcs = fcs_grad_vec[:, indices, :]

                gradient = np.dstack([tmp_eng, tmp_fcs]).swapaxes(1, 2)

        return gradient

    # initialize swarm -- 'positions' are the potentials
    if is_master:
        swarm_positions = init_positions(SWARM_SIZE, potential_template)
        swarm_velocities = init_velocities(SWARM_SIZE, potential_template)
    else:
        swarm_positions = None
        swarm_velocities = None

    swarm_positions = np.array(swarm_positions)

    init_fit = fxn_wrap(swarm_positions)

    if is_master:
        init_fit = np.sum(init_fit, axis=1)
        print("MASTER: initial (UN-minimized) fitnesses:", init_fit, flush=True)
        print("Average value:", np.average(init_fit), flush=True)

    swarm_positions = local_minimization(
        swarm_positions, fxn_wrap, grad_wrap, world_comm, is_master,
        num_to_minimize=NMIN, nsteps=NUM_LMIN_STEPS,
    )

    new_fit = fxn_wrap(swarm_positions)

    if is_master:
        all_fitnesses = np.sum(new_fit, axis=1)
        print(
            "MASTER: initial (minimized) fitnesses:", all_fitnesses, flush=True
        )

        # record initial personal bests
        personal_best_fitnesses = all_fitnesses
        personal_best_positions = np.copy(swarm_positions)

        # record initial global best
        min_idx = np.argmin(all_fitnesses)

        global_best_pos = swarm_positions[min_idx]
        global_best_fit = all_fitnesses[min_idx]

        # prepare logging
        global_best_pos_trace = []
        global_best_fit_trace = []

        # log_f = open(LOG_FILENAME, 'wb')
        # np.savetxt(log_f, [np.concatenate([[0], all_fitnesses.ravel()])])
        # log_f.close()

        start = time.time()
        print("MASTER: step g_best max avg std avg_vel_mag", flush=True)
    else:
        global_best_fit = None

    global_best_fit = world_comm.bcast(global_best_fit, root=0)


    i = 0
    while (i < MAX_NUM_PSO_STEPS) and (global_best_fit > FITNESS_THRESH):
        if is_master:

            avg_vel = np.average(np.linalg.norm(swarm_velocities, axis=1))

            print("STEP: {} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}".format(
                i,
                global_best_fit,
                np.max(all_fitnesses),
                np.average(all_fitnesses),
                np.std(all_fitnesses),
                avg_vel),
                flush=True
                )

        # Run local minimizer if desired
        if (i % LMIN_FREQUENCY == 0) and DO_LOCAL_MIN:
            swarm_positions = local_minimization(
                swarm_positions, fxn_wrap, grad_wrap, world_comm, is_master,
                num_to_minimize=NMIN, nsteps=NUM_LMIN_STEPS,
            )

        if is_master:
            # Generate random step directions using PSO algorithm
            rp = np.random.random(swarm_positions.shape)
            rg = np.random.random(swarm_positions.shape)

            new_velocity = \
                MOMENTUM_WEIGHT*swarm_velocities + \
                COGNITIVE_WEIGHT*rp*(personal_best_positions - swarm_positions) +\
                SOCIAL_WEIGHT*rg*(global_best_pos - swarm_positions)

            swarm_positions += new_velocity
            swarm_velocities = new_velocity

        fitnesses = fxn_wrap(swarm_positions)

        # update personal bests
        if is_master:
            all_fitnesses = np.sum(fitnesses, axis=1)

            found_better = np.where(all_fitnesses < personal_best_fitnesses)[0]
            personal_best_fitnesses[found_better] = all_fitnesses[found_better]
            personal_best_positions[found_better] = swarm_positions[found_better]

            min_idx = np.argmin(all_fitnesses)

            if all_fitnesses[min_idx] < global_best_fit:
                global_best_pos = swarm_positions[min_idx]
                global_best_fit = all_fitnesses[min_idx]

            log_f = open(LOG_FILENAME, 'ab')
            np.savetxt(log_f, [np.concatenate([[i], all_fitnesses.ravel()])])
            log_f.close()

            f = open(BEST_TRACE_FILENAME, 'ab')
            np.savetxt(f, [np.concatenate([[global_best_fit],
                                           global_best_pos.ravel()])])
            f.close()

        global_best_fit = world_comm.bcast(global_best_fit, root=0)

        i += 1

    if is_master:
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

def initialize_potential_template():
    # TODO: BW settings
    potential = MEAM.from_file(LOAD_PATH + 'TiO.meam.spline')

    x_pvec, seed_pvec, indices = src.meam.splines_to_pvec(
        potential.splines)

    potential_template = Template(
        pvec_len=116,
        u_ranges=[(-1, 1), (-1, 1)],
        spline_ranges=[(-1, 4), (-0.5, 0.5), (-1, 1), (-9, 3), (-30, 15),
                       (-0.5, 1), (-0.2, -0.4), (-2, 3), (-7.5, 12.5),
                       (-8, 2), (-1, 1), (-1, 0.2)],
        spline_indices=[(0, 15), (15, 22), (22, 37), (37, 50), (50, 57),
                        (57, 63), (63, 70), (70, 82), (82, 89),
                        (89, 99), (99, 106), (106, 116)]
    )

    potential_template.pvec = seed_pvec.copy()
    mask = np.ones(potential_template.pvec_len)

    mask[:15] = 0 # phi_Ti

    potential_template.pvec[19] = 0;
    mask[19] = 0  # rhs phi_TiO knot
    potential_template.pvec[21] = 0;
    mask[21] = 0  # rhs phi_TiO deriv

    mask[22:37] = 0  # phi_O
    mask[37:50] = 0  # rho_Ti

    potential_template.pvec[54] = 0;
    mask[54] = 0  # rhs rho_O knot
    potential_template.pvec[56] = 0;
    mask[56] = 0  # rhs rho_O deriv

    mask[57:63] = 0  # U_Ti
    mask[70:82] = 0  # f_Ti

    potential_template.pvec[86] = 0;
    mask[86] = 0  # rhs f_O knot
    potential_template.pvec[88] = 0;
    mask[88] = 0  # rhs f_O deriv

    mask[89:99] = 0  # g_Ti
    mask[106:116] = 0  # g_O

    potential_template.active_mask = mask

    return potential_template


def local_minimization(
    swarm_positions, fxn, grad, world_comm, is_master, num_to_minimize, nsteps=20
):

    def lm_fxn_wrap(raveled_pop, original_shape):
        val = fxn(
            raveled_pop.reshape(original_shape)
        )

        val = world_comm.bcast(val, root=0)

        return val.ravel()

    def lm_grad_wrap(raveled_pop, original_shape):
        grads = grad(
            raveled_pop.reshape(original_shape)
        )

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

        return padded_grad

    swarm_positions = world_comm.bcast(swarm_positions, root=0)
    swarm_positions = np.array(swarm_positions)

    opt_results = least_squares(
        lm_fxn_wrap, swarm_positions[:num_to_minimize].ravel(), lm_grad_wrap,
        method='lm', max_nfev=nsteps,
        args=(swarm_positions[:num_to_minimize].shape,)
    )

    if is_master:
        new_pop = opt_results['x'].reshape(
            swarm_positions[:num_to_minimize].shape
        )
    else:
        new_pop = None

    org_fits = fxn(swarm_positions)
    new_fits = fxn(new_pop)

    if is_master:
        updated_swarm_positions = list(swarm_positions)

        for i, ind in enumerate(new_pop):
            if np.sum(new_fits[i]) < np.sum(org_fits[i]):
                updated_swarm_positions[i] = creator.Individual(new_pop[i])
            else:
                updated_swarm_positions[i] = creator.Individual(updated_swarm_positions[i])

        swarm_positions = updated_swarm_positions

    return swarm_positions


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
