import numpy as np
import random
np.set_printoptions(precision=8, linewidth=np.inf, suppress=True)
np.random.seed(42)
random.seed(42)

import os
import glob
import pickle
from mpi4py import MPI
from scipy.optimize import least_squares

################################################################################

# TODO: BW setting
os.chdir('/home/jvita/scripts/s-meam/project/')

POTS_PER_PROC = 1
NUM_LMIN_STEPS = 50

# TODO: BW setting
LOAD_PATH = "data/fitting_databases/leno-redo/"
# LOAD_PATH = "/projects/sciteam/baot/leno-redo/"

DB_PATH = LOAD_PATH + 'structures/'
DB_INFO_FILE_NAME = LOAD_PATH + 'rhophi/info'

SAVEFILE_NAME = LOAD_PATH + "cluster_optimized.dat"

################################################################################

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    is_master_node = (rank == 0)

    if is_master_node:
        structures, weights = load_structures()
        print(structures.keys(), flush=True)

        true_forces, true_energies = load_true_values(structures.keys())

        ex_struct = structures[list(structures.keys())[0]]
        type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
    else:
        spline_indices = None
        structures = None
        weights = None
        true_forces = None
        true_energies = None

    spline_indices = comm.bcast(spline_indices, root=0)
    structures = comm.bcast(structures, root=0)
    weights = comm.bcast(weights, root=0)
    true_forces = comm.bcast(true_forces, root=0)
    true_energies = comm.bcast(true_energies, root=0)

    if is_master_node:
        tmp_pop = initialize_population(mpi_size*POTS_PER_PROC)
        split_pop = np.split(tmp_pop, mpi_size)
    else:
        split_pop = None

    pop = comm.scatter(split_pop, root=0)
    pop = np.atleast_2d(pop)
    N = pop.shape[0]

    eval_fxn, grad_fxn, trace = build_evaluation_functions(structures, weights,
                                                    true_forces, true_energies,
                                                    spline_indices, N)

    optimized = np.zeros(pop.shape)

    CHECKPOINT_FILENAME = LOAD_PATH + "rank{}_trace.dat".format(rank)

    for i,indiv in enumerate(pop):
        print("Rank", rank, "optimizing pot {}/{}".format(i+1, POTS_PER_PROC),
              flush=True)

        opt_results = least_squares(eval_fxn, indiv, grad_fxn, method='lm',
                                    max_nfev=NUM_LMIN_STEPS,
                                    args=(i, CHECKPOINT_FILENAME))

        optimized[i] = opt_results['x']

    print("Rank", rank, "finished all optimizations", flush=True)

    # trace = np.vstack(trace)

    all_optimized = comm.gather(optimized, root=0)
    # all_traces = comm.gather(trace, root=0)

    if is_master_node:
        all_optimized = np.vstack(all_optimized)
        # all_traces = np.vstack(all_traces)

        outfile = open(SAVEFILE_NAME, 'ab')

        np.savetxt(outfile, all_optimized)
        # np.savetxt(outfile, all_traces)

        outfile.close()

################################################################################

def load_structures():
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


def initialize_population(N):

    population = np.zeros((N,83))

    ranges = [(-0.5, 0.5), (-1, 4), (-1, 1), (-9, 3), (-30, 15), (-0.5, 1),
            (-0.2, 0.4)]

    indices = [(0,13), (15,20), (22,35), (37,48), (50,55), (57,61), (63,68)]

    for i in range(N):
        ind = np.zeros(83)

        for rng,ind_tup in zip(ranges, indices):
            r_lo, r_hi = rng
            i_lo, i_hi = ind_tup

            ind[i_lo:i_hi] = np.random.random(i_hi-i_lo)*(r_hi-r_lo) + r_lo

        population[i] = ind

    return population

def build_evaluation_functions(structures, weights, true_forces, true_energies,
                               spline_indices, n_pots):

    trace = [[] for i in range(n_pots)]

    def fxn(pot, pot_id, filename):

        # Convert list of Individuals into a numpy array
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

        slave_outfile = open(filename, 'ab')
        np.savetxt(slave_outfile, [np.array([pot_id+1, np.sum(fitness)])])
        slave_outfile.close()

        print(np.sum(fitness), flush=True)

        trace[pot_id].append(np.sum(fitness))

        return fitness

    def grad(pot, pot_id, file):
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

    return fxn, grad, trace

################################################################################

if __name__ == "__main__":
    main()
