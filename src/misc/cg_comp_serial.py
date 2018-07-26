import time
import h5py
import numpy as np
from scipy.optimize import fmin_powell, fmin_cg
import sys
sys.path.append('./')
sys.path.append('/home/jvita/scripts/s-meam/project/')
print(sys.path)

from src.worker import Worker

################################################################################

LOAD_PATH = "/home/jvita/scripts/s-meam/project/data/fitting_databases/lj/"

DB_FILE_NAME = LOAD_PATH + 'structures.hdf5'

################################################################################

def main():
    print("Loading structures ...")
    structures, weights = load_structures_on_master()

    print("Loading true values ...")
    true_forces, true_energies = load_true_values(structures.keys())

    ex_struct = structures[list(structures.keys())[0]]
    type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
    pvec_len = ex_struct.len_param_vec

    fxn, grad = build_evaluation_functions(structures, weights, true_forces,
                                           true_energies, spline_indices)

    guess = init_potential()

    num_steps = None

    print("Performing Powell minimization ...")
    start = time.time()
    pow_xopt, _, _, _, pow_n_f_calls, _ = fmin_powell(fxn, guess,
                                                      maxiter=num_steps, disp=0,
                                                      ftol=10, xtol=10,
                                                      full_output=True)
    pow_time = time.time() - start

    print("Performing CG minimization using the analytical gradient ...")
    start = time.time()
    cg_xopt, _, cg_n_f_calls, cg_n_g_calls, _ = fmin_cg(fxn, grad,
            guess, maxiter=num_steps, disp=0,
            ftol=10, xtol=10, full_output=True)
    cg_time = time.time() - start

    print("Performing CG minimization using finite differences ...")
    start = time.time()
    cg_fd_xopt, _, cg_fd_n_f_calls, cg_fd_n_g_calls, _ = fmin_cg(fxn,
            guess, maxiter=num_steps, disp=0,
            ftol=10, xtol=10, full_output=True)
    cg_fd_time = time.time() - start

    print("Name\tn_evals\traw_time")
    print("Powell\t" + str(pow_n_f_calls) + "\t" + str(pow_time))
    print("CG\t" + str(cg_n_f_calls) + "\t" + str(cg_time))
    print("CG+FD\t" + str(cg_fd_n_f_calls) + "\t" + str(cg_fd_time))

################################################################################

def load_structures_on_master():
    """Builds Worker objects from the HDF5 database of stored values. Note that
    database weights are determined HERE.
    """

    database = h5py.File(DB_FILE_NAME, 'a',)
    weights = {key:1 for key in database.keys()}

    structures = {}
    weights = {}

    start = time.time()

    for name in database.keys():
        if 'dimer' in name:
            weights[name] = 1
            structures[name] = Worker.from_hdf5(database, name)

    database.close()

    return structures, weights

def load_true_values(all_names):
    """Loads the 'true' values according to the database provided"""

    true_forces = {}
    true_energies = {}

    for name in all_names:

        fcs = np.genfromtxt(open(LOAD_PATH + 'info/info.' + name, 'rb'),
                skip_header=1)
        eng = np.genfromtxt(open(LOAD_PATH + 'info/info.' + name, 'rb'),
                max_rows=1)

        true_forces[name] = fcs
        true_energies[name] = eng

    return true_forces, true_energies

def build_evaluation_functions(structures, weights, true_forces, true_energies,
        spline_indices):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn(pot):
        # Convert list of Individuals into a numpy array
        # full = np.vstack(population)

        # hard-coded for phi splines only TODO: remove this later
        # full = np.hstack([full, np.zeros((full.shape[0], 54))])
        full = np.hstack([pot, np.zeors(108)])

        fitness = np.zeros(full.shape[0])

        # Compute error for each worker on MPI node
        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(full) - true_forces[name]
            eng_err = w.compute_energy(full) - true_energies[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitness += fcs_err*fcs_err*weights[name]
            fitness += eng_err*eng_err*weights[name]

        return fitness

    def grad(pot):
        full = np.hstack([full, np.zeors(108)])

        grad_vec = np.zeros(full.shape[0])

        for name in structures.keys():
            w = structures[name]

            eng_err = w.compute_energy(full) - true_energies[name]
            fcs_err = w.compute_forces(full) - true_forces[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            # compute gradients
            eng_grad = w.energy_gradient_wrt_pvec(full)
            fcs_grad = w.forces_gradient_wrt_pvec(full)
            fcs_grad = np.linalg.norm(fcs_grad, axis=(1,2))

            grad_vec += 2*eng_err*eng_grad
            grad_vec += 2*fcs_err*fcs_grad

    return fxn, grad

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

def init_potential():
    params = np.zeros(36)

    params[:12] += np.linspace(0.2, 0.8*1.75, 12)[::-1]
    params[:12] += np.random.normal(size=(12,), scale=0.1)

    params[12:24] += np.linspace(0.2, 0.8*1.75, 12)[::-1]
    params[12:24] += np.random.normal(size=(12,), scale=0.1)

    params[24:36] += np.linspace(0.2, 0.8*1.75, 12)[::-1]
    params[24:36] += np.random.normal(size=(12,), scale=0.1)

    return params

################################################################################

if __name__ == "__main__":
    main()
