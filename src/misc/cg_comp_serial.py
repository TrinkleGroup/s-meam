import os
os.chdir('/home/jvita/scripts/s-meam/project/')
print(os.getcwd())
import time
import h5py
import glob
import numpy as np
np.random.seed(42)
from scipy.optimize import fmin_powell, fmin_cg, line_search
import warnings
import sys
sys.path.append('./')

import src.lammpsTools
from src.worker import Worker

################################################################################

LOAD_PATH = "/home/jvita/scripts/s-meam/project/data/fitting_databases/lj/"

DB_FILE_NAME = LOAD_PATH + 'structures.hdf5'

THRESH = 10

################################################################################

class AchievedThreshold(Exception):
    pass

class CallbackCollector:

    def __init__(self, f, fprime=None, thresh=0.1):
        self.f = f
        self._thresh = thresh
        self.n_iters = 0

    def __call__(self, xk):
        self.n_iters += 1
        fval = self.f(xk)

        if fval < self._thresh:
            print('The THRESHOLD HAS BEEN ACHIEVED')
            self.x_opt = xk
            raise AchievedThreshold

# @profile
def main():
    print("Loading structures ...")
    structures, weights = load_structures_on_master()

    print("Loading true values ...")
    true_forces, true_energies = load_true_values(structures.keys())

    ex_struct = structures[list(structures.keys())[0]]
    type_indices, spline_indices = find_spline_type_deliminating_indices(ex_struct)
    pvec_len = ex_struct.len_param_vec

    fxn, f_calls, grad_fxn, g_calls = build_evaluation_functions(structures, weights, true_forces,
                                           true_energies, spline_indices)

    guess = init_potential()

    guess = np.vstack([guess]*7)

    fxn(guess)
    grad_fxn(guess)

    return

    num_steps=None

    cb = CallbackCollector(fxn, grad_fxn, thresh=THRESH)
    def cb(x):
        pass

    print("Performing CG minimization using the analytical gradient ...")
    start = time.time()
    try:
        cg_xopt, _, cg_n_f_calls, cg_n_g_calls, _ = fmin_cg(fxn, guess, grad_fxn,
                gtol=1e-8, full_output=True, callback=cb)
    except AchievedThreshold:
        cg_xopt = cb.x_opt
        cg_n_f_calls = f_calls[0]
        cg_n_g_calls = g_calls[0]

    f_calls[0] = 0
    g_calls[0] = 0

    cg_time = time.time() - start

    print("CG")
    print("num CG steps:", cb.n_iters)
    print("num fxn calls:", cg_n_f_calls - cb.n_iters)
    print("num grad calls:", cg_n_g_calls)

    np.savetxt('comp_cg_results.dat', cg_xopt)

    cb2 = CallbackCollector(fxn, thresh=THRESH)
    def cb2(x):
        pass

    print("Performing Powell minimization ...")
    start = time.time()
    try:
        pow_xopt, _, _, _, pow_n_f_calls, _ = fmin_powell(fxn, guess,
                                                          maxiter=num_steps, disp=0,
                                                          ftol=1e-6, xtol=1e-5,
                                                          full_output=True,
                                                          callback=cb2)
    except AchievedThreshold:
        pow_xopt = cb2.x_opt
        pow_n_f_calls = f_calls[0]
        pow_n_g_calls = g_calls[0]

        f_calls[0] = 0
        g_calls[0] = 0
    pow_time = time.time() - start

    np.savetxt('comp_powell_results.dat', pow_xopt)

    print("Powell")
    print("num Powell steps:", cb2.n_iters)
    print("num fxn calls:", pow_n_f_calls)

    print("CG time:", cg_time)
    print("Powell time:", pow_time)

################################################################################

def load_structures_on_master():
    """Builds Worker objects from the HDF5 database of stored values. Note that
    database weights are determined HERE.
    """

    # database = h5py.File(DB_FILE_NAME, 'a',)
    # weights = {key:1 for key in database.keys()}
    # 
    # structures = {}
    # weights = {}
    # 
    # start = time.time()
    # 
    # for name in database.keys():
    #     if 'dimer' in name:
    #         weights[name] = 1
    #         structures[name] = Worker.from_hdf5(database, name)
    # 
    # database.close()

    structures = {}
    weights = {}

    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/lj/new_lj.meam')
    x_pvec, y_pvec, indices = src.meam.splines_to_pvec(pot.splines)

    for name in glob.glob(LOAD_PATH + 'data/*'):
        if 'dimer' in name:
            atoms = src.lammpsTools.atoms_from_file(name, pot.types)

            short_name = os.path.split(name)[-1]
            short_name = '.'.join(short_name.split('.')[1:])

            weights[short_name] = 1

            structures[short_name] = Worker(atoms, x_pvec, indices, pot.types)

    # structures = {key:val for (key,val) in list(structures.items())[0]}
    tmp_key = 'dimer_aa_1.65'
    structures = {tmp_key:structures[tmp_key]}

    print(structures.keys())
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

    def fxn_wrapper(fxn):
       ncalls = [0]

       def wrapper(*wrapper_args):
           ncalls[0] += 1
           return fxn(*(wrapper_args))

       return ncalls, wrapper

    def fxn(pot):
        # Convert list of Individuals into a numpy array
        # full = np.vstack(population)

        # hard-coded for phi splines only TODO: remove this later
        pot = np.atleast_2d(pot)
        full = np.hstack([pot, np.zeros((pot.shape[0], 108))])

        fitness = 0

        # Compute error for each worker on MPI node
        for name in structures.keys():
            w = structures[name]

            fcs_err = w.compute_forces(full) - true_forces[name]
            eng_err = w.compute_energy(full) - true_energies[name]

            # Scale force errors
            fcs_err = np.linalg.norm(fcs_err, axis=(1,2)) / np.sqrt(10)

            fitness += fcs_err*fcs_err*weights[name]
            fitness += eng_err*eng_err*weights[name]

        print(fitness, flush=True)
        return fitness

    def grad(pot):
        full = np.hstack([pot, np.zeros((pot.shape[0], 108))])

        grad_vec = np.zeros((pot.shape[0], full.shape[1]))

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

            grad_vec += eng_err[:, np.newaxis]*eng_grad*2
            grad_vec += 2*summed / 10

        return grad_vec[:, :36]

    f_calls, fxn = fxn_wrapper(fxn)
    g_calls, grad = fxn_wrapper(grad)

    return fxn, f_calls, grad, g_calls
    # return fxn, fd_gradient

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

def cost_fxn_fd_grad(fxn, y_pvec):
    N = y_pvec.ravel().shape[0]

    cd_points = np.array([y_pvec] * N * 2)

    h = 1e-8

    for l in range(N):
        cd_points[2 * l, l] += h
        cd_points[2 * l + 1, l] -= h

        cd_evaluated = fxn(np.array(cd_points))
        fd_gradient = np.zeros(N)

    for l in range(N):
        fd_gradient[l] = \
            (cd_evaluated[2 * l] - cd_evaluated[2 * l + 1]) / h / 2

    return fd_gradient

################################################################################

if __name__ == "__main__":
    main()
