import sys
sys.path.append('/home/jvita/scripts/s-meam/project/')

import numpy as np
from lmfit import minimize, Parameters
from mpi4py import MPI

from src.database import Database
from src.potential_templates import Template


################################################################################

def main():
    LOAD_PATH = "/home/jvita/scripts/s-meam/data/fitting_databases/leno-redo/"
    DB_PATH = LOAD_PATH + 'structures'
    DB_INFO_FILE_NAME = LOAD_PATH + 'rhophi/info'

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    is_master = (rank == 0)

    if is_master:
        database = Database(DB_PATH, DB_INFO_FILE_NAME)
        potential_template = initialize_potential_template()

        subsets, _, _ = group_database_subsets(database, mpi_size)
    else:
        subsets = None
        potential_template = None

    subset = comm.scatter(subsets, root=0)
    potential_template = comm.bcast(potential_template, root=0)

    fxn, grad = build_evaluation_functions(subset, potential_template)

    def fxn_wrap(inp_params, comm):

        inp = np.array([inp_params[n].value for n in inp_params])

        cost = fxn(inp)
        all_costs = comm.gather(cost, root=0)

        if is_master:
            value = np.concatenate(all_costs)
            print('fxn.shape =', value.shape, flush=True)
        else:
            value = None

        value = comm.bcast(value, root=0)

        return value

    # TODO: each proc should get a subset of the database

    def grad_wrap(inp_params, comm):

        inp = np.array([inp_params[n].value for n in inp_params])

        grad_val = grad(inp)
        all_grads = comm.gather(grad_val, root=0)

        if is_master:
            value = np.vstack(all_grads)
            print("grad shape =", value.shape, flush=True)
        else:
            value = None

        value = comm.bcast(value, root=0)

        return value

    params = Parameters()

    guess = potential_template.generate_random_instance()
    guess = guess[np.where(potential_template.active_mask)[0]]

    for i, x in enumerate(guess):
        params.add("var_" + str(i), value=x)

    out = minimize(fxn_wrap, params, Dfun=grad_wrap, args=(comm,))

def initialize_potential_template():

    # TODO: BW settings
    potential_template = Template(
        pvec_len=137,
        u_ranges = [(-1, 1), (-1, 1)],
        spline_ranges=[(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3),
                       (-0.5, 1), (-0.5, 1), (-2, 3), (-2, 3), (-7, 2),
                       (-7, 2), (-7, 2)],
        spline_indices=[(0, 15), (15, 30), (30, 45), (45, 58), (58, 71),
                         (71, 77), (77, 83), (83, 95), (95, 107),
                         (107, 117), (117, 127), (127, 137)]
    )

    mask = np.ones(potential_template.pvec.shape)

    potential_template.pvec[12] = 0; mask[12] = 0 # rhs phi_A knot
    potential_template.pvec[14] = 0; mask[14] = 0 # rhs phi_A deriv

    potential_template.pvec[27] = 0; mask[27] = 0 # rhs phi_B knot
    potential_template.pvec[29] = 0; mask[29] = 0 # rhs phi_B deriv

    potential_template.pvec[42] = 0; mask[42] = 0 # rhs phi_B knot
    potential_template.pvec[44] = 0; mask[44] = 0 # rhs phi_B deriv

    potential_template.pvec[55] = 0; mask[55] = 0 # rhs rho_A knot
    potential_template.pvec[57] = 0; mask[57] = 0 # rhs rho_A deriv

    potential_template.pvec[68] = 0; mask[68] = 0 # rhs rho_B knot
    potential_template.pvec[70] = 0; mask[70] = 0 # rhs rho_B deriv

    potential_template.pvec[92] = 0; mask[92] = 0 # rhs f_A knot
    potential_template.pvec[94] = 0; mask[94] = 0 # rhs f_A deriv

    potential_template.pvec[104] = 0; mask[104] = 0 # rhs f_B knot
    potential_template.pvec[106] = 0; mask[106] = 0 # rhs f_B deriv

    # potential_template.pvec[83:] = 0; mask[83:] = 0 # EAM params only
    potential_template.pvec[45:] = 0; mask[45:] = 0 # EAM params only

    potential_template.active_mask = mask

    return potential_template

def build_evaluation_functions(database, potential_template):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn(pot):
        full = np.atleast_2d(pot)
        full = potential_template.insert_active_splines(full)

        w_energies = np.zeros(len(database.structures))
        t_energies = np.zeros(len(database.structures))

        fcs_fitnesses = np.zeros(len(database.structures))

        ref_energy = 0

        for j,name in enumerate(database.structures.keys()):

            w = database.structures[name]

            w_energies[j] = w.compute_energy(full, potential_template.u_ranges)
            t_energies[j] = database.true_energies[name]

            if name == database.reference_struct:
                ref_energy = w_energies[j]

            w_fcs = w.compute_forces(full, potential_template.u_ranges)
            true_fcs = database.true_forces[name]

            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
            fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

            fcs_fitnesses[j] = fcs_err

        w_energies -= ref_energy
        t_energies -= database.reference_energy

        eng_fitnesses = np.zeros(len(database.structures))

        for j, (w_eng, t_eng) in enumerate(zip(w_energies, t_energies)):
            eng_fitnesses[j] = (w_eng - t_eng) ** 2

        fitnesses = np.concatenate([eng_fitnesses, fcs_fitnesses])

        return fitnesses

    def grad(pot):
        full = np.atleast_2d(pot)
        full = potential_template.insert_active_splines(full)

        fcs_grad_vec = np.zeros((len(database.structures), 137))

        w_energies = np.zeros(len(database.structures))
        t_energies = np.zeros(len(database.structures))

        ref_energy = 0

        for j,name in enumerate(database.structures.keys()):
            w = database.structures[name]

            w_energies[j] = w.compute_energy(full, potential_template.u_ranges)
            t_energies[j] = database.true_energies[name]

            if name == database.reference_struct:
                ref_energy = w_energies[j]

            w_fcs = w.compute_forces(full, potential_template.u_ranges)
            true_fcs = database.true_forces[name]

            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10))

            fcs_grad = w.forces_gradient_wrt_pvec(full, potential_template.u_ranges)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            fcs_grad_vec[j] += (2 * summed / 10).ravel()

        w_energies -= ref_energy
        t_energies -= database.reference_energy

        eng_grad_vec = np.zeros((len(database.structures), 137))
        for j, (w_eng, t_eng) in enumerate(zip(w_energies, t_energies)):
            eng_err = (w_eng - t_eng)
            eng_grad = w.energy_gradient_wrt_pvec(full, potential_template.u_ranges)

            eng_grad_vec[j] += (eng_err * eng_grad * 2).ravel()

        grad_vec = np.vstack([eng_grad_vec, fcs_grad_vec])
        tmp = grad_vec[:, np.where(potential_template.active_mask)[0]]

        return tmp

    return fxn, grad

def compute_relative_weights(database):
    work_weights = []

    name_list = list(database.structures.keys())

    for name in name_list:
        work_weights.append(database.structures[name].natoms)

    work_weights = np.array(work_weights)
    work_weights = work_weights / np.min(work_weights)
    work_weights = work_weights*work_weights # cost assumed to scale as N^2

    return work_weights, name_list


def group_database_subsets(database, mpi_size):
    """Groups workers based on evaluation time to help with load balancing.

    Returns:
        distributed_work (list): the partitioned work
        work_per_proc (float): approximate work done by each processor
    """
    # TODO: record scaling on first run, then redistribute to load balance

    unassigned_structs = list(database.structures.keys())

    work_weights, name_list = compute_relative_weights(database)

    work_per_proc = np.sum(work_weights) / mpi_size


    work_weights = work_weights.tolist()

    assignments = []
    grouped_work = []

    for _ in range(mpi_size):
        cumulated_work = 0

        names = []
        # work_for_one_proc = []

        while unassigned_structs and (cumulated_work < work_per_proc):
            names.append(unassigned_structs.pop())
            cumulated_work += work_weights.pop()

        mini_database = Database.manual_init(
            {name:database.structures[name] for name in names},
            {name:database.true_energies[name] for name in names},
            {name:database.true_forces[name] for name in names},
            {name:database.weights[name] for name in names},
            database.reference_struct,
            database.reference_energy
        )

        assignments.append(mini_database)
        # assignments.append(names)
        grouped_work.append(cumulated_work)

    return assignments, grouped_work, work_per_proc


if __name__ == "__main__":
    main()
