import numpy as np
from src.database import Database

def build_evaluation_functions(database, potential_template):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    def fxn(pot):
        full = np.atleast_2d(pot)
        full = potential_template.insert_active_splines(full)

        w_energies = np.zeros((full.shape[0], len(database.structures)))
        t_energies = np.zeros(len(database.structures))

        fcs_fitnesses = np.zeros((full.shape[0], len(database.structures)))

        # TODO: reference energy needs to be sent from master
        # ref_energy = 0

        keys = sorted(list(database.structures.keys()))
        for j, name in enumerate(keys):

            w = database.structures[name]

            w_energies[:, j] = w.compute_energy(full, potential_template.u_ranges)
            t_energies[j] = database.true_energies[name]

            # if name == database.reference_struct:
            #     ref_energy = w_energies[j]

            w_fcs = w.compute_forces(full, potential_template.u_ranges)
            true_fcs = database.true_forces[name]

            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10)) ** 2
            fcs_err = np.linalg.norm(fcs_err, axis=(1, 2)) / np.sqrt(10)

            fcs_fitnesses[:, j] = fcs_err

        # w_energies -= ref_energy
        # t_energies -= database.reference_energy

        eng_fitnesses = np.zeros((full.shape[0], len(database.structures)))

        for j, (w_eng, t_eng) in enumerate(zip(w_energies.T, t_energies)):
            eng_fitnesses[:, j] = (w_eng - t_eng) ** 2

        fitnesses = np.concatenate([eng_fitnesses, fcs_fitnesses])

        # print("Fitness shape:", fitnesses.shape, flush=True)

        return eng_fitnesses, fcs_fitnesses

    def grad(pot):
        full = np.atleast_2d(pot)
        full = potential_template.insert_active_splines(full)

        # stacking will occur along the database dimension, which works best
        # when this is the first or last dimension

        fcs_grad_vec = np.zeros(
            (full.shape[0], len(potential_template.pvec),
            len(database.structures))
        )

        w_energies = np.zeros((full.shape[0], len(database.structures)))
        t_energies = np.zeros(len(database.structures))

        # ref_energy = 0

        keys = sorted(list(database.structures.keys()))
        for j, name in enumerate(keys):
            w = database.structures[name]

            w_energies[:, j] = w.compute_energy(full, potential_template.u_ranges)
            t_energies[j] = database.true_energies[name]

            # if name == database.reference_struct:
            #     ref_energy = w_energies[j]

            w_fcs = w.compute_forces(full, potential_template.u_ranges)
            true_fcs = database.true_forces[name]

            fcs_err = ((w_fcs - true_fcs) / np.sqrt(10))

            fcs_grad = w.forces_gradient_wrt_pvec(full, potential_template.u_ranges)

            scaled = np.einsum('pna,pnak->pnak', fcs_err, fcs_grad)
            summed = scaled.sum(axis=1).sum(axis=1)

            fcs_grad_vec[:, :, j] += (2 * summed / 10)#.ravel()

        # w_energies -= ref_energy
        # t_energies -= database.reference_energy

        eng_grad_vec = np.zeros(
            (full.shape[0], len(potential_template.pvec),
            len(database.structures))
        )

        for j, (name, w_eng, t_eng) in enumerate(
            zip(keys, w_energies.T, t_energies)):

            w = database.structures[name]

            eng_err = (w_eng - t_eng)
            eng_grad = w.energy_gradient_wrt_pvec(full, potential_template.u_ranges)

            eng_grad_vec[:, :, j] += (eng_err[:, np.newaxis] * eng_grad * 2)
            # eng_grad_vec[:, j, :] += np.einsum('ik,j->ijk', eng_grad, eng_err)*2

        tmp_eng = eng_grad_vec[:, np.where(potential_template.active_mask)[0],:]
        tmp_fcs = fcs_grad_vec[:, np.where(potential_template.active_mask)[0],:]

        return tmp_eng, tmp_fcs

    return fxn, grad


def compute_relative_weights(database):
    work_weights = []

    name_list = sorted(list(database.structures.keys()))

    for name in name_list:
        work_weights.append(database.structures[name].natoms)

    work_weights = np.array(work_weights)
    work_weights = work_weights / np.min(work_weights)
    work_weights = work_weights*work_weights # cost assumed to scale as N^2

    return work_weights, name_list

def group_database_subsets(database, num_managers):
    """Groups workers based on evaluation time to help with load balancing.

    Returns:
        distributed_work (list): the partitioned work
        num_managers (int): desired number of managers to be used
    """
    # TODO: record scaling on first run, then redistribute to load balance
    # TODO: managers should just split Database; determine cost later

    # Number of managers should not exceed the number of structures
    num_managers = min(num_managers, len(database.structures))

    unassigned_structs = sorted(list(database.structures.keys()))

    work_weights, name_list = compute_relative_weights(database)

    work_per_proc = np.sum(work_weights)# / num_managers

    # work_weights = work_weights.tolist()

    # Splitting code taken from SO: https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
    cum_arr = np.cumsum(work_weights) / np.sum(work_weights)

    idx = np.searchsorted(cum_arr, np.linspace(0, 1, num_managers, endpoint=False)[1:])
    name_chunks = np.split(unassigned_structs, idx)
    weight_chunks = np.split(work_weights, idx)

    subsets = []
    grouped_work = []

    # for _ in range(num_managers):
    for name_chunk, work_chunk in zip(name_chunks, weight_chunks):
        cumulated_work = 0

        names = []

        # while unassigned_structs and (cumulated_work < work_per_proc):
        for n,w in zip(name_chunk, work_chunk):
            names.append(n)
            cumulated_work += w
            # names.append(unassigned_structs.pop())
            # cumulated_work += work_weights.pop()

        mini_database = Database.manual_init(
            {name: database.structures[name] for name in names},
            {name: database.true_energies[name] for name in names},
            {name: database.true_forces[name] for name in names},
            {name: database.weights[name] for name in names},
            database.reference_struct,
            database.reference_energy
        )

        subsets.append(mini_database)
        grouped_work.append(cumulated_work)

    return subsets, grouped_work


def compute_procs_per_subset(struct_natoms, total_num_procs, method='natoms'):
    """
    Should split the number of processors into groups based on the relative
    evaluation costs of each Database subset.

    Possible methods for computing weights:
        'natoms': number of atoms in cell
        'time': running tests to see how long each one takes

    Args:
        struct_natoms (list[int]): number of atoms in each structure
        total_num_procs (int): total number of processors available
        method (str): 'natoms' or 'time' (default is 'natoms')

    Returns:
        ranks_per_struct (list[list]): processor ranks for each struct
    """

    structures = list(struct_natoms)
    num_structs = len(struct_natoms)

    # Note: ideally, should have at least as many cores as struct   s
    supported_methods = ['natoms', 'time']

    # natoms = [worker.natoms for worker in struct_natoms]

    # Error checking on  desired method
    if method == 'natoms':
        weights = [n*n for n in struct_natoms]
        weights /= np.sum(weights)
    elif method == 'time':
        raise NotImplementedError("Oops ...")
    else:
        raise ValueError(
            "Invalid method. Must be either", ' or '.join(supported_methods)
        )

    # TODO: possible problem where too many large structs get passed to one node
    # TODO: dimer got more workers than a trimer

    # Sort weights and structs s.t. least expensive is last
    sort = np.argsort(weights)[::-1]

    weights = weights[sort]
    struct_natoms = [s for _, s in sorted(zip(sort, struct_natoms))]

    # Compute how many processors would "ideally" be needed for each struct
    num_procs_needed_per_struct = weights * total_num_procs

    # Every struct gets at least once processor
    num_procs_needed_per_struct = np.clip(num_procs_needed_per_struct, 1, None)
    num_procs_needed_per_struct = np.ceil(num_procs_needed_per_struct)

    # if too many procs assigned
    if np.sum(num_procs_needed_per_struct) > total_num_procs:
        while np.sum(num_procs_needed_per_struct) != total_num_procs:
            max_idx = np.argmax(num_procs_needed_per_struct)
            num_procs_needed_per_struct[max_idx] -= 1
    else: # not enough procs assigned
        while np.sum(num_procs_needed_per_struct) != total_num_procs:
            min_idx = np.argmin(num_procs_needed_per_struct)
            num_procs_needed_per_struct[min_idx] += 1

    # Create list of indices for np.split()
    num_procs_needed_per_struct = num_procs_needed_per_struct.astype(int)
    split_indices = np.cumsum(num_procs_needed_per_struct)

    return np.split(np.arange(total_num_procs), split_indices)[:-1]
