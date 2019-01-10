import os
import numpy as np
from src.database import Database
import src.meam
from src.meam import MEAM
from src.potential_templates import Template

def build_evaluation_functions(
    potential_template, master_database, all_struct_names, manager, is_master,
    is_manager, manager_comm, flatten=False
    ):
    """Builds the function to evaluate populations. Wrapped here for readability
    of main code."""

    if is_master:
        num_structs = len(all_struct_names)

    def fxn_wrap(master_pop, weights):
        """Master: returns all potentials for all structures.

        NOTE: the order of 'weights' should match the order of the database
        entries

        """
        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
            pop = np.atleast_2d(pop)
        else:
            pop = None

        eng = manager.compute_energy(pop)
        fcs = manager.compute_forces(pop)

        fitnesses = 0

        if is_manager:
            mgr_eng = manager_comm.gather(eng, root=0)
            mgr_fcs = manager_comm.gather(fcs, root=0)

            if is_master:
                # note: can't stack mgr_fcs b/c different dimensions per struct
                all_eng = np.vstack(mgr_eng)
                all_fcs = mgr_fcs

                # fcs_fitnesses = np.zeros((len(pop), num_structs))
                # 
                # eng_fitnesses = np.zeros(
                #     (len(pop), len(master_database.reference_structs))
                # )

                fitnesses = np.zeros((len(pop), len(master_database.entries)))

                # for name in master_database.true_forces.keys():
                for fit_id, (entry, weight) in enumerate(
                        zip(master_database.entries, weights)):

                    name = entry.struct_name
                    s_id = all_struct_names.index(name)

                    if entry.type == 'forces':

                        # if "oct.Ti" == name:
                        #     weight = 0.0682
                        # elif "hex.Ti" == name:
                        #     weight = 0.03
                        # elif "crowd.Ti" == name:
                        #     weight = 0.00362
                        # elif "oh.Ti" == name:
                        #     weight = 0.0460
                        # elif "oc.Ti" == name:
                        #     weight = 0.07665
                        # elif "hc.Ti" == name:
                        #     weight = 0.0372
                        # elif "oo.Ti" == name:
                        #     weight = 0.0395


                        w_fcs = all_fcs[s_id]
                        # true_fcs = master_database.true_forces[name]
                        true_fcs = entry.value

                        diff = w_fcs - true_fcs

                        # zero out interactions outside of range of O atom
                        force_weights = master_database.force_weighting[name]
                        diff *= force_weights[:, np.newaxis]

                        epsilon = np.linalg.norm(diff, 'fro', axis=(1, 2)) / np.sqrt(10)
                        # fcs_fitnesses[:, s_id] = epsilon * epsilon * weight
                        fitnesses[:, fit_id] = epsilon * epsilon * weight

                    elif entry.type == 'energy':

                # for fit_id, (s_name, ref) in enumerate(
                #         master_database.reference_structs.items()):
                        r_name = entry.ref_struct
                        true_ediff = entry.value

                        # find index of structures to know which energies to use
                        # s_id = all_struct_names.index(s_name)
                        r_id = all_struct_names.index(r_name)

                        comp_ediff = all_eng[s_id, :] - all_eng[r_id, :]

                        tmp = (comp_ediff - true_ediff) ** 2
                        # eng_fitnesses[:, fit_id] = tmp * ref.weight
                        fitnesses[:, fit_id] = tmp * weight

                    # print(s_name, "-", r_name, ref.weight,
                    #         logistic(tmp*ref.weight, 'eng'))

                # fitnesses = np.hstack([eng_fitnesses, fcs_fitnesses])

                print(np.sum(fitnesses, axis=1), flush=True)

        # if return_ni: return fitnesses, per_u_max_ni
        # else: return fitnesses
        return fitnesses


    def grad_wrap(master_pop, weights):
        """Evalautes the gradient for all potentials in the population"""

        if is_manager:
            pop = manager_comm.bcast(master_pop, root=0)
        else:
            pop = None

        eng = manager.compute_energy(pop)
        fcs = manager.compute_forces(pop)

        eng_grad = manager.compute_energy_grad(pop)
        fcs_grad = manager.compute_forces_grad(pop)


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

                # fcs_grad_vec = np.zeros(
                #     (len(pop), potential_template.pvec_len, num_structs)
                # )
                # 
                # eng_grad_vec = np.zeros(
                #     (len(pop), potential_template.pvec_len,
                #      len(master_database.reference_structs))
                # )

                gradient = np.zeros((
                    len(pop), potential_template.pvec_len,
                    len(master_database.entries)
                ))

                ref_energy = 0

                # for s_id, name in enumerate(all_struct_names):
                # for name in master_database.true_forces.keys():

                for fit_id, (entry, weight) in enumerate(
                        zip(master_database.entries, weights)):

                    name = entry.struct_name

                    if entry.type == 'forces':

                    # if "oct.Ti" == name:
                    #     weight = 0.0682
                    # elif "hex.Ti" == name:
                    #     weight = 0.03
                    # elif "crowd.Ti" == name:
                    #     weight = 0.00362
                    # elif "oh.Ti" == name:
                    #     weight = 0.0460
                    # elif "oc.Ti" == name:
                    #     weight = 0.07665
                    # elif "hc.Ti" == name:
                    #     weight = 0.0372
                    # elif "oo.Ti" == name:
                    #     weight = 0.0395


                        s_id = all_struct_names.index(name)

                        w_fcs = all_fcs[s_id]
                        # true_fcs = master_database.true_forces[name]
                        true_fcs = entry.value

                        diff = w_fcs - true_fcs

                        # zero out interactions outside of range of O atom
                        force_weights = master_database.force_weighting[name]
                        diff *= force_weights[:, np.newaxis]

                        fcs_grad = mgr_fcs_grad[s_id]

                        scaled = np.einsum('pna,pnak->pnak', diff, fcs_grad)
                        summed = scaled.sum(axis=1).sum(axis=1)

                        # fcs_grad_vec[:, :, s_id] += (2 * summed / 10) * weight
                        gradient[:, :, fit_id] += (2 * summed / 10) * weight

                # for fit_id, (s_name, ref) in enumerate(
                #         master_database.reference_structs.items()):

                    elif entry.type == 'energy':
                        r_name = entry.ref_struct
                        true_ediff = entry.value

                        # find index of structures to know which energies to use
                        s_id = all_struct_names.index(name)
                        r_id = all_struct_names.index(r_name)

                        comp_ediff = all_eng[s_id, :] - all_eng[r_id, :]

                        eng_err = comp_ediff - true_ediff
                        s_grad = mgr_eng_grad[s_id]
                        r_grad = mgr_eng_grad[r_id]

                        # eng_grad_vec[:, :, fit_id] += (
                        #         eng_err[:, np.newaxis] * (s_grad - r_grad) * 2
                        # ) * ref.weight

                        gradient[:, :, fit_id] += (
                                eng_err[:, np.newaxis] * (s_grad - r_grad) * 2
                        ) * weight

                indices = np.where(potential_template.active_mask)[0]
                # tmp_eng = eng_grad_vec[:, indices, :]
                # tmp_fcs = fcs_grad_vec[:, indices, :]
                # 
                # gradient = np.dstack([tmp_eng, tmp_fcs]).swapaxes(1, 2)
                gradient = gradient[:, indices, :].swapaxes(1, 2)

        return gradient

    return fxn_wrap, grad_wrap


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


def initialize_potential_template(load_path):
    # TODO: BW settings
    potential = MEAM.from_file(os.path.join(load_path, 'TiO.meam.spline'))

    x_pvec, seed_pvec, indices = src.meam.splines_to_pvec(
        potential.splines)

    potential_template = Template(
        pvec_len=116,
        u_ranges=[(-55, -24), (-24, 8.3)],
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

    potential_template.pvec[22:37] = 0; mask[22:37] = 0  # phi_O
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
