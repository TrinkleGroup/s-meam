import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
import src.partools as partools

np.set_printoptions(precision=3)


################################################################################

def sgd(parameters, database, template, is_manager, manager, manager_comm):

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()

    is_master = (world_rank == 0)

    if is_master:
        # TODO: prepare logfiles (cost, ni_trace)

        all_struct_names = [s.encode('utf-8').strip().decode('utf-8') for s in
                            database.unique_structs]

        if is_master:
            original_mask = template.active_mask.copy()
    else:
        all_struct_names = None

    # TODO: does everyone need all_struct_names?

    fxn_wrap, grad_wrap = partools.build_evaluation_functions(
        template, database, all_struct_names, manager,
        is_master, is_manager, manager_comm, "Ti48Mo80_type1_c18"
    )

    # build starting population that will be equilibrated
    if is_master:
        master_pot = np.atleast_2d([
            template.generate_random_instance()
            for _ in range(parameters['POP_SIZE'])
        ])

        working_pot = master_pot[:, np.where(template.active_mask)[0]]

        weights = np.ones(len(database.entries))
    else:
        weights = None
        master_pot = np.empty((parameters['POP_SIZE'], template.pvec_len))
        working_pot = np.zeros(
            (parameters['POP_SIZE'], len(np.where(template.active_mask)[0]))
        )

    weights = world_comm.bcast(weights, root=0)

    init_cost, max_ni, min_ni, avg_ni = fxn_wrap(
        master_pot, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    # perform initial rescaling
    if is_master:
        print('init min/max ni', min_ni[0], max_ni[0])

        master_pot = partools.rescale_ni(master_pot, min_ni, max_ni, template)

    costs, max_ni, min_ni, avg_ni = fxn_wrap(
        potential, weights, return_ni=True
    )

    if is_master:
        costs = np.sum(costs, axis=1)

        partools.checkpoint(
            potential, costs, max_ni, min_ni, avg_ni, 0, parameters, template,
            parameters['SGD_NSTEPS']
        )

    num_steps_taken = 0

    T = 1

    # potential = partools.mcmc(
    #     potential, weights, fxn_wrap, template, T, parameters,
    #     np.arange(12), is_master, num_steps_taken, suffix="warm-up",
    #     max_nsteps=parameters['MCMC_BLOCK_SIZE'], penalty=True
    # )

    cost = fxn_wrap(potential, weights,)

    eta = parameters['SGD_STEP_SIZE']

    if is_master:
        current_cost = np.sum(cost, axis=1)

        print(
            "{} {} {} {} {}".format(
                num_steps_taken, eta, np.min(current_cost),
                np.max(current_cost), np.average(current_cost),
            ),
            flush=True
        )

    while (num_steps_taken < parameters['SGD_NSTEPS']):
        if is_master:

            # choose random subset of structures
            choices = np.random.choice(
                        len(database.entries),
                        max([1, parameters['SGD_BATCH_SIZE']])
                    )

            entries_batch = []
            for k in choices:
                entries_batch.append(database.entries[k])

            batch_ids = [e.struct_name for e in entries_batch]


            # rescale by ni if desired
            if parameters['DO_RESCALE'] and \
                    ((num_steps_taken + 1) % parameters['RESCALE_FREQ'] == 0):
                if is_master:
                    if (num_steps_taken + 1) < parameters['RESCALE_STOP_STEP']:
                        print("Rescaling ...")

                        potential = partools.rescale_ni(
                            potential, min_ni, max_ni, template
                        )

        costs, max_ni, min_ni, avg_ni = fxn_wrap(
            potential, weights, return_ni=True
        )

        if is_master:
            # shift U domains if desired
            if parameters['DO_SHIFT'] and ((num_steps_taken + 1) %
                    parameters['SHIFT_FREQ'] == 0):

                current_costs = np.sum(cost, axis=1)

                min_ni = min_ni[np.argsort(current_costs)]
                max_ni = max_ni[np.argsort(current_costs)]

                new_u_domains = partools.shift_u(min_ni, max_ni)

                print("New U domains:", new_u_domains)

                template.u_ranges = new_u_domains
        else:
            batch_ids = None

        batch_ids = world_comm.bcast(batch_ids, root=0)

        gradient = np.zeros(
                (potential.shape[0], template.pvec_len, parameters['SGD_BATCH_SIZE'])
            )

        if is_manager:
            potential = np.atleast_2d(potential)
            potential = manager_comm.bcast(potential, root=0)

        if is_master:
            ref_name = database.ref_name
        else:
            ref_name = None

        ref_name = world_comm.bcast(ref_name)

        # TODO: use a generator here
        # compute batch gradient
        if manager.struct_name in batch_ids + [ref_name]:

            # managers gather from workers
            eng, _, _, _, _, _ = manager.compute_energy(
                    potential
                )

            fcs = manager.compute_forces(potential)

            eng_grad = manager.compute_energy_grad(potential)
            fcs_grad = manager.compute_forces_grad(potential)
        else:  # TODO: avoid sending these empty messages
            eng = np.zeros(potential.shape[0])
            fcs = np.zeros(potential.shape[0])

            min_ni = np.zeros((potential.shape[0], template.ntypes))
            max_ni = np.zeros((potential.shape[0], template.ntypes))

            eng_grad = np.zeros(potential.shape)
            fcs_grad = np.zeros(potential.shape)

        # master gathers from managers
        if is_manager:
            mgr_eng = manager_comm.gather(eng, root=0)
            mgr_fcs = manager_comm.gather(fcs, root=0)

            mgr_eng_grad = manager_comm.gather(eng_grad, root=0)
            mgr_fcs_grad = manager_comm.gather(fcs_grad, root=0)

        if is_master:

            all_eng = np.vstack(mgr_eng)
            all_fcs = mgr_fcs

            for cost_id, entry in enumerate(entries_batch):
                # TODO: have managers compute forces difference before send

                if entry.type == 'forces':

                    s_id = all_struct_names.index(entry.struct_name)

                    w_fcs = all_fcs[s_id]
                    true_fcs = entry.value

                    diff = w_fcs - true_fcs

                    # zero out interactions outside of range of O atom
                    fcs_grad = mgr_fcs_grad[s_id]

                    scaled = np.einsum('pna,pnak->pnak', diff, fcs_grad)
                    summed = scaled.sum(axis=1).sum(axis=1)

                    gradient[:, :, cost_id] += (2*summed/10)*weights[s_id]

                elif entry.type == 'energy':
                    r_name = entry.ref_struct
                    true_ediff = entry.value

                    # find index of structures to know which energies to use
                    s_id = all_struct_names.index(entry.struct_name)
                    r_id = all_struct_names.index(r_name)

                    comp_ediff = all_eng[s_id, :] - all_eng[r_id, :]

                    eng_err = comp_ediff - true_ediff
                    s_grad = mgr_eng_grad[s_id]
                    r_grad = mgr_eng_grad[r_id]

                    tmp = (eng_err[:, np.newaxis]*(s_grad - r_grad)*2)*weights[s_id]
                    gradient[:, :, cost_id] += tmp

            # end gradient calculations

            tmp = np.atleast_2d(np.average(gradient, axis=2))
            potential -= eta*tmp[:, np.where(template.active_mask)[0]]

            # TODO: add better adaptive learning rates
            # eta *= 0.75

        cost, max_ni, min_ni, avg_ni = fxn_wrap(
            potential, weights, return_ni=True,
        )

        if is_master:
            current_cost = np.sum(cost, axis=1)

            print(
                "{} {} {} {} {}".format(
                    num_steps_taken, eta, np.min(current_cost),
                    np.max(current_cost), np.average(current_cost),
                ),
                flush=True
            )

            partools.checkpoint(
                potential, current_cost, max_ni, min_ni, avg_ni,
                num_steps_taken, parameters, template, parameters['SGD_NSTEPS']
            )

        num_steps_taken += 1

    if parameters['DO_LMIN']:
        if is_master:
            print("Performing Levenberg-Marquardt ...", flush=True)

            potential = potential[np.argsort(current_cost)]
            subset = potential[:10]

        subset = partools.local_minimization(
            subset, template.u_ranges, fxn_wrap, grad_wrap, weights, world_comm,
            is_master, nsteps=parameters['LMIN_NSTEPS'], lm_output=True,
        )

        if is_master:
            potential[:10] = subset

    cost, max_ni, min_ni, avg_ni = fxn_wrap(
        potential, weights, return_ni=True
    )

    if is_master:
        current_cost = np.sum(cost, axis=1)

        print(
            "Final: {} {} {} {} {}".format(
                num_steps_taken, eta, np.average(current_cost),
                np.min(current_cost), np.max(current_cost),
            ),
            flush=True
        )

        partools.checkpoint(
            potential, current_cost, max_ni, min_ni, avg_ni,
            num_steps_taken, parameters, template, parameters['SGD_NSTEPS']
        )

################################################################################
