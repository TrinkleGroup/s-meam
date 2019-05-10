import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
import src.partools as partools


################################################################################

def sgd(parameters, database, template, is_manager, manager,
        manager_comm):

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
        # TODO: prepare logfiles (cost, ni_trace)

        all_struct_names = [s.encode('utf-8').strip().decode('utf-8') for s in
                            database.unique_structs]

        struct_natoms = database.unique_natoms
        num_structs = len(all_struct_names)

        print(all_struct_names)

        old_copy_names = list(all_struct_names)

        worker_ranks = partools.compute_procs_per_subset(
            struct_natoms, world_size
        )

        print("worker_ranks:", worker_ranks)
    else:
        database = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

    # TODO: does everyone need all_struct_names?

    num_structs = world_comm.bcast(num_structs, root=0)

    fxn_wrap, grad_wrap = partools.build_evaluation_functions(
        template, database, all_struct_names, manager,
        is_master, is_manager, manager_comm, "Ti48Mo80_type1_c18"
    )

    # build starting population that will be equilibrated

    # TODO: print the center of the U[] for each to see when stable

    if is_master:
        potential = np.atleast_2d(template.generate_random_instance())
        potential = np.atleast_2d(
                [template.generate_random_instance() for _ in
                    range(parameters['POP_SIZE'])
                    ]
                )
        print('potential.shape:', potential.shape)
        potential = potential[:, np.where(template.active_mask)[0]]
        # potential = np.ones(potential.shape)

        ud = np.concatenate(template.u_ranges)
        u_domains = np.atleast_2d(np.tile(ud, (potential.shape[0], 1)))

        weights = np.ones(len(database.entries))
    else:
        potential = None
        u_domains = None
        weights = None

    potential = world_comm.bcast(potential, root=0)
    u_domains = world_comm.bcast(u_domains, root=0)
    weights = world_comm.bcast(weights, root=0)

    init_cost, max_ni, min_ni, avg_ni = fxn_wrap(
        np.hstack([potential, u_domains]), weights, return_ni=True, penalty=True
    )

    # perform initial rescaling
    if is_master:
        potential = partools.rescale_ni(potential, min_ni, max_ni, template)

    costs, max_ni, min_ni, avg_ni = fxn_wrap(
        np.hstack([potential, u_domains]), weights, return_ni=True, penalty=True
    )

    if is_master:
        costs = np.sum(costs, axis=1)

        partools.checkpoint(
            potential, costs, max_ni, min_ni, avg_ni, 0, parameters, template
        )

    num_steps_taken = 0

    cost = fxn_wrap(np.hstack([potential, u_domains]), weights,)

    if is_master:
        print("{} {}".format(num_steps_taken, np.sum(cost, axis=1)), flush=True)

    eta = parameters['SGD_STEP_SIZE']
    while (num_steps_taken < parameters['SGD_NSTEPS']):
        """
        choose random subset of structures
        compute gradient of subset
        update potential
        """
        if is_master:
            # choose random subset of structures

            # print(np.random.choice(database.entries, max([1,parameters['SGD_BATCH_SIZE']])))
            choices = np.random.choice(
                        len(database.entries),
                        max([1, parameters['SGD_BATCH_SIZE']])
                    )

            entries_batch = []
            for k in choices:
                entries_batch.append(database.entries[k])

            batch_ids = [e.struct_name for e in entries_batch]
        else:
            batch_ids = None

        batch_ids = world_comm.bcast(batch_ids, root=0)

        # TODO: problem; how do you compare inputs for things like C_ij?

        gradient = np.zeros(
                (potential.shape[0], template.pvec_len, parameters['SGD_BATCH_SIZE'])
                # (template.pvec_len, parameters['SGD_BATCH_SIZE'])
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
                    np.hstack([potential, u_domains])
                )

            fcs = manager.compute_forces(np.hstack([potential, u_domains]))

            eng_grad = manager.compute_energy_grad(np.hstack([potential, u_domains]))
            fcs_grad = manager.compute_forces_grad(np.hstack([potential, u_domains]))
        else:  # TODO: avoid sending these empty messages
            eng = np.zeros(potential.shape[0])
            fcs = np.zeros(potential.shape[0])

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
                    summed = scaled.sum(axis=1).sum(axis=1)#.ravel()

                    gradient[:, :, cost_id] += (2*summed/10)*weights[s_id]
                    # gradient[:, cost_id] += (2*summed/10)*weights[s_id]

                elif entry.type == 'energy':
                    r_name = entry.ref_struct
                    true_ediff = entry.value

                    # find index of structures to know which energies to use
                    s_id = all_struct_names.index(entry.struct_name)
                    r_id = all_struct_names.index(r_name)

                    comp_ediff = all_eng[s_id, :] - all_eng[r_id, :]

                    eng_err = comp_ediff - true_ediff
                    s_grad = mgr_eng_grad[s_id]#.ravel()
                    r_grad = mgr_eng_grad[r_id]#.ravel()

                    tmp = (eng_err[:, np.newaxis]*(s_grad - r_grad)*2)*weights[s_id]
                    gradient[:, :, cost_id] += tmp#.ravel()
                    # gradient[:, cost_id] += tmp.ravel()

            # end gradient calculations

            tmp = np.atleast_2d(np.average(gradient, axis=2))
            potential -= eta*tmp[:, np.where(template.active_mask)[0]]

            eta *= 0.95

        cost = fxn_wrap(np.hstack([potential, u_domains]), weights,)

        if is_master:
            print("{} {} {}".format(num_steps_taken, eta, np.sum(cost, axis=1)), flush=True)

        num_steps_taken += 1

        digits = np.floor(np.log10(parameters['SGD_NSTEPS']))

        format_str = os.path.join(
            parameters['SAVE_DIRECTORY'],
            'pop_{0:0' + str(int(digits) + 1)+ 'd}.dat'
        )

        np.savetxt(format_str.format(num_steps_taken), potential)

################################################################################
