import os
import cma, comocma
import time
import pickle
import random
import itertools
import numpy as np
from mpi4py import MPI

import src.partools


def COMO_CMAES(parameters, template, node_manager, manager_comm, cost_fxn):
    # MPI setup
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
        print()
        print('='*80)
        print()
        for k,v in parameters.items():
            print(k, v)
        print()
        print('='*80)
        print()

    template = world_comm.bcast(template, root=0)

    # have to set a "weight" because NodeManager was meant to change weights
    node_manager.weights = dict(zip(
        node_manager.loaded_structures,
        np.ones(len(node_manager.loaded_structures))
    ))

    # figure out what structures exist on the node managers
    if node_manager.is_node_master:
        all_struct_names = collect_structure_names(
            node_manager, manager_comm, is_master
        )

        weights = np.ones(len(all_struct_names))
    else:
        all_struct_names = None
        weights = None

    weights = node_manager.comm.bcast(weights, root=0)

    # gather the true values of each objective target
    true_values = prepare_true_values(node_manager, world_comm, is_master)

    # run the function constructor to build the objective function
    objective_fxn, gradient = src.partools.build_evaluation_functions(
        template, all_struct_names, node_manager,
        world_comm, is_master, true_values, parameters,
        manager_comm=manager_comm
    )

    if is_master:
        num_solvers = parameters['NUM_SOLVERS']
        full_solution = template.pvec.copy()
        full_solution = np.tile(full_solution, (num_solvers, 1))

        active_ind = np.where(template.active_mask)[0]

        for ii in range(num_solvers):
            full_solution[ii, active_ind] = template.generate_random_instance()[active_ind]

        solutions = full_solution[:, active_ind].copy()

        print('solutions.shape:', solutions.shape)
    else:
        solutions = None

    errors, max_ni, min_ni, avg_ni = objective_fxn(
        template.insert_active_splines(np.atleast_2d(solutions)), weights,
        return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Initial min/max ni:", min_ni[0], max_ni[0])

        errors[:, 0:-4:3] *= parameters['ENERGY_WEIGHT']
        errors[:, 1:-4:3] *= parameters['FORCES_WEIGHT']
        errors[:, 2:-4:3] *= parameters['STRESS_WEIGHT']

        solutions = full_solution[:, active_ind]

    errors, max_ni, min_ni, avg_ni = objective_fxn(
        template.insert_active_splines(np.atleast_2d(solutions)), weights,
        return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Rescaled initial min/max ni:", min_ni[0], max_ni[0])

        errors[:, 0:-4:3] *= parameters['ENERGY_WEIGHT']
        errors[:, 1:-4:3] *= parameters['FORCES_WEIGHT']
        errors[:, 2:-4:3] *= parameters['STRESS_WEIGHT']


    solutions = world_comm.bcast(solutions, root=0)

    if is_master:

        solvers = [
            comocma.como.CmaKernel(
                sol,
                parameters['CMAES_STEP_SIZE'],
                {
                    'popsize': parameters['POP_SIZE'],
                    'verb_disp': 1,
                }
            )

            for sol in solutions
        ]

        # Build the indices for identifying surface structures

        group_conditions = [
                lambda n: 1 if 'strain' in n else 0,
                lambda n: 1 if 'surface' in n else 0,
                lambda n: 1 if 'Vacancy' in n else 0,
                lambda n: 1 if ('6000_K' in n) and ('Vacancy' not in n) else 0,
        ]

        group_indices = []

        for cond in group_conditions:
            indices = np.array([
                # 1 if group_name in n else 0 for n in all_struct_names
                cond(n) for n in all_struct_names
            ])

            indices = np.where(indices)[0].tolist()

            group_indices.append(indices)


        everything_else = np.arange(len(all_struct_names)).tolist()

        del_indices = []
        for group in group_indices:
            del_indices += group

        del_indices = sorted(del_indices)

        for ind in del_indices[::-1]:
            del everything_else[ind]

        group_indices.append(everything_else)

        first_costs = cost_fxn(
            errors, sum_all=False, return_penalty=False,
            group_indices=group_indices,
        )

        reference_point = np.max(first_costs, axis=0)*10
        # reference_point = [1]*2*len(group_indices)

        ideal_hypervolume = np.prod(reference_point)

        # set the ni penalty to be extremely large (relative to the ideal cost)
        # to ensure that the ni sampling is optimized early

        parameters['NI_PENALTY'] = ideal_hypervolume*0.5

        print('Reference point:', reference_point)
        print('Ideal hyper-volume:', np.prod(reference_point), flush=True)

        moes = comocma.Sofomore(
            solvers,
            reference_point=reference_point,
            opts={
                'verb_disp': 1
            },
            threads_per_node=parameters['PROCS_PER_PHYS_NODE']
        )

        cma_start_time = time.time()
    else:
        population = None

    time_to_stop = False

    grow_id = 0
    generation_number = 1
    while (not time_to_stop) and (generation_number <= parameters['NSTEPS']):
        if is_master:

            population = np.array(moes.ask('all'))
            population = template.insert_active_splines(population)

        errors, max_ni, min_ni, avg_ni = objective_fxn(
            population, weights, return_ni=True,
            penalty=parameters['PENALTY_ON']
        )

        if is_master:

            # TODO: shouldn't logging go _after_ updating moes?
            if (generation_number % parameters['CHECKPOINT_FREQ'] == 0) and (
                    generation_number > 1):

                digits = np.floor(np.log10(parameters['NSTEPS']))

                format_str = os.path.join(
                    parameters['SAVE_DIRECTORY'],
                    'archive_{0:0' + str(int(digits) + 1)+ 'd}.pkl'
                ).format(generation_number)


                pickle.dump(moes.archive, open(format_str, 'wb'))

                # # TODO: edit CmaKernel to save solutions of pareto_front_cut
                # format_str = os.path.join(
                #     parameters['SAVE_DIRECTORY'],
                #     'front_{0:0' + str(int(digits) + 1)+ 'd}.pkl'
                # ).format(generation_number)

                # pickle.dump(moes.pareto_front_cut, open(format_str, 'wb'))

                # src.partools.checkpoint(
                #     population, errors, max_ni, min_ni, avg_ni,
                #     generation_number, parameters, template,
                #     parameters['NSTEPS']
                # )

            org_errors = errors.copy()
            # only apply weights AFTER logging unweighted data
            errors[:, 0:-4:3] *= parameters['ENERGY_WEIGHT']
            errors[:, 1:-4:3] *= parameters['FORCES_WEIGHT']
            errors[:, 2:-4:3] *= parameters['STRESS_WEIGHT']

            new_costs, penalty_costs = cost_fxn(
                errors, sum_all=False, return_penalty=True,
                group_indices=group_indices,
            )

            moes.tell(
                population[:, active_ind], new_costs, penalties=penalty_costs
            )

            # ref_point_check = np.max(np.atleast_2d(moes.archive), axis=0)

            # # dynamically update reference point
            # # this might cause weirdness?
            # if np.any(ref_point_check < np.array(moes.reference_point)/10):
            #     new_ref_point = [
            #         ref_point_check[ri]*1.5
            #         if ref_point_check[ri] < moes.reference_point[ri]
            #         else moes.reference_point[ri]
            #         for ri in range(len(moes.reference_point))
            #     ]

            #     print(
            #         "Updating reference point:\n\t",
            #         moes.reference_point,
            #         " -> ",
            #         new_ref_point,
            #         flush=True
            #     )

            #     moes.reference_point = new_ref_point

            moes.disp()

            # seems like the front is kicking things out 

            summed = new_costs.sum(axis=0)

            stopping_info = moes.stop()

            if stopping_info:
                time_to_stop = True

        if is_master:

            min_idx = np.argmin(new_costs, axis=0)
            best_fit = org_errors[min_idx]
            best = population[min_idx]
            best = np.atleast_2d(best)

            # log full cost vector of best ever potential
            with open(parameters['BEST_FIT_FILE'], 'ab') as cost_save_file:
                np.savetxt(cost_save_file, np.atleast_2d(best_fit))

            # log best ever potential
            with open(parameters['BEST_POT_FILE'], 'ab') as pot_save_file:
                np.savetxt(pot_save_file, best)

        generation_number += 1
        time_to_stop = world_comm.bcast(time_to_stop, root=0)

    # end CMA-ES loop

    if is_master:
        pickle.dump(
            moes,
            open(os.path.join(parameters['SAVE_DIRECTORY'], 'moes.pkl'), 'wb')
        )

        print('Saved MOES object as moes.pkl')

        cma_runtime = time.time() - cma_start_time

        polish_start_time = time.time()
    else:
        best = np.empty((1, template.pvec_len))

    # best = src.partools.local_minimization(
    #     best[:, np.where(template.active_mask)[0]], best, template, objective_fxn,
    #     gradient, weights, world_comm, is_master, nsteps=10, lm_output=True,
    #     penalty=parameters['PENALTY_ON']
    # )

    # if is_master:
        # sorted_pop[0, np.where(template.active_mask)[0]] = best
        # sorted_pop[0, active_ind] = best
        # sorted_pop[0] = best
        # population = sorted_pop

    errors, max_ni, min_ni, avg_ni = objective_fxn(
        population, weights, return_ni=True,
        penalty=parameters['PENALTY_ON']
    )

    if is_master:
        polish_runtime = time.time() - polish_start_time

        errors[:, 0:-4:3] *= parameters['ENERGY_WEIGHT']
        errors[:, 1:-4:3] *= parameters['FORCES_WEIGHT']
        errors[:, 2:-4:3] *= parameters['STRESS_WEIGHT']


        final_costs = cost_fxn(
            errors, sum_all=False, return_penalty=False,
            group_indices=group_indices,
        )

        # src.partools.checkpoint(
        #     # population, final_costs, tmp_max_ni, tmp_min_ni, tmp_avg_ni,
        #     population, errors, max_ni, min_ni, avg_ni,
        #     generation_number, parameters, template,
        #     parameters['NSTEPS']
        # )

        print()
        print("Final best cost = ", final_costs[0])

        print()
        print(
            "CMA-ES runtime = {:.2f} (s)".format(cma_runtime), flush=True
        )

        print(
            "CMA-ES average time per step = {:.2f}"
            " (s)".format(cma_runtime / parameters['NSTEPS']), flush=True
        )

        print()
        print(
            "LM runtime = {:.2f} (s)".format(polish_runtime), flush=True
        )



def prepare_true_values(node_manager, world_comm, is_master):
    local_true_eng = {}
    local_true_ref = {}

    for key in node_manager.loaded_structures:
        local_true_eng[key] = node_manager.get_true_value('energy', key)
        local_true_ref[key] = node_manager.get_true_value('ref_struct', key)

    local_true_values = {
            'energy': local_true_eng,
            'ref_struct': local_true_ref,
        }

    all_true_values = world_comm.gather(local_true_values, root=0)

    if is_master:
        all_eng = {}
        # all_fcs = {}
        all_ref = {}
        ref_names = {}

        for d in all_true_values:
            all_eng.update(d['energy'])
            # all_fcs.update(d['forces'])
            all_ref.update(d['ref_struct'])

        true_values = {
                'energy': all_eng,
                # 'forces': all_fcs,
                'ref_struct': all_ref,
            }

    else:
        true_values = None


    return true_values


def collect_structure_names(node_manager, world_comm, is_master):
    local_structs = world_comm.gather(node_manager.loaded_structures, root=0)

    if is_master:
        all_struct_names = []
        for slist in local_structs:
            all_struct_names += slist

    else:
        all_struct_names = None

    all_struct_names = world_comm.bcast(all_struct_names, root=0)

    return all_struct_names
