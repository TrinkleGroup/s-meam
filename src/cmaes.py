import cma
import time
import random
import itertools
import numpy as np
from mpi4py import MPI
from deap import tools

import src.partools
import src.pareto


def CMAES(parameters, template, node_manager,):
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()

    is_master = (world_rank == 0)


    template = world_comm.bcast(template, root=0)

    # have to set a "weight" because NodeManager was meant to change weights
    node_manager.weights = dict(zip(
        node_manager.loaded_structures,
        np.ones(len(node_manager.loaded_structures))
    ))

    # figure out what structures exist on the node managers
    all_struct_names = collect_structure_names(
        node_manager, world_comm, is_master
    )

    # gather the true values of each objective target
    true_values = prepare_true_values(node_manager, world_comm, is_master)

    # run the function constructor to build the objective function
    objective_fxn, gradient = src.partools.build_evaluation_functions(
        template, all_struct_names, node_manager,
        world_comm, is_master, true_values, parameters
    )

    if is_master:
        full_solution = template.pvec.copy()

        active_ind = np.where(template.active_mask)[0]

        full_solution[active_ind] = template.generate_random_instance()[active_ind]
        # solution = template.generate_random_instance()[active_ind]

        solution = full_solution[active_ind].copy()

        print('solution.shape:', solution.shape)
    else:
        solution = None

    weights = np.ones(len(all_struct_names))

    costs, max_ni, min_ni, avg_ni = objective_fxn(
        template.insert_active_splines(np.atleast_2d(solution)), weights,
        return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Initial min/max ni:", min_ni[0], max_ni[0])

        full_solution = src.partools.rescale_ni(
            np.atleast_2d(full_solution), min_ni, max_ni, template
        )[0]

        solution = full_solution[active_ind]

    costs, max_ni, min_ni, avg_ni = objective_fxn(
        template.insert_active_splines(np.atleast_2d(solution)), weights,
        return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Rescaled initial min/max ni:", min_ni[0], max_ni[0])

    solution = world_comm.bcast(solution, root=0)

    if is_master:
        # opts = cma.CMAOptions()
        #
        # for key, val in opts.defaults().items():
        #     print(key, val)

        es = cma.CMAEvolutionStrategy(
            solution,
            parameters['CMAES_STEP_SIZE'],
            {
                'verb_disp': 1,
                'popsize': parameters['POP_SIZE'],
                # 'CMA_mu': 19,
                # 'CMA_rankmu': 0.1
            }
        )

        es.opts.set({'verb_disp': 1})

        cma_start_time = time.time()
    else:
        population = None

    shift_time = 0

    if parameters['DO_SHIFT']:
        shift_time = parameters['SHIFT_FREQ']

    stop = False

    grow_id = 0
    generation_number = 0
    while (not stop) and (generation_number < parameters['NSTEPS']):
        if is_master:
            population = np.array(es.ask_geno())
            # population = np.array(es.ask())
            population = template.insert_active_splines(population)

        costs, max_ni, min_ni, avg_ni = objective_fxn(
            population, weights, return_ni=True,
            penalty=parameters['PENALTY_ON']
        )

        if is_master:
            new_costs = np.sum(costs, axis=1)

            es.tell(
                population[:, active_ind], new_costs
            )
            es.disp()
            stop = es.stop()

            sort_indices = np.argsort(new_costs)

            sorted_pop = population[sort_indices]
            tmp_max_ni = max_ni[sort_indices]
            tmp_min_ni = min_ni[sort_indices]
            tmp_avg_ni = avg_ni[sort_indices]
            new_costs = new_costs[sort_indices]

            if generation_number % parameters['CHECKPOINT_FREQ'] == 0:
                src.partools.checkpoint(
                    sorted_pop, new_costs, tmp_max_ni, tmp_min_ni, tmp_avg_ni,
                    generation_number, parameters, template,
                    parameters['NSTEPS']
                )

        if parameters['DO_GROW']:
            if grow_id < len(parameters['GROW_SCHED']):
                if generation_number + 1 == parameters['GROW_SCHED'][grow_id]:
                    es = cma.CMAEvolutionStrategy(
                        es.result.xbest,
                        0.01,
                        {
                            'verb_disp': 1,
                            'popsize': parameters['GROW_SIZE'][grow_id],
                            # 'verb_append': generation_number,
                        }
                    )

                    grow_id += 1

        if parameters['DO_SHIFT']:
            if shift_time == 0:

                if is_master:
                    best = template.insert_active_splines(np.atleast_2d(
                        es.result.xbest
                    ))

                else:
                    best = None

                best_costs, best_max_ni, best_min_ni, best_avg_ni = objective_fxn(
                    best, weights, return_ni=True,
                    penalty=parameters['PENALTY_ON']
                )

                # if is_master:
                # 
                #     print('Rescaling ...')
                #     solution = src.partools.rescale_ni(
                #         template.insert_active_splines(np.atleast_2d(es.result.xbest)),
                #         best_min_ni, best_max_ni, template
                #     )
                # 
                # best_costs, best_max_ni, best_min_ni, best_avg_ni = objective_fxn(
                #     solution, weights, return_ni=True,
                #     penalty=parameters['PENALTY_ON']
                # )

                if is_master:

                    solution = solution[:, np.where(template.active_mask)[0]]

                    new_u_domains = src.partools.shift_u(
                        best_min_ni, best_max_ni
                    )

                    print("New U domains:", new_u_domains)
                    print("Restarting with new U[]...", flush=True)

                    es = cma.CMAEvolutionStrategy(
                        solution[0],
                        parameters['CMAES_STEP_SIZE'],
                        {
                            'verb_disp': 1,
                            'popsize': parameters['POP_SIZE'],
                            'verb_append': generation_number,
                            # 'CMA_mu': 19,
                            # 'CMA_rankmu': 0.1
                        }
                    )

                else:
                    new_u_domains = None

                new_u_domains = world_comm.bcast(new_u_domains, root=0)
                template.u_ranges = new_u_domains

                shift_time = parameters['SHIFT_FREQ'] - 1 
            else:
                shift_time -= 1

        generation_number += 1
        stop = world_comm.bcast(stop, root=0)

    # end CMA-ES loop

    if is_master:
        cma_runtime = time.time() - cma_start_time

        es.result_pretty()

        # best = np.atleast_2d(es.result.xbest[np.where(template.active_mask)[0]])
        best = np.atleast_2d(es.result.xbest)
        best = np.atleast_2d(template.insert_active_splines(best))
        print("Fitness before LM:", es.result.fbest)

        polish_start_time = time.time()
    else:
        best = np.empty((1, template.pvec_len))

    best = src.partools.local_minimization(
        best[:, np.where(template.active_mask)[0]], best, template, objective_fxn,
        gradient, weights, world_comm, is_master, nsteps=10, lm_output=True,
        penalty=parameters['PENALTY_ON']
    )

    if is_master:
        # sorted_pop[0, np.where(template.active_mask)[0]] = best
        sorted_pop[0, active_ind] = best
        # sorted_pop[0] = best
        population = sorted_pop

    costs, max_ni, min_ni, avg_ni = objective_fxn(
        population, weights, return_ni=True,
        penalty=parameters['PENALTY_ON']
    )

    if is_master:
        polish_runtime = time.time() - polish_start_time

        final_costs = np.sum(costs, axis=1)

        sort_indices = np.argsort(final_costs)
        tmp_max_ni = max_ni[sort_indices]
        tmp_min_ni = min_ni[sort_indices]
        tmp_avg_ni = avg_ni[sort_indices]
        final_costs = final_costs[sort_indices]

        src.partools.checkpoint(
            population, final_costs, tmp_max_ni, tmp_min_ni, tmp_avg_ni,
            generation_number, parameters, template,
            parameters['NSTEPS']
        )

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


