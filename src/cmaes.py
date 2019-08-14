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
    objective_fxn, _ = src.partools.build_evaluation_functions(
        template, all_struct_names, node_manager,
        world_comm, is_master, true_values
    )

    if is_master:
        solution = template.pvec.copy()

        active_ind = np.where(template.active_mask)[0]
        solution[active_ind] = template.generate_random_instance()[active_ind]
    else:
        solution = None

    weights = np.ones(len(all_struct_names))

    costs, max_ni, min_ni, avg_ni = objective_fxn(
        solution, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Initial min/max ni:", min_ni[0], max_ni[0])

        solution = src.partools.rescale_ni(
            np.atleast_2d(solution), min_ni, max_ni, template
        )[0]

    costs, max_ni, min_ni, avg_ni = objective_fxn(
        solution, weights, return_ni=True, penalty=parameters['PENALTY_ON']
    )

    if is_master:
        print("Rescaled initial min/max ni:", min_ni[0], max_ni[0])

    solution = world_comm.bcast(solution, root=0)

    if is_master:
        opts = cma.CMAOptions()

        for key,val in opts.defaults().items():
            print(key, val)

        es = cma.CMAEvolutionStrategy(solution, 0.2, {'popsize': 100})

        es.opts.set({'verb_disp': 1})
    else:
        population = None

    stop = False

    generation_number = 0
    while (not stop) and (generation_number < parameters['NSTEPS']):
        if is_master:
            population = np.array(es.ask(100))

        costs, max_ni, min_ni, avg_ni = objective_fxn(
            population, weights, return_ni=True,
            penalty=parameters['PENALTY_ON']
        )

        if is_master:
            new_costs = np.sum(costs, axis=1)

            es.tell(population, new_costs)
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

        generation_number += 1
        stop = world_comm.bcast(stop, root=0)

    if is_master:
        es.result_pretty()


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


