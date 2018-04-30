"""Library for computing cost functions and performing parameter optimization.

Authors: Josh Vita (UIUC), Dallas Trinkle (UIUC)"""

import os
import pickle
import glob
import array
import numpy as np
np.set_printoptions(precision=16, suppress=True)

from deap import base, creator, tools, algorithms

from src.worker import Worker

os.chdir("/home/jvita/scripts/s-meam/project/")

################################################################################

def get_structure_list():
    full_paths = glob.glob("data/fitting_databases/seed_42/evaluator.*")
    file_names = [os.path.split(path)[-1] for path in full_paths]

    return [os.path.splitext(f_name)[-1][1:] for f_name in file_names]

def load_workers(all_names):
    path = "data/fitting_databases/seed_42/evaluator."

    return [pickle.load(open(path + name, 'rb')) for name in all_names]

def load_true_energies_and_forces():
    path = "data/fitting_databases/seed_42/info."

    true_energies = {}
    true_forces = {}

    for struct_name in glob.glob(path + '*'):
        f_name = os.path.split(struct_name)[-1]
        atoms_name = os.path.splitext(f_name)[-1][1:]

        # with open(struct_name, 'rb') as f:
        eng = np.genfromtxt(open(struct_name, 'rb'), max_rows=1)
        fcs = np.genfromtxt(open(struct_name, 'rb'), skip_header=1)

        true_energies[atoms_name] = eng
        true_forces[atoms_name] = fcs

    return true_energies, true_forces

def build_ga_toolbox(pvec_len):

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual",
            np.ndarray, fitness=creator.CostFunctionMinimizer)
            # array.array, typecode='l', fitness=creator.CostFunctionMinimizer)

    # TODO: suggested is to use array.array b/c faster copying

    import src.meam
    from src.meam import MEAM

    pot = MEAM.from_file('data/fitting_databases/seed_42/seed_42.meam')
    _, y_pvec, _ = src.meam.splines_to_pvec(pot.splines)

    noise = np.random.normal(0, 1, y_pvec.shape)
    y_pvec += noise

    def ret_pvec():
        return y_pvec

    toolbox = base.Toolbox()
    toolbox.register("attr_float", ret_pvec)
    toolbox.register("individual", tools.initRepeat,
            creator.Individual, toolbox.attr_float, n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selBest)

    return toolbox

def serial_ga_with_workers():
    """A serial genetic algorithm using actual Worker objects"""

    true_energies, true_forces = load_true_energies_and_forces()
    energy_weights = np.ones(len(true_energies)).tolist()

    structure_names = get_structure_list()

    sorted_true_energies = [true_energies[key] for key in structure_names]
    sorted_true_forces = [true_forces[key] for key in structure_names]

    workers = load_workers(structure_names)

    PVEC_LEN = workers[0].len_param_vec
    POP_SIZE = 100

    toolbox = build_ga_toolbox(PVEC_LEN)

    def evaluate(individual):
        """Note: an 'individual' is just a set of parameters"""
        energies = []
        forces = []

        for w in workers:
            eng = w.compute_energy(individual) / len(w.atoms)
            fcs = w.compute_forces(individual) / len(w.atoms)

            energies.append(eng)
            forces.append(fcs)

        energies = [results[0] for results in energies]
        forces = [results[0] for results in forces]

        return force_matching(
                forces, sorted_true_forces, energies, sorted_true_energies,
                energy_weights),

    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=POP_SIZE)

    # record statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
            ngen=50, stats=stats, verbose=True,)

    # print(logbook)

def genetic_algorithm_example():
    """A simple genetic algorithm using the DEAP library"""

    from deap import base, creator, tools, algorithms

    # define base classes for the fitness evaluator and the individual class
    creator.create("FitnessMin", base.Fitness, weights=(-1., 1.))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    SIZE = 10

    # register a function 'individual' that constructs an individual
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.random)
    toolbox.register("individual",
            tools.initRepeat, creator.Individual, toolbox.attr_float, n=SIZE)

    indiv1 = toolbox.individual()
    print("One individual:\n{}".format(indiv1))

    # register a function to build populations of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=5)
    print("A population of 5 individuals:\n{}".format(pop))

    # evaluate individuals
    def evaluate(individual):
        a = sum(individual)
        b = len(individual)

        return a, 1./b

    for indiv in pop:
        indiv.fitness.values = evaluate(indiv)

    print("Evaluation of population:\n{}".format([ind.fitness for ind in pop]))

    # perform mutations; indiv1 is untouched; mutant and indiv2 are identical
    mutant = toolbox.clone(indiv1)
    indiv2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
    del mutant.fitness.values # delete old values copied from indiv1

    print(indiv1)
    print(mutant)
    print("indiv2 == mutant:", indiv2 is mutant)

    # blend two individuals; again, copy beforehand
    child1 = toolbox.clone(pop[0])
    child2 = toolbox.clone(pop[1])
    print("Before blend:\n", child1, child2)

    tools.cxBlend(child1, child2, 0.5)

    print("After blend:\n", child1, child2)

    # select the 2 best individuals using their fitness values
    selected = tools.selBest(pop, 2)
    print("Top 2 fitnesses:", [ind.fitness.values for ind in selected])

    # add the above tools to the toolbox; condenses the algorithm into the box
    toolbox.register("evaluate", evaluate)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selBest)

    val = evaluate(pop[0])
    new1, new2 = toolbox.mate(child1, child2)
    mutant2 = toolbox.mutate(mutant)
    top2 = toolbox.select(pop, 2)

    # record statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
            ngen=50, stats=stats, verbose=False,)

    print(logbook)

def force_matching(computed_forces, true_forces, computed_others=[],
        true_others=[], weights=[]):
    """Computes the cost function using the force-matching method, as outlined
    in the original paper by Ercolessi. Notation matches original paper.

    Args:
        computed_forces (list): computed force matrices
        true_forces (list): true force matrices
        computed_others(list): other constraints (e.g. energies, lattice
            constants, etc)
        true_others (list): true values for other constraints; order matches
            computed
        weights (list): weights on additional constraints

    Returns:
        Z (float): the evaluation of the cost function
    """

    # TODO: this function needs to be profiled and optimized
    # TODO: vectorize for multiple paramsets at once

    if computed_others:
        if not true_others: raise ValueError("must specify true values")
        elif not weights: raise ValueError("must specify weights")

    Zf = sum([np.sum((comp - true)**2)
                for comp,true in zip(computed_forces, true_forces)])

    Zf /= 3*sum([force_mat.shape[0] for force_mat in computed_forces])

    Zc = 0
    if computed_others:
        Zc = sum([w*((comp - true)**2)
                for w,comp,true in zip(weights, computed_others, true_others)])

    return Zf + Zc

################################################################################

if __name__ == "__main__":
    serial_ga_with_workers()
