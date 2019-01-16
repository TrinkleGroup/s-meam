import numpy as np
from scipy.optimize import line_search
from mpi4py import MPI

"""
A module for doing operations that have been parallelized over potentials
"""

def cg(f, fprime, population, nsteps, comm):
    """Performs CG minimization on every potential in the population

    Args:
        f (callable): the function
        fprime (callable): the gradient
        population (np.arr): each row is an individual potential
        nsteps (int): the number of CG steps to take
        comm (MPI.Comm): comm over master/slave pool; master assumed as rank 0

    Returns:
        minimized (np.arr): the minimized potentials; each row is an individual
    """

    # TODO: function that does F and gradF together

    dir0 = -f(population)
