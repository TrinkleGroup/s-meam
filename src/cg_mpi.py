import os
os.chdir("/home/jvita/scripts/s-meam/project/")

import numpy as np
import random
np.set_printoptions(precision=16, linewidth=np.inf, suppress=True)
np.random.seed(42)
random.seed(42)

import pickle
import glob
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import fmin_cg
from scipy.interpolate import CubicSpline
from mpi4py import MPI

from deap import base, creator, tools, algorithms

################################################################################

MASTER_RANK = 0

################################################################################

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    is_master_node = (rank == MASTER_RANK)
