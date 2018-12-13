"""A genetic algorithm module for use in potential fitting and database
optimization. In terms of the original paper, this module would be intended to
run a GA over the fitting databases as well as a GA to find the theta_MLE.

Authors: Josh Vita (UIUC), Dallas Trinkle (UIUC)
"""

import os
os.chdir("/home/jvita/scripts/s-meam/project/")

import numpy as np
np.random.seed(42)
np.set_printoptions(precision=16, suppress=True)

import h5py
from deap import base, creator, tools

from src.worker import Worker

################################################################################


def get_hdf5_testing_database(load=False):
    """Builds or loads a new HDF5 database of workers.

    Args:
        load (bool): True if database already exists; default is False
    """

    return h5py.File("data/fitting_databases/seed_42/workers.hdf5", 'a',
            driver='core')

def load_workers_and_names_from_database(hdf5_file, weights):
    workers = []
    names = []

    for i,name in enumerate(hdf5_file.keys()):
        if weights[i] > 0:
            workers.append(Worker.from_hdf5(hdf5_file, name))
            names.append(name)

    return workers, names


################################################################################
