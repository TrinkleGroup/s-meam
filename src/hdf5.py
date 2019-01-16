"""A package for loading and evaluating Workers (and subclasses) from HDF5
format"""

import h5py

class HDF5_Spline():

    def __init__(self, path):
        self.dataset = h5py.File(path, 'r')

    def __del__(self):
        self.dataset.close()

    
