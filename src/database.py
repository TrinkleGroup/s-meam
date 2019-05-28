"""
An HDF5 implementation of a database of structures. Serves as an interface into
an HDF5 file containing the structure vectors and metadata of each structure.
"""

import h5py

class Database(h5py.File):

    def __init__(self):
        pass

    def add_structure(self, new_struct):
        pass

    def compute_energy(self, struct_name, potentials):
        pass

    def compute_forces(self, struct_name, potentials):
        pass

    def compute_energy_grad(self, struct_name, potentials):
        pass

    def compute_forces_grad(self, struct_name, potentials):
        pass

