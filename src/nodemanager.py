"""
Handles energy/forces/etc evaluation with hybrid OpenMP/MPI.

Each NodeManager object handles the set of processes on a single compute node.
"""

import numpy as np
import multiprocessing as mp
import src.partools
from itertools import repeat
from src.database import Database

class NodeManager:
    def __init__(self, node_id, template, database_file_name):
        self.node_id = node_id
        # TODO: only need world_comm?
        # self.comm = comm
        self.template = template

        # TODO: database should just be file name - Pool has to open file on own
        self.database_file_name = database_file_name

        # self.rank_on_node = comm.Get_rank()
        self.node_size = mp.cpu_count()

        self.pool = mp.Pool(self.node_size)

        # self.struct_vecs = {}
        # self.ffg_grad_indices = {}
        # self.ntypes = None
        # self.num_u_knots = []
        # self.type_of_each_atoms = {}

    def parallel_compute(self, struct_name, potentials, u_domains,
                         compute_type):
        """
        The function called by the processor Pool. Computes the desired
        property for all structures in 'struct_list'.

        Args:
            struct_name: (str)
                structure name to be evaluated. i.e. HDF5 file key

            potentials: (np.arr)
                potentials to be evaluated

            u_domains: (list[tuple])
                lower and upper-bound knots for each U spline

            compute_type: (str)
                what value to compute; must be one of ['energy', 'forces',
                'energy_grad', 'forces_grad']

        Returns:
            Returns a dictionary of dict[struct_name] = computed_value as
            specified by 'compute_type'.

        """

        with Database(self.database_file_name, 'r') as database:
            if compute_type == 'energy':
                # ret = (energy, ni)
                ret = database.compute_energy(
                    struct_name, potentials, u_domains
                )

            elif compute_type == 'forces':
                # ret = forces
                ret = database.compute_forces(
                    struct_name, potentials, u_domains
                )

            elif compute_type == 'energy_grad':
                # ret = energy_gradient
                ret = database.compute_energy_grad(
                    struct_name, potentials, u_domains
                )

            elif compute_type == 'forces_grad':
                # ret = forces_gradient
                ret = database.compute_forces_grad(
                    struct_name, potentials, u_domains
                )

            else:
                raise ValueError(
                    "'compute_type' must be one of ['energy', 'forces',"
                    "'energy_grad', 'forces_grad']"
                )

            return ret

    def compute(self, compute_type, struct_list, potentials, u_domains):
        """
        Computes a given property using the worker pool on the set of
        structures in 'struct_list'

        Args:
            compute_type:
                (str) 'energy', 'forces', 'energy_grad', or 'forces_grad'

            struct_list: (list[str])
                a list of structure names (HDF5 keys)

            potentials: (np.arr)
                the parameters to be used for evaluation

            u_domains:
                lower and upper bounds on U spline domains

        Returns:
            return value depending upon specified 'compute_type'

        """

        type_list = ['energy', 'forces', 'energy_grad', 'forces_grad']

        if compute_type not in type_list:
            raise ValueError(
                "'compute_type' must be one of ['energy', 'forces',"
                "'energy_grad', 'forces_grad']"
            )


        if type(struct_list) is not list:
            raise ValueError("struct_list must be a list of keys")

        return_values = self.pool.starmap(
            self.parallel_compute,
            zip(
                struct_list, repeat(potentials), repeat(u_domains),
                repeat(compute_type)
            )
        )

        return dict(zip(struct_list, return_values))

    # def load_structures(self, struct_list, hdf5_file):
    #     """
    #     Loads the structures from the HDF5 file into shared memory.
    #
    #     Args:
    #         struct_list: (list[str])
    #             list of structures to be loaded
    #
    #         hdf5_file: (h5py.File)
    #             HDF5 file to load structures from
    #
    #     Returns:
    #         None. Updates class variables, loading structure vectors into
    #         shared memory.
    #
    #     """
    #
    #     # TODO: can write to an HDF5 file in /dev/shm
    #     # things to load: ntypes, num_u_knots, phi, rho, ffg, types_per_atom
    #     # TODO: do you need to use Dataset.value on attributes?
    #     self.ntypes = hdf5_file.attrs['ntypes']
    #     self.num_u_knots = hdf5_file.attrs['num_u_knots']
    #
    #     # TODO: when deleting, try to delete the smallest struct
    #
    #     # doing it this way so that it only loads a certain amount at once
    #     for struct_name in struct_list:
    #         self.load_one_struct(struct_name, hdf5_file)
    #
    # def load_one_struct(self, struct_name, hdf5_file):
    #     atom_types = hdf5_file[struct_name].attrs['type_of_each_atom'].value
    #     self.type_of_each_atoms[struct_name] = atom_types
    #
    #     self.struct_vecs[struct_name] = {'phi': {}, 'rho': {}, 'ffg': {}}
    #
    #     # load phi structure vectors
    #     for idx in hdf5_file[struct_name]['phi']['energy']:
    #
    #         # TODO: may have to use Dataset.value to keep in memory?
    #         # https://stackoverflow.com/questions/26517795/how-do-i-keep-an-h5py-group-in-memory-after-closing-the-file
    #
    #         eng = hdf5_file[struct_name]['phi']['energy'][idx].value
    #         fcs = hdf5_file[struct_name]['phi']['forces'][idx].value
    #
    #         self.struct_vecs[struct_name]['phi']['energy'] = eng
    #         self.struct_vecs[struct_name]['phi']['forces'] = fcs
    #
    #     # load rho structure vectors
    #     for idx in hdf5_file[struct_name]['rho']['energy']:
    #
    #         eng = hdf5_file[struct_name]['rho']['energy'][idx].value
    #         fcs = hdf5_file[struct_name]['rho']['forces'][idx].value
    #
    #         self.struct_vecs[struct_name]['rho']['energy'] = eng
    #         self.struct_vecs[struct_name]['rho']['forces'] = fcs
    #
    #     # load ffg structure vectors and indices
    #     for j in hdf5_file[struct_name]['ffg']['energy']:
    #         self.struct_vecs[struct_name]['ffg']['energy'][j] = {}
    #         self.struct_vecs[struct_name]['ffg']['forces'][j] = {}
    #
    #         self.ffg_grad_indices[struct_name]['ffg_grad_indices'][j] = {}
    #
    #         for k in hdf5_file[struct_name]['ffg']['energy'][j]:
    #
    #             # structure vectors
    #             eng = hdf5_file[struct_name]['ffg']['energy'][j][k].value
    #             fcs = hdf5_file[struct_name]['ffg']['forces'][j][k].value
    #
    #             self.struct_vecs[struct_name]['ffg']['energy'][j][k] = eng
    #             self.struct_vecs[struct_name]['ffg']['forces'][j][k] = fcs
    #
    #             # indices for indexing gradients
    #             indices_group = hdf5_file[struct_name]['ffg_grad_indices'][j][k]
    #
    #             fj = indices_group['fj_indices']
    #             fk = indices_group['fk_indices']
    #             g = indices_group['g_indices']
    #
    #             self.ffg_grad_indices[struct_name][j][k]['fj_indices'] = fj
    #             self.ffg_grad_indices[struct_name][j][k]['fk_indices'] = fk
    #             self.ffg_grad_indices[struct_name][j][k]['g_indices'] = g

    def __getstate__(self):
        """Can't pickle Pool objects. _getstate_ says what to pickle"""

        self_dict = self.__dict__.copy()
        del self_dict['pool']

        return self_dict