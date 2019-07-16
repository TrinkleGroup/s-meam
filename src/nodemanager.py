"""
Handles energy/forces/etc evaluation with hybrid OpenMP/MPI.

Each NodeManager object handles the set of processes on a single compute node.
"""

import time
import numpy as np
import multiprocessing as mp
import src.partools
from itertools import repeat
from multiprocessing import Manager
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: might be able to avoid using mp.Array since it's read only?

struct_vecs = {}
true_values = {'energy': {}, 'forces': {}}

class NodeManager:
    def __init__(self, node_id, template):
        self.node_id = node_id
        self.template = template

        self.loaded_structures = []

        # self.rank_on_node = comm.Get_rank()
        # self.node_size = mp.cpu_count()

        self.pool = None  # should only be started once local data is loaded

        # struct_vecs = {}

        self.x_indices = None
        self.ntypes = None
        self.len_pvec = None
        self.nphi = None

        self.ffg_grad_indices = {}
        self.num_u_knots = []
        self.natoms = {}
        self.type_of_each_atom = {}

        self.weights = {}
        self.ref_name = None

    def start_pool(self, node_size=None):
        if node_size is None:
            node_size = mp.cpu_count()

        self.pool = mp.Pool(node_size)

    # def initialize_shared_memory(self):
    #     manager = Manager()
    #     struct_vecs = manager.dict(struct_vecs)

    def parallel_compute(self, struct_name, potentials, u_domains,
                         compute_type, convert_to_cost):
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

        # TODO: condense energies/forces into costs before MPI send
        # TODO: condense ni into stats before MPI send

        # TODO: instead of reading from Database, take from shared memory
        if compute_type == 'energy':
            # ret = (energy, ni)
            ret = self.compute_energy(
                struct_name, potentials, u_domains
            )

            # ret = list(ret)
            # 
            # ret[0] = self.energy_to_costs(
            #     ret[0], potentials.shape[0], struct_name
            # )

        elif compute_type == 'forces':
            # ret = forces
            ret = self.compute_forces(
                struct_name, potentials, u_domains
            )

            if convert_to_cost:
                ret = self.forces_to_costs(ret, potentials.shape[0], struct_name)

        elif compute_type == 'energy_grad':
            # ret = energy_gradient
            ret = self.compute_energy_grad(
                struct_name, potentials, u_domains
            )

        elif compute_type == 'forces_grad':
            # ret = forces_gradient

            forces = self.compute_forces(
                struct_name, potentials, u_domains
            )

            grad = self.compute_forces_grad(
                struct_name, potentials, u_domains
            )

            ret = self.condense_force_grads(
                forces, grad, potentials.shape[0], struct_name
            )

        else:
            raise ValueError(
                "'compute_type' must be one of ['energy', 'forces',"
                "'energy_grad', 'forces_grad']"
            )

        return ret

    def energy_to_costs(self, energy, npots, struct_name):
        """
        Args:
            energy (dict): key=struct_name, val=energy
            npots (int): number of potentials that were evaluated
            struct_name (str): name of structure that was evaluated

        Note:
            assumes that the weights have been properly updated
        """

        # energy_costs = np.zeros((npots, len(energy)))

        # for cost_id, (name, energy) in enumerate(energy.items()):
        true_ediff = true_values['energy'][struct_name]

        # TODO: can't evaluate energy cost here since requires ref struct
        comp_ediff = energy[struct_name] - all_eng[self.ref_name]

        tmp = (comp_ediff - true_ediff)**2

        energy_costs[:, cost_id] = tmp*self.weights[struct_name]

        # return energy_costs
        return tmp*self.weights[struct_name]


    def forces_to_costs(self, forces, npots, struct_name):
        """
        Args:
            forces (dict): key=struct_name, val=forces
            npots (int): number of potentials that were evaluated
            struct_name (str): name of structure that was evaluated

        Note:
            assumes that the weights have been properly updated
        """

        # force_costs = np.zeros((npots, len(forces)))

        # for cost_id, (name, forces) in enumerate(forces.items()):
        true_forces = true_values['forces'][struct_name]

        diff = forces - true_forces

        epsilon = np.linalg.norm(diff, 'fro', axis=(1, 2))/np.sqrt(10)

        # force_costs[:, cost_id] = epsilon*epsilon*self.weights[struct_name]

        # return force_costs
        return epsilon*epsilon*self.weights[struct_name]

    def condense_force_grads(self, forces, force_grad, npots, struct_name):
        """
        Args:
            forces (dict): key=struct_name, val=forces
            npots (int): number of potentials that were evaluated
            struct_name (str): name of structure that was evaluated

        Note:
            assumes that the weights have been properly updated
        """

        true_forces = true_values['forces'][struct_name]

        diff = forces - true_forces

        scaled = np.einsum('pna,pnak->pnak', diff, force_grad)
        summed = scaled.sum(axis=1).sum(axis=1)

        return summed


    def compute(self, compute_type, struct_list, potentials, u_domains,
            convert_to_cost=True):
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
                repeat(compute_type), repeat(convert_to_cost)
            )
        )

        ret_dict = dict(zip(struct_list, return_values))

        return ret_dict
        # return dict(zip(struct_list, return_values))

    def load_structures(self, struct_list, hdf5_file, load_true=False):
        """
        Loads the structures from the HDF5 file into shared memory.

        Args:
            struct_list: (list[str])
                list of structures to be loaded

            hdf5_file: (h5py.File)
                HDF5 file to load structures from

        Returns:
            None. Updates class variables, loading structure vectors into
            shared memory.

        """

        # things to load: ntypes, num_u_knots, phi, rho, ffg, types_per_atom
        self.ntypes = hdf5_file.attrs['ntypes']
        self.len_pvec = hdf5_file.attrs['len_pvec']
        self.num_u_knots = hdf5_file.attrs['num_u_knots']
        self.x_indices = hdf5_file.attrs['x_indices']
        self.nphi = hdf5_file.attrs['nphi']

        # doing it this way so that it only loads a certain amount at once
        for struct_name in struct_list:
            self.load_one_struct(struct_name, hdf5_file, load_true)
            self.loaded_structures.append(struct_name)

    def load_one_struct(self, struct_name, hdf5_file, load_true):
        natoms = hdf5_file[struct_name].attrs['natoms']
        self.natoms[struct_name] = natoms

        atom_types = hdf5_file[struct_name].attrs['type_of_each_atom']
        self.type_of_each_atom[struct_name] = atom_types

        struct_vecs[struct_name] = {
            'phi': {'energy': {}, 'forces': {}},
            'rho': {'energy': {}, 'forces': {}},
            'ffg': {'energy': {}, 'forces': {}}
        }

        # TOOD: do you need the [()] still? probably

        # load phi structure vectors
        for idx in hdf5_file[struct_name]['phi']['energy']:

            eng = hdf5_file[struct_name]['phi']['energy'][idx][()]
            fcs = hdf5_file[struct_name]['phi']['forces'][idx][()]

            # fcs_shm = mp.Array('d', ni*nj*nk, lock=False)
            # fcs_loc = np.frombuffer(fcs_shm)
            # fcs_loc = fcs_loc.reshape((ni, nj, nk))
            # fcs_loc[:] = fcs[:]

            struct_vecs[struct_name]['phi']['energy'][idx] = eng
            struct_vecs[struct_name]['phi']['forces'][idx] = fcs

        # load rho structure vectors
        for idx in hdf5_file[struct_name]['rho']['energy']:

            eng = hdf5_file[struct_name]['rho']['energy'][idx][()]
            fcs = hdf5_file[struct_name]['rho']['forces'][idx][()]

            struct_vecs[struct_name]['rho']['energy'][idx] = eng
            struct_vecs[struct_name]['rho']['forces'][idx] = fcs

        self.ffg_grad_indices[struct_name] = {}
        self.ffg_grad_indices[struct_name]['ffg_grad_indices'] = {}

        # load ffg structure vectors and indices
        for j in hdf5_file[struct_name]['ffg']['energy']:
            struct_vecs[struct_name]['ffg']['energy'][j] = {}
            struct_vecs[struct_name]['ffg']['forces'][j] = {}

            self.ffg_grad_indices[struct_name][j] = {}

            for k in hdf5_file[struct_name]['ffg']['energy'][j]:

                # structure vectors
                eng = hdf5_file[struct_name]['ffg']['energy'][j][k][()]
                fcs = hdf5_file[struct_name]['ffg']['forces'][j][k][()]

                struct_vecs[struct_name]['ffg']['energy'][j][k] = eng
                struct_vecs[struct_name]['ffg']['forces'][j][k] = fcs

                # indices for indexing gradients
                indices_group = hdf5_file[struct_name]['ffg_grad_indices'][j][k]

                fj = indices_group['fj_indices'][()]
                fk = indices_group['fk_indices'][()]
                g = indices_group['g_indices'][()]

                self.ffg_grad_indices[struct_name][j][k] = {}

                self.ffg_grad_indices[struct_name][j][k]['fj_indices'] = fj
                self.ffg_grad_indices[struct_name][j][k]['fk_indices'] = fk
                self.ffg_grad_indices[struct_name][j][k]['g_indices'] = g

        if load_true:
            # load true values
            true_eng = hdf5_file[struct_name]['true_values']['energy'][()]
            true_fcs = hdf5_file[struct_name]['true_values']['forces'][()]

            true_values['energy'][struct_name] = true_eng
            true_values['forces'][struct_name] = true_fcs

    def compute_energy(self, struct_name, potentials, u_ranges):
        potentials = np.atleast_2d(potentials)

        n_pots = potentials.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(potentials)

        energy = np.zeros(n_pots)

        # pair interactions
        for i, y in enumerate(phi_pvecs):
            energy += \
                struct_vecs[struct_name]['phi']['energy'][str(i)] @ y.T

        # embedding terms
        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)

        energy += self.embedding_energy(
            struct_name, ni, u_pvecs, u_ranges
        )

        grouped_ni = [
            np.array(ni[:, self.type_of_each_atom[struct_name] - 1 == i])
            for i in range(self.ntypes)
        ]

        return energy, grouped_ni

    def compute_forces(self, struct_name, potentials, u_ranges):
        potentials = np.atleast_2d(potentials)

        # N = self.natoms[struct_name]

        n_pots = potentials.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(potentials)

        forces = np.zeros((n_pots, self.natoms[struct_name], 3))

        # pair forces (phi)
        for phi_idx, y in enumerate(phi_pvecs):
            forces += np.einsum(
                'ijk,pk->pij',
                struct_vecs[struct_name]['phi']['forces'][str(phi_idx)],
                y
            )

        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(
            struct_name, ni, u_pvecs, u_ranges, second=False
        )

        # electron density embedding term (rho)
        embedding_forces = np.zeros((n_pots, 3*(self.natoms[struct_name]**2)))

        for rho_idx, y in enumerate(rho_pvecs):
            embedding_forces += \
                (struct_vecs[struct_name]['rho']['forces'][str(rho_idx)] @ y.T).T

        for j, y_fj in enumerate(f_pvecs):
            for k, y_fk in enumerate(f_pvecs):
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)
                y_g = g_pvecs[g_idx]

                cart1 = np.einsum('ij,ik->ijk', y_fj, y_fk)
                cart1 = cart1.reshape(
                    (cart1.shape[0], cart1.shape[1]*cart1.shape[2])
                )

                cart2 = np.einsum('ij,ik->ijk', cart1, y_g)
                cart_y = cart2.reshape(
                    (cart2.shape[0], cart2.shape[1]*cart2.shape[2])
                )

                embedding_forces += \
                    (struct_vecs[struct_name]['ffg']['forces'][str(j)][str(k)] @ cart_y.T).T

        embedding_forces = embedding_forces.reshape(
            (n_pots, 3, self.natoms[struct_name], self.natoms[struct_name])
        )

        embedding_forces = np.einsum('pijk,pk->pji', embedding_forces, uprimes)

        return forces + embedding_forces

    def compute_energy_grad(self, struct_name, potentials, u_ranges):
        potentials = np.atleast_2d(potentials)

        gradient = np.zeros(potentials.shape)

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(potentials)

        n_pots = potentials.shape[0]

        grad_index = 0

        # gradients of phi are just their structure vectors
        for phi_idx, y in enumerate(phi_pvecs):
            gradient[:, grad_index:grad_index + y.shape[1]] += \
                struct_vecs[struct_name]['phi']['energy'][str(phi_idx)]

            grad_index += y.shape[1]

        # chain rule on U means dU/dn values are needed
        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)

        uprimes = self.evaluate_uprimes(
            struct_name, ni, u_pvecs, u_ranges, second=False
        )

        for rho_idx, y in enumerate(rho_pvecs):
            partial_ni = struct_vecs[struct_name]['rho']['energy'][str(rho_idx)]

            gradient[:, grad_index:grad_index + y.shape[1]] += \
                (uprimes @ np.array(partial_ni))

            grad_index += y.shape[1]

        # add in first term of chain rule
        for u_idx, y in enumerate(u_pvecs):

            ni_sublist = ni[:, self.type_of_each_atom[struct_name] - 1 == u_idx]

            num_knots = self.num_u_knots[u_idx]

            new_knots = np.linspace(
                u_ranges[u_idx][0], u_ranges[u_idx][1], num_knots
            )

            knot_spacing = new_knots[1] - new_knots[0]

            # U splines assumed to have fixed derivatives at boundaries
            M = src.partools.build_M(
                num_knots, knot_spacing, ['fixed', 'fixed']
            )

            extrap_dist = (u_ranges[u_idx][1] - u_ranges[u_idx][0]) / 2

            u_energy_sv = np.zeros((n_pots, num_knots + 2))

            if ni_sublist.shape[1] > 0:

                # begin: equivalent to old u.add_to_energy_struct_vec()
                abcd = self.get_abcd(
                    ni_sublist.ravel(), new_knots, M, extrap_dist
                )

                abcd = abcd.reshape(list(ni_sublist.shape) + [abcd.shape[1]])

                u_energy_sv += np.sum(abcd, axis=1)

                # end

                gradient[:, grad_index:grad_index + y.shape[1]] += u_energy_sv

            grad_index += y.shape[1]

        ffg_indices = self.build_ffg_grad_index_list(
            grad_index, f_pvecs, g_pvecs
        )

        # add in second term of chain rule

        for j, y_fj in enumerate(f_pvecs):
            n_fj = y_fj.shape[1]

            for k, y_fk in enumerate(f_pvecs):

                g_idx = src.meam.ij_to_potl(
                    j + 1, k + 1, self.ntypes
                )

                y_g = g_pvecs[g_idx]

                n_fk = y_fk.shape[1]
                n_g = y_g.shape[1]

                scaled_sv = np.einsum(
                    'pz,zk->pk',
                    uprimes,
                    struct_vecs[struct_name]['ffg']['energy'][str(j)][str(k)]
                )

                coeffs_for_fj = np.einsum("pi,pk->pik", y_fk, y_g)
                coeffs_for_fk = np.einsum("pi,pk->pik", y_fj, y_g)
                coeffs_for_g = np.einsum("pi,pk->pik", y_fj, y_fk)

                coeffs_for_fj = coeffs_for_fj.reshape(
                    (n_pots, y_fk.shape[1] * y_g.shape[1])
                )

                coeffs_for_fk = coeffs_for_fk.reshape(
                    (n_pots, y_fj.shape[1] * y_g.shape[1])
                )

                coeffs_for_g = coeffs_for_g.reshape(
                    (n_pots, y_fj.shape[1] * y_fk.shape[1])
                )

                # every ffgSpline affects grad(f_j), grad(f_k), and grad(g)

                # assumed order: fj_indices, fk_indices, g_indices
                indices_tuple = self.ffg_grad_indices[struct_name][str(j)][str(k)]

                stack = np.zeros((n_pots, n_fj, n_fk * n_g))

                # grad(f_j) contribution
                for l in range(n_fj):
                    sample_indices = indices_tuple['fj_indices'][l]

                    stack[:, l, :] = scaled_sv[:, sample_indices]

                # stack = stack @ coeffs_for_fj
                stack = np.einsum('pzk,pk->pz', stack, coeffs_for_fj)
                gradient[:, ffg_indices[j]:ffg_indices[j] + n_fj] += stack

                stack = np.zeros((n_pots, n_fk, n_fj * n_g))

                # grad(f_k) contribution
                for l in range(n_fk):
                    sample_indices = indices_tuple['fk_indices'][l]

                    stack[:, l, :] = scaled_sv[:, sample_indices]

                stack = np.einsum('pzk,pk->pz', stack, coeffs_for_fk)

                gradient[:, ffg_indices[k]:ffg_indices[k] + n_fk] += stack

                stack = np.zeros((n_pots, n_g, n_fj * n_fk))

                # grad(g) contribution
                for l in range(n_g):
                    sample_indices = indices_tuple['g_indices'][l]

                    stack[:, l, :] = scaled_sv[:, sample_indices]

                stack = np.einsum('pzk,pk->pz', stack, coeffs_for_g)

                tmp_idx = ffg_indices[self.ntypes + g_idx]
                gradient[:, tmp_idx:tmp_idx + n_g] += stack

        return gradient

    def compute_forces_grad(self, struct_name, potentials, u_ranges):
        potentials = np.atleast_2d(potentials)
        n_pots = potentials.shape[0]

        gradient = np.zeros(
            (
                n_pots,
                self.natoms[struct_name],
                3,
                self.len_pvec
            )
        )

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(potentials)

        grad_index = 0

        # N = self.natoms[struct_name]

        # gradients of phi are just their structure vectors
        for phi_idx, y in enumerate(phi_pvecs):
            sv = struct_vecs[struct_name]['phi']['forces'][str(phi_idx)]

            sv = sv.reshape(self.natoms[struct_name], 3, y.shape[1])

            gradient[:, :, :, grad_index:grad_index + y.shape[1]] += sv

            grad_index += y.shape[1]

        # chain rule on U functions means dU/dn values are needed
        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)

        uprimes, uprimes_2, u_deriv_svs = self.evaluate_uprimes(
            struct_name, ni, u_pvecs, u_ranges, second=True, return_sv=True
        )

        embedding_forces = np.zeros(
            (n_pots, 3, self.natoms[struct_name], self.natoms[struct_name])
        )

        # pre-compute all rho forces
        for rho_idx, y in enumerate(rho_pvecs):
            rho_sv = struct_vecs[struct_name]['rho']['forces'][str(rho_idx)]
            embedding_forces += (rho_sv @ y.T).T.reshape(
                n_pots, 3, self.natoms[struct_name], self.natoms[struct_name]
            )

        # pre-compute all ffg forces
        for j, y_fj in enumerate(f_pvecs):
            for k, y_fk in enumerate(f_pvecs):
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)
                y_g = g_pvecs[g_idx]

                cart1 = np.einsum('ij,ik->ijk', y_fj, y_fk)
                cart1 = cart1.reshape(
                    (cart1.shape[0], cart1.shape[1]*cart1.shape[2])
                )

                cart2 = np.einsum('ij,ik->ijk', cart1, y_g)
                cart_y = cart2.reshape(
                    (cart2.shape[0], cart2.shape[1]*cart2.shape[2])
                )

                sv = struct_vecs[struct_name]['ffg']['forces'][str(j)][str(k)]
                ffg_forces = (sv @ cart_y.T).T

                embedding_forces += ffg_forces.reshape(
                    (n_pots, 3, self.natoms[struct_name], self.natoms[struct_name])
                )

        # rho gradient term; there's a U'' and a U' term for each rho
        for rho_idx, y in enumerate(rho_pvecs):
            rho_e_sv = struct_vecs[struct_name]['rho']['energy'][str(rho_idx)]

            rho_f_sv = struct_vecs[struct_name]['rho']['forces'][str(rho_idx)]
            rho_f_sv = np.array(rho_f_sv).reshape(
                (3, self.natoms[struct_name], self.natoms[struct_name], y.shape[1])
            )

            # U'' term
            uprimes_scaled = np.einsum('pi,ij->pij', uprimes_2, rho_e_sv)

            stacking_results = np.zeros(
                (n_pots, self.natoms[struct_name], 3, y.shape[1])
            )

            stacking_results += np.einsum(
                'pij,pkli->plkj', uprimes_scaled, embedding_forces
            )

            gradient[:, :, :, grad_index:grad_index + y.shape[1]] += \
                stacking_results

            # U' term
            up_contracted_sv = np.einsum('ijkl,pk->pjil', rho_f_sv, uprimes)

            gradient[:, :, :, grad_index:grad_index + y.shape[1]] += \
                up_contracted_sv

            grad_index += y.shape[1]

        # save indices so that embedding_forces can be added later
        tmp_U_indices = []

        # prep for U gradient term
        for i, y in enumerate(u_pvecs):
            tmp_U_indices.append((grad_index, grad_index + y.shape[1]))
            grad_index += y.shape[1]

        # TODO: this should occur in __init__
        ffg_indices = self.build_ffg_grad_index_list(
            grad_index, f_pvecs, g_pvecs
        )

        # ffg gradient terms
        for j, y_fj in enumerate(f_pvecs):
            n_fj = y_fj.shape[1]

            for k, y_fk in enumerate(f_pvecs):
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)
                y_g = g_pvecs[g_idx]

                n_fk = y_fk.shape[1]
                n_g = y_g.shape[1]

                full_len = n_fj * n_fk * n_g

                # U'' term
                upp_contrib = np.einsum(
                    'pzk,paiz->piak',
                    np.einsum(
                        'pz,zk->pzk',
                        uprimes_2,
                        struct_vecs[struct_name]['ffg']['energy'][str(j)][str(k)]
                    ),
                    embedding_forces
                )

                ffg_sv = struct_vecs[struct_name]['ffg']['forces'][str(j)][str(k)]

                # U' term
                up_contrib = np.einsum(
                    'pz,aizk->paik',
                    uprimes,
                    np.array(ffg_sv).reshape(
                        (3, self.natoms[struct_name], self.natoms[struct_name], full_len)
                    )
                )

                up_contrib = np.transpose(up_contrib, axes=(0, 2, 1, 3))

                # Group terms and add to gradient

                coeffs_for_fj = np.einsum("pi,pk->pik", y_fk, y_g)
                coeffs_for_fk = np.einsum("pi,pk->pik", y_fj, y_g)
                coeffs_for_g = np.einsum("pi,pk->pik", y_fj, y_fk)

                coeffs_for_fj = coeffs_for_fj.reshape(
                    (n_pots, y_fk.shape[1] * y_g.shape[1])
                )

                coeffs_for_fk = coeffs_for_fk.reshape(
                    (n_pots, y_fj.shape[1] * y_g.shape[1])
                )

                coeffs_for_g = coeffs_for_g.reshape(
                    (n_pots, y_fj.shape[1] * y_fk.shape[1])
                )

                # pre-computed indices for outer product indexing
                indices_tuple = \
                    self.ffg_grad_indices[struct_name][str(j)][str(k)]

                stack_up = np.zeros(
                    (n_pots, self.natoms[struct_name], 3, n_fj, n_fk * n_g)
                )

                stack_upp = np.zeros(
                    (n_pots, self.natoms[struct_name], 3, n_fj, n_fk * n_g)
                )

                for l in range(n_fj):
                    sample_indices = indices_tuple['fj_indices'][l]

                    stack_up[:, :, :, l, :] = up_contrib[:, :, :,
                                              sample_indices]
                    stack_upp[:, :, :, l, :] = upp_contrib[:, :, :,
                                               sample_indices]

                stack_up = np.einsum('pzakt,pt->pzak', stack_up,
                                     coeffs_for_fj)
                stack_upp = np.einsum('pzakt,pt->pzak', stack_upp,
                                      coeffs_for_fj)

                tmp_ind = ffg_indices[j]
                gradient[:, :, :, tmp_ind:tmp_ind + n_fj] += stack_up
                gradient[:, :, :, tmp_ind:tmp_ind + n_fj] += stack_upp

                stack_up = np.zeros(
                    (n_pots, self.natoms[struct_name], 3, n_fk, n_fj * n_g)
                )

                stack_upp = np.zeros(
                    (n_pots, self.natoms[struct_name], 3, n_fk, n_fj * n_g)
                )

                for l in range(n_fk):
                    sample_indices = indices_tuple['fk_indices'][l]

                    stack_up[:, :, :, l, :] = up_contrib[:, :, :,
                                              sample_indices]
                    stack_upp[:, :, :, l, :] = upp_contrib[:, :, :,
                                               sample_indices]

                stack_up = np.einsum('pzakt,pt->pzak', stack_up,
                                     coeffs_for_fk)
                stack_upp = np.einsum('pzakt,pt->pzak', stack_upp,
                                      coeffs_for_fk)

                tmp_ind = ffg_indices[k]
                gradient[:, :, :, tmp_ind:tmp_ind + n_fk] += stack_up
                gradient[:, :, :, tmp_ind:tmp_ind + n_fk] += stack_upp

                stack_up = np.zeros(
                    (n_pots, self.natoms[struct_name], 3, n_g, n_fj * n_fk)
                )

                stack_upp = np.zeros(
                    (n_pots, self.natoms[struct_name], 3, n_g, n_fj * n_fk)
                )

                for l in range(n_g):
                    sample_indices = indices_tuple['g_indices'][l]

                    stack_up[:, :, :, l, :] = up_contrib[:, :, :,
                                              sample_indices]
                    stack_upp[:, :, :, l, :] = upp_contrib[:, :, :,
                                               sample_indices]

                stack_up = np.einsum('pzakt,pt->pzak', stack_up,
                                     coeffs_for_g)
                stack_upp = np.einsum('pzakt,pt->pzak', stack_upp,
                                      coeffs_for_g)

                tmp_ind = ffg_indices[self.ntypes + g_idx]
                gradient[:, :, :, tmp_ind:tmp_ind + n_g] += stack_up
                gradient[:, :, :, tmp_ind:tmp_ind + n_g] += stack_upp

        # U gradient terms

        for u_idx, (indices, y) in enumerate(zip(tmp_U_indices, u_pvecs)):
            tmp_U_indices.append((grad_index, grad_index + y.shape[1]))
            start, stop = indices

            u_term = np.einsum(
                'zk,paiz->piak', u_deriv_svs[u_idx][0], embedding_forces
            )

            gradient[:, :, :, start:stop] += u_term

        return gradient

    def compute_ni(self, struct_name, rho_pvecs, f_pvecs, g_pvecs):
        """
        Computes ni values for all atoms

        Args:
            struct_name: (str) struct name key
            rho_pvecs: (list) parameter vectors for rho splines
            f_pvecs: (list) parameter vectors for f splines
            g_pvecs: (list) parameter vectors for g splines

        Returns:
            ni: embedding values for each potential for each atom
        """
        n_pots = rho_pvecs[0].shape[0]

        ni = np.zeros((n_pots, self.natoms[struct_name]))

        # Rho contribution
        # for y, rho in zip(rho_pvecs, self.rhos):
        for i, y in enumerate(rho_pvecs):
            ni += (struct_vecs[struct_name]['rho']['energy'][str(i)] @ y.T).T

        if self.natoms[struct_name] < 3:
            return ni

        # Three-body contribution
        for j, y_fj in enumerate(f_pvecs):
            for k, y_fk in enumerate(f_pvecs):

                g_idx = src.meam.ij_to_potl(
                    j + 1, k + 1, self.ntypes
                )

                y_g = g_pvecs[g_idx]

                cart1 = np.einsum("ij,ik->ijk", y_fj, y_fk)
                cart1 = cart1.reshape(
                    (cart1.shape[0], cart1.shape[1]*cart1.shape[2])
                )

                cart2 = np.einsum("ij,ik->ijk", cart1, y_g)

                cart_y = cart2.reshape(
                    (cart2.shape[0], cart2.shape[1]*cart2.shape[2])
                )

                ni += (struct_vecs[struct_name]['ffg']['energy'][str(j)][str(k)] @ cart_y.T).T

        return ni

    def embedding_energy(self, struct_name, ni, u_pvecs, new_range):
        n_pots = u_pvecs[0].shape[0]

        u_energy = np.zeros(n_pots)

        # evaluate U, U'
        for i, y in enumerate(u_pvecs):
            num_knots = self.num_u_knots[i]

            u_energy_sv = np.zeros((n_pots, num_knots + 2))

            # extract ni values for atoms of type i
            ni_sublist = ni[:, self.type_of_each_atom[struct_name] - 1 == i]

            num_embedded = ni_sublist.shape[1]

            if num_embedded > 0:
                u_range = new_range[i]

                # begin: equivalent to old u.update_knot_positions()
                new_knots = np.linspace(u_range[0], u_range[1], num_knots)
                knot_spacing = new_knots[1] - new_knots[0]

                # U splines assumed to have fixed derivatives at boundaries
                M = src.partools.build_M(
                    num_knots, knot_spacing, ['fixed', 'fixed']
                )

                extrap_dist = (u_range[1] - u_range[0]) / 2

                # end

                # begin: equivalent to old u.add_to_energy_struct_vec()
                abcd = self.get_abcd(
                    ni_sublist.ravel(), new_knots, M, extrap_dist
                )

                abcd = abcd.reshape(list(ni_sublist.shape) + [abcd.shape[1]])

                u_energy_sv += np.sum(abcd, axis=1)

                # end

                u_energy += np.einsum("ij,ij->i", u_energy_sv, y)

        return u_energy

    def evaluate_uprimes(
            self, struct_name, ni, u_pvecs, u_ranges, second=False,
            return_sv=False
    ):
        tags = np.arange(self.natoms[struct_name])
        shifted_types = self.type_of_each_atom[struct_name] - 1

        n_pots = len(u_pvecs[0])

        uprimes = np.zeros((n_pots, self.natoms[struct_name]))
        uprimes_2 = None

        structure_vectors = []

        if second:
            uprimes_2 = np.zeros((n_pots, self.natoms[struct_name]))

        for i, y in enumerate(u_pvecs):
            indices = tags[shifted_types == i]

            num_knots = self.num_u_knots[i]

            u_deriv_sv = np.zeros(
                (n_pots, self.natoms[struct_name], num_knots + 2)
            )

            u_2nd_deriv_sv = None

            new_knots = np.linspace(
                u_ranges[i][0], u_ranges[i][1], num_knots
            )

            knot_spacing = new_knots[1] - new_knots[0]

            # U splines assumed to have fixed derivatives at boundaries
            M = src.partools.build_M(
                num_knots, knot_spacing, ['fixed', 'fixed']
            )
            extrap_dist = (u_ranges[i][1] - u_ranges[i][0]) / 2

            if second:
                u_2nd_deriv_sv = np.zeros(
                    (n_pots, self.natoms[struct_name], num_knots + 2)
                )

            if indices.shape[0] > 0:
                values = ni[:, shifted_types == i].ravel()

                abcd = self.get_abcd(values, new_knots, M, extrap_dist, deriv=1)

                abcd = abcd.reshape(list(values.shape) + [abcd.shape[1]])
                abcd = abcd.reshape(
                    (n_pots, abcd.shape[0]//n_pots, abcd.shape[-1])
                )

                u_deriv_sv[:, indices, :] = abcd

                if second:
                    abcd = self.get_abcd(
                        values, new_knots, M, extrap_dist, deriv=2
                    )

                    abcd = abcd.reshape(list(values.shape) + [abcd.shape[1]])
                    abcd = abcd.reshape(
                        (n_pots, abcd.shape[0]//n_pots, abcd.shape[-1])
                    )

                    u_2nd_deriv_sv[:, indices, :] = abcd

            uprimes += np.einsum('ijk,ik->ij', u_deriv_sv, y)

            structure_vectors.append(u_deriv_sv)

            if second:
                uprimes_2 += np.einsum('ijk,ik->ij', u_2nd_deriv_sv, y)

        if return_sv:
            return uprimes, uprimes_2, structure_vectors

        if second:
            return uprimes, uprimes_2

        return uprimes

    def get_abcd(self, x, knots, M, extrap_dist, deriv=0):
        """Calculates the spline coefficients for a set of points x

        Args:
            x (np.arr): list of points to be evaluated
            deriv (int): optionally compute the 1st derivative instead

        Returns:
            alpha: vector of coefficients to be added to alpha
            beta: vector of coefficients to be added to betas
            lhs_extrap: vector of coefficients to be added to lhs_extrap vector
            rhs_extrap: vector of coefficients to be added to rhs_extrap vector
        """
        x = np.atleast_1d(x)
        n_knots = len(knots)

        # mn, mx = onepass_min_max(x)
        mn = np.min(x)
        mx = np.max(x)

        lhs_extrap_dist = max(float(extrap_dist), knots[0] - mn)
        rhs_extrap_dist = max(float(extrap_dist), mx - knots[-1])

        # add ghost knots
        knots = list([knots[0] - lhs_extrap_dist]) + knots.tolist() +\
                list([knots[-1] + rhs_extrap_dist])

        knots = np.array(knots)

        # indicates the splines that the points fall into
        spline_bins = np.digitize(x, knots, right=True) - 1
        spline_bins = np.clip(spline_bins, 0, len(knots) - 2)

        if (np.min(spline_bins) < 0) or (np.max(spline_bins) >  n_knots+2):
            raise ValueError("Bad extrapolation; a point lies outside of the "
                             "computed extrapolation range")

        prefactor = knots[spline_bins + 1] - knots[spline_bins]

        t = (x - knots[spline_bins]) / prefactor
        t2 = t*t
        t3 = t2*t

        if deriv == 0:

            A = 2*t3 - 3*t2 + 1
            B = t3 - 2*t2 + t
            C = -2*t3 + 3*t2
            D = t3 - t2

        elif deriv == 1:

            A = 6*t2 - 6*t
            B = 3*t2 - 4*t + 1
            C = -6*t2 + 6*t
            D = 3*t2 - 2*t

        elif deriv == 2:

            A = 12*t - 6
            B = 6*t - 4
            C = -12*t + 6
            D = 6*t - 2
        else:
            raise ValueError("Only allowed derivative values are 0, 1, and 2")

        scaling = 1 / prefactor
        scaling = scaling**deriv

        B *= prefactor
        D *= prefactor

        A *= scaling
        B *= scaling
        C *= scaling
        D *= scaling

        alpha = np.zeros((len(x), n_knots))
        beta = np.zeros((len(x), n_knots))

        # values being extrapolated need to be indexed differently
        lhs_extrap_mask = spline_bins == 0
        rhs_extrap_mask = spline_bins == n_knots

        lhs_extrap_indices = np.arange(len(x))[lhs_extrap_mask]
        rhs_extrap_indices = np.arange(len(x))[rhs_extrap_mask]

        if True in lhs_extrap_mask:
            alpha[lhs_extrap_indices, 0] += A[lhs_extrap_mask]
            alpha[lhs_extrap_indices, 0] += C[lhs_extrap_mask]

            beta[lhs_extrap_indices, 0] += A[lhs_extrap_mask]*(-lhs_extrap_dist)
            beta[lhs_extrap_indices, 0] += B[lhs_extrap_mask]
            beta[lhs_extrap_indices, 0] += D[lhs_extrap_mask]

        if True in rhs_extrap_mask:
            alpha[rhs_extrap_indices, -1] += A[rhs_extrap_mask]
            alpha[rhs_extrap_indices, -1] += C[rhs_extrap_mask]

            beta[rhs_extrap_indices, -1] += B[rhs_extrap_mask]
            beta[rhs_extrap_indices, -1] += C[rhs_extrap_mask]*rhs_extrap_dist
            beta[rhs_extrap_indices, -1] += D[rhs_extrap_mask]

        # now add internal knots
        internal_mask = np.logical_not(lhs_extrap_mask + rhs_extrap_mask)

        shifted_indices = spline_bins[internal_mask] - 1

        np.add.at(alpha, (np.arange(len(x))[internal_mask], shifted_indices),
                  A[internal_mask])
        np.add.at(alpha, (np.arange(len(x))[internal_mask], shifted_indices + 1),
                  C[internal_mask])

        np.add.at(beta, (np.arange(len(x))[internal_mask], shifted_indices),
                  B[internal_mask])
        np.add.at(beta, (np.arange(len(x))[internal_mask], shifted_indices + 1),
                  D[internal_mask])

        big_alpha = np.concatenate([alpha, np.zeros((len(x), 2))], axis=1)

        gamma = np.einsum('ij,ik->kij', M, beta.T)

        return big_alpha + np.sum(gamma, axis=1)

    def build_ffg_grad_index_list(self, grad_index, f_pvecs, g_pvecs):
        """A helper function to simplify indexing the ffg parts of the gradient"""

        # TODO: don't use the pvecs, use the num_knots and do this in __init__

        tmp_index = grad_index
        ffg_indices = [grad_index]

        for y_fj in f_pvecs:
            ffg_indices.append(tmp_index + y_fj.shape[1])
            tmp_index += y_fj.shape[1]

        for y_g in g_pvecs:
            ffg_indices.append(tmp_index + y_g.shape[1])
            tmp_index += y_g.shape[1]

        return ffg_indices

    def parse_parameters(self, parameters):
        """Separates the pre-ordered array of vectors of all spline parameters
        into groups.

        Args:
            parameters (np.arr):
                2D array of knot points and boundary conditions for ALL
                splines for ALL intervals for ALL potentials

        Returns:
            *_pvecs (np.arr):
                each return is a list of arrays of parameters. e.g.
                phi_pvecs[0] is the parameters for the first phi spline for
                every potential
        """
        # Parse parameter vector
        y_indices = [self.x_indices[i] + 2 * i
                     for i in range(len(self.x_indices))]

        params_split = np.split(parameters, y_indices[1:], axis=1)

        nphi = self.nphi
        ntypes = self.ntypes

        phi_pvecs = params_split[:nphi]
        rho_pvecs = params_split[nphi: nphi + ntypes]
        u_pvecs = params_split[nphi + ntypes:nphi + 2 * ntypes]
        f_pvecs = params_split[nphi + 2 * ntypes:nphi + 3 * ntypes]
        g_pvecs = params_split[nphi + 3 * ntypes:]

        return phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs

    def __getstate__(self):
        """Can't pickle Pool objects. _getstate_ says what to pickle"""

        self_dict = self.__dict__.copy()
        del self_dict['pool']
        # del self_dict['struct_vecs']

        return self_dict
