"""
An HDF5 implementation of a database of structures. Serves as an interface into
an HDF5 file containing the structure vectors and metadata of each structure.

/root
    /structure
        /phi
            /energy (first index corresponds to phi_A, phi_AB, ...
            /forces
        /rho
        /U
        /f
        /g
        /attributes
            - ntypes
            - num_atoms
            - pvec_len
            - type_of_each_atom
            - nphi
            - ffg_grad_indices
"""

import os
import h5py
import glob
import pickle
import logging
import itertools
import numpy as np
from ase.neighborlist import NeighborList

import src.meam
import src.partools
import src.lammpsTools
from src.workerSplines import WorkerSpline, RhoSpline, ffgSpline, USpline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Database(h5py.File):

    def __init__(self, file_name, len_pvec=None, types=None, knot_xcoords=None,
                 x_indices=None, cutoffs=None, overwrite=False):
        """
        Initializes all of the attributes that are intrinsic to the database.
        Note that the length of the parameter vector _is_ an intrinsic
        property of the database since it determines the shapes of the
        structure vectors.

        Note: all arguments (except the file name) are optional -- if they
        aren't provided, it's assumed that they are being read as attributes
        in an existing HDF5 file.

        Args:
            file_name: (str)
                file name

            len_pvec: (int)
                the number of elements in the full vector of spline parameters

            types: (list[str])
                the elemental types described by the database

            knot_xcoords: (np.arr)
                a 1D array of knots points where points are ordered by spline
                type. Spline groups are ordered as [phi, rho, u, f, g] where
                each group has a number of splines depending on the number of
                elements in the system, and each spline has an arbitrary
                number of knots depending on how the potential was created

                e.g. if the system is Ti-O, the potential looks as follows:
                    phi_Ti-Ti, phi_Ti-O, phi_O-O, rho_Ti, rho_O, U_Ti, U_O,
                    f_Ti, f_O, g_Ti-Ti, g_Ti-O, g_O-O

                    where phi_Ti-Ti could have 5 knots, phi_Ti-O 9, rho_O 13,
                    etc.

                This array will be broken apart into individual splines (
                phi_Ti-Ti, phi_Ti-O, ...), then grouped according to spline
                type (phi, rho, u, f, g)

            x_indices: (list)
                starting index of each spline. since each spline does not
                necessarily have the same number of knots, a list of indices
                must be provided to deliminate each spline in the 1D vector.

            cutoffs: (list)
                [lower, upper] bound cutoff ranges for the potential

            overwrite: (bool)
                if true, deletes the current copy of the HDF5 file

        """

        if overwrite:
            if os.path.isfile(file_name):
                os.remove(file_name)

        super().__init__(file_name, "a")

        optional_args = ['len_pvec', 'types', 'ntypes', 'nphi',
                         'knot_xcoords', 'x_indices', 'cutoffs']

        # check if attributes already exist in the HDF5 file
        new_arg = None
        for arg in optional_args:
            if arg not in self.attrs:
                if arg == 'len_pvec':       new_arg = len_pvec
                if arg == 'types':          new_arg = np.array(types, dtype='S')
                if arg == 'ntypes':         new_arg = len(types)
                if arg == 'knot_xcoords':   new_arg = knot_xcoords
                if arg == 'x_indices':      new_arg = x_indices
                if arg == 'cutoffs':        new_arg = cutoffs
                if arg == 'nphi':
                    new_arg = int(
                        (self.attrs["ntypes"] + 1) * self.attrs["ntypes"] / 2
                    )

                self.attrs[arg] = new_arg

        tmp = self.attrs['nphi'] + len(types)

        num_u_knots = []
        for i in range(len(types)):
            num_u_knots.append(
                self.attrs['x_indices'][tmp + 1]
                - self.attrs['x_indices'][tmp]
            )

        self.attrs['num_u_knots'] = num_u_knots

    def add_structure(self, new_group_name, atoms, overwrite=False):
        """
        Adds a group to the database and prepares spline structure vectors.

        Args:
            new_group_name: (str) what to name the new HDF5 group
            atoms: (ase.Atoms) atomic structure
        """

        # if don't want to overwrite, just return
        if (new_group_name in self) and (not overwrite):
            return

        # if overwriting and already exists, delete current copy
        if new_group_name in self:
            del self[new_group_name]

        # if group doesn't already exist
        new_group = self.create_group(new_group_name)

        new_group.attrs["type_of_each_atom"] = np.array(
            list(
                map(
                    lambda t: src.lammpsTools.symbol_to_type(
                        # t, self.attrs['types'].tolist()
                        t, [s.decode('utf-8') for s in self.attrs[
                            'types'].tolist()]
                    ),
                    atoms.get_chemical_symbols()
                )
            )
        )

        new_group.attrs["natoms"] = len(atoms)

        all_splines = self.build_spline_lists(
            self.attrs['knot_xcoords'], self.attrs['x_indices'], len(atoms)
        )

        phis = list(all_splines[0])
        rhos = list(all_splines[1])
        fs = list(all_splines[3])
        gs = list(all_splines[4])

        ffgs = self.build_ffg_list(fs, gs, len(atoms))

        ffg_grad_indices = self.compute_grad_indices(ffgs)

        keys = ['fj_indices', 'fk_indices', 'g_indices']

        for j, ffg_list in enumerate(ffgs):
            for k, ffg in enumerate(ffg_list):
                # assumed order: fj_indices, fk_indices, g_indices

                for idx, ind_list in enumerate(ffg_grad_indices[j][k]):
                    string = '/'.join(
                        [new_group_name, 'ffg_grad_indices', str(j), str(k),
                         keys[idx]]
                    )

                    self[string] = np.array(ind_list)

        # No double counting of bonds; needed for pair interactions
        nl_noboth = NeighborList(
            np.ones(new_group.attrs["natoms"]) * (self.attrs["cutoffs"][-1]/2.),
            self_interaction=False, bothways=False, skin=0.0
        )

        nl_noboth.update(atoms)

        # Allows double counting bonds; needed for embedding energy calculations
        nl = NeighborList(
            np.ones(new_group.attrs["natoms"]) * (self.attrs["cutoffs"][-1]/2.),
            self_interaction=False, bothways=True, skin=0.0
        )

        nl.update(atoms)

        all_rij = []
        all_costheta = []

        for i, atom in enumerate(atoms):
            # Record atom type info
            itype = new_group.attrs["type_of_each_atom"][i]
            ipos = atom.position

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # Stores pair information for phi
            for j, offset in zip(neighbors_noboth, offsets_noboth):
                jtype = new_group.attrs["type_of_each_atom"][j]

                # Find displacement vector (with periodic boundary conditions)
                jpos = atoms[j].position + np.dot(offset, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0] ** 2 + jvec[1] ** 2 + jvec[2] ** 2)
                jvec /= rij

                all_rij.append(rij)

                # Add distance/index/direction information to necessary lists
                phi_idx = src.meam.ij_to_potl(
                    itype, jtype, self.attrs["ntypes"]
                )

                # phi
                phis[phi_idx].add_to_energy_struct_vec(rij)

                phis[phi_idx].add_to_forces_struct_vec(rij, jvec, i)
                phis[phi_idx].add_to_forces_struct_vec(rij, -jvec, j)

                # rho
                rhos[jtype - 1].add_to_energy_struct_vec(rij, i)
                rhos[itype - 1].add_to_energy_struct_vec(rij, j)

                rhos[jtype - 1].add_to_forces_struct_vec(rij, jvec, i, i)
                rhos[jtype - 1].add_to_forces_struct_vec(rij, -jvec, j, i)

                rhos[itype - 1].add_to_forces_struct_vec(rij, jvec, i, j)
                rhos[itype - 1].add_to_forces_struct_vec(rij, -jvec, j, j)

            # Store distances, angle, and index info for embedding terms
            # TODO: rename j_idx to be more clear
            j_idx = 0  # for tracking neighbor
            for j, offsetj in zip(neighbors, offsets):

                jtype = new_group.attrs["type_of_each_atom"][j]

                # offset accounts for periodic images
                jpos = atoms[j].position + np.dot(offsetj, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0] ** 2 + jvec[1] ** 2 + jvec[2] ** 2)
                jvec /= rij

                # prepare for angular calculations
                a = jpos - ipos
                na = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

                fj_idx = jtype - 1

                j_idx += 1
                for k, offsetk in zip(neighbors[j_idx:], offsets[j_idx:]):
                    if k != j:
                        ktype = new_group.attrs["type_of_each_atom"][k]
                        kpos = atoms[k].position + np.dot(offsetk,
                                                          atoms.get_cell())

                        kvec = kpos - ipos
                        rik = np.sqrt(
                            kvec[0] ** 2 + kvec[1] ** 2 + kvec[2] ** 2)
                        kvec /= rik

                        b = kpos - ipos
                        nb = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

                        cos_theta = np.dot(a, b) / na / nb
                        all_costheta.append(cos_theta)

                        fk_idx = ktype - 1

                        d0 = jvec
                        d1 = -cos_theta * jvec / rij
                        d2 = kvec / rij
                        d3 = kvec
                        d4 = -cos_theta * kvec / rik
                        d5 = jvec / rik

                        dirs = np.vstack([d0, d1, d2, d3, d4, d5])

                        ffgs[fj_idx][fk_idx].add_to_energy_struct_vec(
                            rij, rik, cos_theta, i)

                        ffgs[fj_idx][fk_idx].add_to_forces_struct_vec(
                            rij, rik, cos_theta, dirs, i, j, k)

        # prepare groups for structures vectors
        new_group.create_group("phi")
        new_group.create_group("rho")
        new_group.create_group("ffg")

        for spline_type in ['phi', 'rho', 'ffg']:
            new_group[spline_type].create_group('energy')
            new_group[spline_type].create_group('forces')

        # save all energy structure vectors
        for spline_type, splines in zip(['phi', 'rho'], all_splines):
            for i, sp in enumerate(splines):

                new_group[spline_type]['energy'][str(i)] = \
                    sp.structure_vectors['energy']

                new_group[spline_type]['forces'][str(i)] = \
                    sp.structure_vectors['forces']

        for j, ffg_list in enumerate(ffgs):

            new_group['ffg']['energy'].create_group(str(j))
            new_group['ffg']['forces'].create_group(str(j))

            for k, ffg in enumerate(ffg_list):

                new_group['ffg']['energy'][str(j)][str(k)] = \
                    ffg.structure_vectors['energy']

                new_group['ffg']['forces'][str(j)][str(k)] = \
                    ffg.structure_vectors['forces']

    def add_from_existing_workers(self, path_to_workers):
        """
        Assumes that pickled Worker objects already exist and are stored in
        'path_to_workers'.

        Args:
            new_file_name: (str)
                the name of the HDF5 file to be constructed

            path_to_workers: (str)
                full path to directory containing pickled Worker objects

        Returns:
            None; creates and saves an HDF5 file with name 'new_file_name'

        """
        for struct_path in glob.glob(os.path.join(path_to_workers, '*.pkl')):
            worker = pickle.load(open(struct_path, 'rb'))

            struct_name = os.path.splitext(os.path.split(struct_path)[-1])[0]

            logging.debug(struct_name)

            new_group = self.create_group(struct_name)

            new_group.attrs['type_of_each_atom'] = np.array(
                worker.type_of_each_atom
            )

            new_group.attrs['natoms'] = worker.natoms

            for j, ffg_list in enumerate(worker.ffg_grad_indices):
                for k, ffg_indices in enumerate(ffg_list):
                    for idx, key in enumerate(['fj_indices', 'fk_indices',
                                               'g_indices']):
                        string = '/'.join(
                            [struct_name, 'ffg_grad_indices', str(j), str(k),
                             key]
                        )

                        self[string] = np.array(ffg_indices[key])

            # prepare groups for structures vectors
            new_group.create_group("phi")
            new_group.create_group("rho")
            new_group.create_group("ffg")

            for spline_type in ['phi', 'rho', 'ffg']:
                new_group[spline_type].create_group('energy')
                new_group[spline_type].create_group('forces')

            # save all energy structure vectors
            for spline_type, splines in zip(['phi', 'rho'], [worker.phis,
                                                             worker.rhos]):
                for i, sp in enumerate(splines):

                    new_group[spline_type]['energy'][str(i)] = \
                        sp.structure_vectors['energy']

                    new_group[spline_type]['forces'][str(i)] = \
                        sp.structure_vectors['forces']

            for j, ffg_list in enumerate(worker.ffgs):

                new_group['ffg']['energy'].create_group(str(j))
                new_group['ffg']['forces'].create_group(str(j))

                for k, ffg in enumerate(ffg_list):

                    new_group['ffg']['energy'][str(j)][str(k)] = \
                        ffg.structure_vectors['energy']

                    new_group['ffg']['forces'][str(j)][str(k)] = \
                        ffg.structure_vectors['forces']

    def compute_energy(self, struct_name, potentials, u_ranges):
        potentials = np.atleast_2d(potentials)

        n_pots = potentials.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(potentials)

        energy = np.zeros(n_pots)

        # pair interactions
        for i, y in enumerate(phi_pvecs):
            energy += self[struct_name]['phi']['energy'][str(i)] @ y.T

        # embedding terms
        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)

        energy += self.embedding_energy(
            struct_name, ni, u_pvecs, u_ranges
        )

        return energy, ni

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

        ni = np.zeros((n_pots, self[struct_name].attrs['natoms']))

        # Rho contribution
        # for y, rho in zip(rho_pvecs, self.rhos):
        for i, y in enumerate(rho_pvecs):
            ni += (self[struct_name]['rho']['energy'][str(i)] @ y.T).T

        # Three-body contribution
        for j, y_fj in enumerate(f_pvecs):
            for k, y_fk in enumerate(f_pvecs):
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.attrs['ntypes'])
                y_g = g_pvecs[g_idx]

                cart1 = np.einsum("ij,ik->ijk", y_fj, y_fk)
                cart1 = cart1.reshape(
                    (cart1.shape[0], cart1.shape[1]*cart1.shape[2])
                )

                cart2 = np.einsum("ij,ik->ijk", cart1, y_g)

                cart_y = cart2.reshape(
                    (cart2.shape[0], cart2.shape[1]*cart2.shape[2])
                )

                ni += (self[struct_name]['ffg']['energy'][str(j)][str(k)] @ cart_y.T).T

        return ni

    def embedding_energy(self, struct_name, ni, u_pvecs, new_range):
        n_pots = u_pvecs[0].shape[0]

        u_energy = np.zeros(n_pots)

        # evaluate U, U'
        for i, y in enumerate(u_pvecs):
            num_knots = self.attrs['num_u_knots'][i]

            u_energy_sv = np.zeros((n_pots, num_knots + 2))

            # extract ni values for atoms of type i
            ni_sublist = ni[
                 :, self[struct_name].attrs['type_of_each_atom'] - 2 == i
             ]

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

                # rzm: db is adding U energy when Worker isn't (for dimer_ab)

                u_energy += np.einsum("ij,ij->i", u_energy_sv, y)

                logging.info("db U energy: {}".format(np.einsum("ij,ij->i", u_energy_sv, y)))


        return u_energy

    def evaluate_uprimes(
            self, struct_name, ni, u_pvecs, u_ranges, second=False,
            return_sv=False
    ):
        tags = np.arange(self[struct_name].attrs['natoms'])
        shifted_types = self[struct_name].attrs['type_of_each_atom'] - 1

        n_pots = len(u_pvecs[0])

        uprimes = np.zeros((n_pots, self[struct_name].attrs['natoms']))
        uprimes_2 = None

        structure_vectors = []

        if second:
            uprimes_2 = np.zeros((n_pots, self[struct_name].attrs['natoms']))

        for i, y in enumerate(u_pvecs):
            indices = tags[shifted_types == i]

            num_knots = self.attrs['num_u_knots'][i]

            u_deriv_sv = np.zeros(
                (n_pots, self[struct_name].attrs['natoms'], num_knots + 2)
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
                    (n_pots, self[struct_name].attrs['natoms'], num_knots + 2)
                )

            if indices.shape[0] > 0:
                values = ni[:, shifted_types == i].ravel()

                abcd = self.get_abcd(values, new_knots, M, extrap_dist, deriv=1)

                abcd = abcd.reshape(list(values.shape) + [abcd.shape[1]])

                u_deriv_sv[:, indices, :] = abcd

                if second:
                    abcd = self.get_abcd(
                        values, new_knots, M, extrap_dist, deriv=2
                    )

                    abcd = abcd.reshape(list(values.shape) + [abcd.shape[1]])

                    u_2nd_deriv_sv[:, indices, :] = abcd

            uprimes += np.einsum('ijk,ik->ij', u_deriv_sv, y)

            structure_vectors.append(u_deriv_sv)

            if second:
                uprimes_2 += np.einsum('ijk,ik->ij', u_2nd_deriv_sv, y)

        # if none: return uprimes
        # if second: return uprimes, uprimes_2
        # if return_sv:

        # returns = [uprimes]

        if return_sv:
            return uprimes, uprimes_2, structure_vectors

        if second:
            return uprimes, uprimes_2

        return uprimes

    def compute_forces(self, struct_name, potentials, u_ranges):
        potentials = np.atleast_2d(potentials)

        n_pots = potentials.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(potentials)

        forces = np.zeros((n_pots, self[struct_name].attrs['natoms'], 3))

        # pair forces (phi)
        for phi_idx, y in enumerate(phi_pvecs):
            forces += np.einsum(
                'ijk,pk->pij',
                self[struct_name]['phi']['forces'][str(phi_idx)],
                y
            )

        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(
            struct_name, ni, u_pvecs, u_ranges, second=False
        )

        # electron density embedding term (rho)
        embedding_forces = np.zeros(
            (n_pots, 3*(self[struct_name].attrs['natoms']**2))
        )

        for rho_idx, y in enumerate(rho_pvecs):
            embedding_forces += (self[struct_name]['rho']['forces'][str(rho_idx)] @ y.T).T

        for j, y_fj in enumerate(f_pvecs):
            for k, y_fk in enumerate(f_pvecs):
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.attrs['ntypes'])
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
                    (self[struct_name]['ffg']['forces'][str(j)][str(k)] @ cart_y.T).T

        N = self[struct_name].attrs['natoms']

        embedding_forces = embedding_forces.reshape((n_pots, 3, N, N))
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
                self[struct_name]['phi']['energy'][str(phi_idx)]

            grad_index += y.shape[1]

        # chain rule on U means dU/dn values are needed
        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)

        uprimes = self.evaluate_uprimes(
            struct_name, ni, u_pvecs, u_ranges, second=False
        )

        for rho_idx, y in enumerate(rho_pvecs):
            partial_ni = self[struct_name]['rho']['energy'][str(rho_idx)]

            gradient[:, grad_index:grad_index + y.shape[1]] += \
                (uprimes @ np.array(partial_ni))

            grad_index += y.shape[1]

        # add in first term of chain rule
        for u_idx, y in enumerate(u_pvecs):

            ni_sublist = ni[:, self[struct_name].attrs['type_of_each_atom'] - 1 == u_idx]

            num_knots = self.attrs['num_u_knots'][u_idx]

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
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.attrs['ntypes'])
                y_g = g_pvecs[g_idx]

                n_fk = y_fk.shape[1]
                n_g = y_g.shape[1]

                scaled_sv = np.einsum(
                    'pz,zk->pk',
                    uprimes,
                    self[struct_name]['ffg']['energy'][str(j)][str(k)]
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
                indices_tuple = self[struct_name]['ffg_grad_indices'][str(j)][str(k)]

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

                tmp_idx = ffg_indices[self.attrs['ntypes'] + g_idx]
                gradient[:, tmp_idx:tmp_idx + n_g] += stack

        return gradient

    def compute_forces_grad(self, struct_name, potentials, u_ranges):
        potentials = np.atleast_2d(potentials)
        n_pots = potentials.shape[0]

        gradient = np.zeros(
            (
                n_pots,
                self[struct_name].attrs['natoms'],
                3,
                self.attrs['len_pvec']
            )
        )

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(potentials)

        grad_index = 0

        N = self[struct_name].attrs['natoms']

        # gradients of phi are just their structure vectors
        for phi_idx, y in enumerate(phi_pvecs):
            sv = np.array(self[struct_name]['phi']['forces'][str(phi_idx)])
            sv = sv.reshape(N, 3, y.shape[1])

            gradient[:, :, :, grad_index:grad_index + y.shape[1]] += sv

            grad_index += y.shape[1]

        # chain rule on U functions means dU/dn values are needed
        ni = self.compute_ni(struct_name, rho_pvecs, f_pvecs, g_pvecs)

        uprimes, uprimes_2, u_deriv_svs = self.evaluate_uprimes(
            struct_name, ni, u_pvecs, u_ranges, second=True, return_sv=True
        )

        embedding_forces = np.zeros((n_pots, 3, N, N))

        # pre-compute all rho forces
        for rho_idx, y in enumerate(rho_pvecs):
            rho_sv = self[struct_name]['rho']['forces'][str(rho_idx)]
            embedding_forces += (rho_sv @ y.T).T.reshape(n_pots, 3, N, N)

        # pre-compute all ffg forces
        for j, y_fj in enumerate(f_pvecs):
            for k, y_fk in enumerate(f_pvecs):
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.attrs['ntypes'])
                y_g = g_pvecs[g_idx]

                cart1 = np.einsum('ij,ik->ijk', y_fj, y_fk)
                cart1 = cart1.reshape(
                    (cart1.shape[0], cart1.shape[1]*cart1.shape[2])
                )

                cart2 = np.einsum('ij,ik->ijk', cart1, y_g)
                cart_y = cart2.reshape(
                    (cart2.shape[0], cart2.shape[1]*cart2.shape[2])
                )

                ffg_forces = \
                    (self[struct_name]['ffg']['forces'][str(j)][str(k)] @ cart_y.T).T

                embedding_forces += ffg_forces.reshape((n_pots, 3, N, N))

        # rho gradient term; there's a U'' and a U' term for each rho
        for rho_idx, y in enumerate(rho_pvecs):
            rho_e_sv = self[struct_name]['rho']['energy'][str(rho_idx)]

            rho_f_sv = self[struct_name]['rho']['forces'][str(rho_idx)]
            rho_f_sv = np.array(rho_f_sv).reshape((3, N, N, y.shape[1]))

            # U'' term
            uprimes_scaled = np.einsum('pi,ij->pij', uprimes_2, rho_e_sv)

            stacking_results = np.zeros((n_pots, N, 3, y.shape[1]))

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
        ffg_indices = self.build_ffg_grad_index_list(grad_index, f_pvecs,
                                                     g_pvecs)

        # ffg gradient terms
        for j, y_fj in enumerate(f_pvecs):
            n_fj = y_fj.shape[1]

            for k, y_fk in enumerate(f_pvecs):
                g_idx = src.meam.ij_to_potl(j + 1, k + 1, self.attrs['ntypes'])
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
                        self[struct_name]['ffg']['energy'][str(j)][str(k)]
                    ),
                    embedding_forces
                )

                ffg_sv = self[struct_name]['ffg']['forces'][str(j)][str(k)]

                # U' term
                up_contrib = np.einsum(
                    'pz,aizk->paik',
                    uprimes,
                    np.array(ffg_sv).reshape(
                        (3, N, N, full_len)
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
                    self[struct_name]['ffg_grad_indices'][str(j)][str(k)]

                stack_up = np.zeros((n_pots, N, 3, n_fj, n_fk * n_g))
                stack_upp = np.zeros((n_pots, N, 3, n_fj, n_fk * n_g))

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

                stack_up = np.zeros((n_pots, N, 3, n_fk, n_fj * n_g))
                stack_upp = np.zeros((n_pots, N, 3, n_fk, n_fj * n_g))

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

                stack_up = np.zeros((n_pots, N, 3, n_g, n_fj * n_fk))
                stack_upp = np.zeros((n_pots, N, 3, n_g, n_fj * n_fk))

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

                tmp_ind = ffg_indices[self.attrs['ntypes'] + g_idx]
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

    def build_spline_lists(self, knot_xcoords, x_indices, natoms):
            """
            Builds lists of phi, rho, u, f, and g WorkerSpline objects

            Args:
                knot_xcoords: joined array of knot coordinates for all splines
                x_indices: starting index in knot_xcoords of each spline

            Returns:
                splines: (list) list of lists of splines; [phis, rhos, us, fs, gs]
            """

            knots_split = np.split(knot_xcoords, x_indices[1:])

            # TODO: could specify bc outside of Worker and pass in
            # bc_type = ('natural', 'natural')
            bc_type = ('fixed', 'fixed')

            splines = []

            for i, knots in enumerate(knots_split):
                if i < self.attrs['nphi']:
                    s = WorkerSpline(knots, bc_type, natoms)

                elif (self.attrs['nphi'] + self.attrs['ntypes'] <= i
                      < self.attrs['nphi'] + 2 * self.attrs['ntypes']):

                    s = USpline(knots, bc_type, natoms)
                elif i >= self.attrs['nphi'] + 2 * self.attrs['ntypes']:
                    s = WorkerSpline(knots, bc_type, natoms)
                else:
                    s = RhoSpline(knots, bc_type, natoms)

                s.index = self.attrs['x_indices'][i]
                splines.append(s)

            split_indices = [self.attrs['nphi'], self.attrs['nphi'] +
                             self.attrs['ntypes'],
                             self.attrs['nphi'] + 2 * self.attrs['ntypes'],
                             self.attrs['nphi'] + 3 * self.attrs['ntypes']]

            return np.split(np.array(splines), split_indices)

    def build_ffg_list(self, fs, gs, natoms):
        """
        Creates all combinations of f*f*g splines for use with triplet
        calculations.

        Args:
            fs : list of all f WorkerSpline objects
            gs : list of all g WorkerSpline objects

        Returns:
            ffg_list: 2D list where ffg_list[i][j] is the ffgSpline built
                using splines f_i, f_k, and g_ij
        """

        if not fs:
            raise ValueError("f splines have not been set yet")
        elif not gs:
            raise ValueError("g splines have not been set yet")

        ffg_list = [[] for _ in range(len(fs))]
        for j, fj in enumerate(fs):
            for k, fk in enumerate(fs):
                g = gs[src.meam.ij_to_potl(j + 1, k + 1, self.attrs['ntypes'])]

                ffg_list[j].append(ffgSpline(fj, fk, g, natoms))

        return ffg_list

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
        y_indices = [self.attrs['x_indices'][i] + 2 * i
                     for i in range(len(self.attrs['x_indices']))]

        params_split = np.split(parameters, y_indices[1:], axis=1)

        nphi = self.attrs['nphi']
        ntypes = self.attrs['ntypes']

        phi_pvecs = params_split[:nphi]
        rho_pvecs = params_split[nphi: nphi + ntypes]
        u_pvecs = params_split[nphi + ntypes:nphi + 2 * ntypes]
        f_pvecs = params_split[nphi + 2 * ntypes:nphi + 3 * ntypes]
        g_pvecs = params_split[nphi + 3 * ntypes:]

        return phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs

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

    def compute_grad_indices(self, ffgs):
        """Prepares lists of indices for extracting partial derivatives of the
        outer product of f_j*f_k*g
        """

        # TODO: this doubles the memory footprint, but improves performance

        ffg_indices = []

        for ffg_list in ffgs:
            tmp_list = []

            for ffg in ffg_list:
                n_fj = len(ffg.fj.knots) + 2
                n_fk = len(ffg.fk.knots) + 2
                n_g = len(ffg.g.knots) + 2

                fj_indices = np.zeros((n_fj, n_fk * n_g))
                for l in range(n_fj):
                    fj_indices[l] = np.arange(n_fk * n_g) + l * n_fk * n_g

                tmp_indices = np.arange(n_g)

                fk_indices = np.zeros((n_fk, n_fj * n_g))
                for l in range(n_fk):
                    k_indices = np.arange(l * n_g, n_fj * n_fk * n_g,
                                          n_fk * n_g)

                    gen = itertools.product(k_indices, tmp_indices)
                    fk_indices[l] = [e1 + e2 for e1, e2 in gen]

                g_indices = np.zeros((n_g, n_fj * n_fk))
                for l in range(n_g):
                    g_indices[l] = np.arange(l, n_fk * n_fj * n_g, n_g)

                fj_indices = fj_indices.astype(int)
                fk_indices = fk_indices.astype(int)
                g_indices = g_indices.astype(int)

                # assumed order: fj_indices, fk_indices, g_indices
                tmp_list.append([fj_indices, fk_indices, g_indices])

            ffg_indices.append(tmp_list)

        return ffg_indices


