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
from ase.io import read
from ase.neighborlist import NeighborList

import src.meam
import src.partools
import src.lammpsTools
from src.workerSplines import WorkerSpline, RhoSpline, ffgSpline, USpline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Database(h5py.File):

    def __init__(self, file_name, open_type='a', len_pvec=None, types=None,
                 knot_xcoords=None, x_indices=None, cutoffs=None,
                 overwrite=False, *args, **kwargs):
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

            open_type: (str)
                'r', 'w', 'a'

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

        super().__init__(file_name, open_type, *args, **kwargs)

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

        tmp = self.attrs['nphi'] + len(self.attrs['types'])

        num_u_knots = []
        for i in range(len(self.attrs['types'])):
            num_u_knots.append(
                self.attrs['x_indices'][tmp + 1]
                - self.attrs['x_indices'][tmp]
            )

        self.attrs['num_u_knots'] = num_u_knots

    def add_true_value(self, info_file_name, ref_name):
        # TODO: needs to be able to able to handle diff refs for each struct

        energy = np.genfromtxt(info_file_name, max_rows=1)
        forces = np.genfromtxt(info_file_name, skip_header=1)

        struct_name = os.path.split(info_file_name)[-1].split('.')[-1]

        self[struct_name].attrs['ref_struct'] = ref_name
        true_values_group = self[struct_name].create_group('true_values')

        true_values_group['energy'] = energy
        true_values_group['forces'] = forces

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

# def format_true_values(input_path, output_path, ref_config_name):
#     """
#     Reads values from VASP OUTCAR files and writes into the proper format for
#     use by add_true_value()
#
#     Args:
#         input_path: (str)
#             path to folder containing all OUTCAR files
#
#         output_path: (str)
#             path to write info.* files to
#
#         ref_config_name: (str)
#             name of reference configuration to be used as for energy differences
#
#     # TODO: need a way to note reference configurations for energy differences
#
#     Returns:
#         None; creates info files in proper format
#
#     Output file format:
#
#     info.<structure_name>
#         # header comment line; should specify reference structure
#
#     """
#
#     ref_struct = read(
#         os.path.join(input_path, ref_config_name), format='vasp-out'
#     )
#
#     # TODO: when multiple ref structs are available, move this into for loop
#     header_str = "# ref_name = {}".format(ref_config_name)
#
#     # assumes that OUTCAR files are renamed to be the desired structure name
#     for vasp_file in glob.glob(os.path.join(input_path, '*')):
#         struct_name = os.path.split(vasp_file)[-1]
#
#         atoms = read(vasp_file, format='vasp-out')
#
#         ref_energy = ref_struct.get_potential_energy()
#
#         struct_energy = atoms.get_potential_energy() - ref_energy
#         struct_forces = atoms.get_forces()
#
#         with open(os.path.join(output_path, "info", "info." + struct))
#
#
#
#     # available attrs: cutoffs, knot_xcoords, x_indices, types
#     pass

