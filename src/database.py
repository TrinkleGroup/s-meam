"""
An HDF5 implementation of a database of structures. Serves as an interface into
an HDF5 file containing the structure vectors and metadata of each structure.

/root
    /structure
        /phi
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

import h5py
import numpy as np
from ase.neighborlist import NeighborList

import src.meam
import src.lammpsTools
from src.workerSplines import WorkerSpline, RhoSpline, ffgSpline, USpline

class Database(h5py.File):

    def __init__(self, file_name, len_pvec=None, types=None, knot_xcoords=None,
                 x_indices=None, cutoffs=None):
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


        """

        super().__init__(file_name, "a")

        optional_args = ['len_pvec', 'types', 'ntypes', 'nphi',
                         'knot_xcoords', 'x_indices', 'cutoffs']

        # check if attributes already exist in the HDF5 file
        new_arg = None
        for arg in optional_args:
            if arg not in self.attrs:
                if arg == 'len_pvec':       new_arg = len_pvec
                if arg == 'types':          new_arg = types
                if arg == 'ntypes':         new_arg = len(types)
                if arg == 'knot_xcoords':   new_arg = knot_xcoords
                if arg == 'x_indices':      new_arg = x_indices
                if arg == 'cutoffs':        new_arg = cutoffs
                if arg == 'nphi':
                    new_arg = int(
                        (self.attrs["ntypes"] + 1) * self.attrs["ntypes"] / 2
                    )

                self.attrs[arg] = new_arg

    def build_from_lammps_files(self, path_to_lammps_files):
        pass

    def build_from_existing_workers(self, path_to_workers):
        pass

    def add_structure(self, new_group_name, atoms):
        """
        Adds a group to the database and prepares spline structure vectors.

        Args:
            new_group_name: (str) what to name the new HDF5 group
            atoms: (ase.Atoms) atomic structure
        """

        new_group = self.create_group(new_group_name)

        new_group.attrs["type_of_each_atom"] = np.array(
            list(
                map(
                    lambda t: src.lammpsTools.symbol_to_type(
                        t, self.file.types
                    ),
                    atoms.get_chemical_symbols()
                )
            )
        )

        new_group.attrs["natoms"] = len(atoms)

        # TODO: splines should be local variables in add_structure()
        all_splines = self.build_spline_lists(
            self.attrs['knot_xcoords'], self.attrs['x_indices']
        )

        phis = list(all_splines[0])
        rhos = list(all_splines[1])
        fs = list(all_splines[3])
        gs = list(all_splines[4])

        ffgs = self.build_ffg_list(fs, gs)

        # No double counting of bonds; needed for pair interactions
        nl_noboth = NeighborList(
            np.ones(self.attrs["natoms"]) * (self.attrs["cutoff"] / 2.),
            self_interaction=False, bothways=False, skin=0.0
        )

        nl_noboth.update(atoms)

        # Allows double counting bonds; needed for embedding energy calculations
        nl = NeighborList(
            np.ones(self.attrs["natoms"]) * (self.attrs["cutoff"] / 2.),
            self_interaction=False, bothways=True, skin=0.0
        )

        nl.update(atoms)

        all_rij = []
        all_costheta = []

        for i, atom in enumerate(atoms):
            # Record atom type info
            itype = self.attrs["type_of_each_atom"][i]
            ipos = atom.position

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # Stores pair information for phi
            for j, offset in zip(neighbors_noboth, offsets_noboth):
                jtype = self.attrs["type_of_each_atom"][j]

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

                jtype = self.attrs["type_of_each_atom"][j]

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
                        ktype = self.attrs["type_of_each_atom"][k]
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

        # save all energy structure vectors
        for spline_type, splines in zip(['phi', 'rho', 'ffg'], all_splines):

            new_group[spline_type]['energy'] = np.stack(
                [sp.structure_vectors['energy'] for sp in splines]
            )

            new_group[spline_type]['forces'] = np.stack(
                [sp.structure_vectors['forces'] for sp in splines]
            )

    def compute_energy(self, struct_name, potentials):
        pass

    def compute_forces(self, struct_name, potentials):
        pass

    def compute_energy_grad(self, struct_name, potentials):
        pass

    def compute_forces_grad(self, struct_name, potentials):
        pass

    def build_spline_lists(self, knot_xcoords, x_indices):
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
                s = WorkerSpline(knots, bc_type, self.attrs['natoms'])

            elif (self.attrs['nphi'] + self.attrs['ntypes'] <= i
                  < self.nphi + 2 * self.attrs['ntypes']):

                s = USpline(knots, bc_type, self.attrs['natoms'])
            elif i >= self.attrs['nphi'] + 2 * self.attrs['ntypes']:
                s = WorkerSpline(knots, bc_type, self.attrs['natoms'])
            else:
                s = RhoSpline(knots, bc_type, self.attrs['natoms'])

            s.index = self.attrs['x_indices'][i]
            splines.append(s)

        split_indices = [self.attrs['nphi'], self.attrs['nphi'] +
                         self.attrs['ntypes'],
                         self.attrs['nphi'] + 2 * self.attrs['ntypes'],
                         self.attrs['nphi'] + 3 * self.attrs['ntypes']]

        return np.split(np.array(splines), split_indices)

    def build_ffg_list(self, fs, gs):
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

        if not self.fs:
            raise ValueError("f splines have not been set yet")
        elif not self.gs:
            raise ValueError("g splines have not been set yet")

        ffg_list = [[] for _ in range(len(self.fs))]
        for j, fj in enumerate(fs):
            for k, fk in enumerate(fs):
                g = gs[src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                ffg_list[j].append(ffgSpline(fj, fk, g, self.natoms))

        return ffg_list

