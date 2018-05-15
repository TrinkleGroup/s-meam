import numpy as np
import logging
from scipy.sparse import lil_matrix

from ase.neighborlist import NeighborList
from pympler import muppy, summary

import src.lammpsTools
import src.meam
from src.workerSplines import WorkerSpline, RhoSpline, ffgSpline, USpline

logger = logging.getLogger(__name__)


class Worker:
    """Worker designed to compute energies and forces for a single structure
    using a variety of different MEAM potentials

    Assumptions:
        - knot x-coordinates never change
        - the atomic structure never changes
    """

    # TODO: in general, need more descriptive variable/function names

    # @profile
    def __init__(self, atoms, knot_xcoords, x_indices, types, load_file=False):
        """Organizes data structures and pre-computes structure information.

        Args:
            atoms (ASE.Atoms):
                an ASE representation of an atomic system

            knot_xcoords (np.arr):
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

            x_indices (list):
                starting index of each spline. since each spline does not
                necessarily have the same number of knots, a list of indices
                must be provided to deliminate each spline in the 1D vector.

            types (list):
                set of atomic types described by the potential. note: this
                cannot be inferred from 'atoms' since the structure may not
                have every atom type in it.

            load_file (bool): True if loading from HDF5
        """

        # Basic variable initialization
        # self.atoms      = atoms
        # self.types      = types
        # self.name       = name

        if load_file: return

        ntypes          = len(types)
        self.ntypes     = ntypes
        self.natoms     = len(atoms)

        self.len_param_vec = len(knot_xcoords) + 2*len(x_indices)

        f = lambda t: src.lammpsTools.symbol_to_type(t, types)
        self.type_of_each_atom = list(map(f, atoms.get_chemical_symbols()))

        # TODO: rename self.nphi to be more clear
        # there are nphi phi functions and nphi g fxns
        nphi            = int((self.ntypes+1)*self.ntypes/2)
        self.nphi       = nphi

        all_splines = self.build_spline_lists(knot_xcoords, x_indices)

        self.phis = list(all_splines[0])
        self.rhos = list(all_splines[1])
        self.us = list(all_splines[2])
        self.fs = list(all_splines[3])
        self.gs = list(all_splines[4])

        self.ffgs = self.build_ffg_list(self.fs, self.gs)

        # Compute full potential cutoff distance (based only on radial fxns)
        radial_fxns = self.phis + self.rhos + self.fs
        cutoff = np.max([max(s.x) for s in radial_fxns])

        # Build neighbor lists

        # No double counting of bonds; needed for pair interactions
        nl_noboth = NeighborList(np.ones(self.natoms) * (cutoff / 2.),
                                 self_interaction=False, bothways=False,
                                 skin=0.0)
        nl_noboth.update(atoms)

        # Allows double counting bonds; needed for embedding energy calculations
        nl = NeighborList(np.ones(self.natoms) * (cutoff / 2.),
                          self_interaction=False, bothways=True, skin=0.0)
        nl.update(atoms)

        for i, atom in enumerate(atoms):
            # Record atom type info
            itype = self.type_of_each_atom[i]
            ipos = atom.position

            # Extract neigbor list for atom i
            neighbors_noboth, offsets_noboth = nl_noboth.get_neighbors(i)
            neighbors, offsets = nl.get_neighbors(i)

            # Stores pair information for phi
            for j, offset in zip(neighbors_noboth, offsets_noboth):

                jtype = self.type_of_each_atom[j]

                # Find displacement vector (with periodic boundary conditions)
                jpos = atoms[j].position + np.dot(offset, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0]**2 + jvec[1]**2 + jvec[2]**2)
                jvec /= rij

                # Add distance/index/direction information to necessary lists
                phi_idx = src.meam.ij_to_potl(itype, jtype, self.ntypes)

                # phi
                self.phis[phi_idx].add_to_energy_struct_vec(rij)

                self.phis[phi_idx].add_to_forces_struct_vec(rij, jvec, i)
                self.phis[phi_idx].add_to_forces_struct_vec(rij, -jvec, j)

                # rho
                self.rhos[jtype-1].add_to_energy_struct_vec(rij, i)
                self.rhos[itype-1].add_to_energy_struct_vec(rij, j)

                self.rhos[jtype-1].add_to_forces_struct_vec(rij, jvec, i, i)
                self.rhos[jtype-1].add_to_forces_struct_vec(rij, -jvec, j, i)

                self.rhos[itype-1].add_to_forces_struct_vec(rij, jvec, i, j)
                self.rhos[itype-1].add_to_forces_struct_vec(rij, -jvec, j, j)

            # Store distances, angle, and index info for embedding terms
            # TODO: rename j_idx to be more clear
            j_idx = 0  # for tracking neighbor
            for j, offsetj in zip(neighbors, offsets):

                jtype = self.type_of_each_atom[j]

                # offset accounts for periodic images
                jpos = atoms[j].position + np.dot(offsetj, atoms.get_cell())

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0]**2 + jvec[1]**2 + jvec[2]**2)
                jvec /= rij

                # prepare for angular calculations
                a = jpos - ipos
                na = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

                fj_idx = jtype - 1

                j_idx += 1
                for k, offsetk in zip(neighbors[j_idx:], offsets[j_idx:]):
                    if k != j:
                        ktype = self.type_of_each_atom[k]
                        kpos = atoms[k].position + np.dot(offsetk,
                                                          atoms.get_cell())

                        kvec = kpos - ipos
                        rik = np.sqrt(kvec[0]**2 + kvec[1]**2 + kvec[2]**2)
                        kvec /= rik

                        b = kpos - ipos
                        nb = np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)

                        cos_theta = np.dot(a, b) / na / nb

                        fk_idx = ktype - 1

                        d0 = jvec
                        d1 = -cos_theta * jvec / rij
                        d2 = kvec / rij
                        d3 = kvec
                        d4 = -cos_theta * kvec / rik
                        d5 = jvec / rik

                        dirs = np.vstack([d0, d1, d2, d3, d4, d5])

                        self.ffgs[fj_idx][fk_idx].add_to_energy_struct_vec(
                            rij, rik, cos_theta, i)

                        self.ffgs[fj_idx][fk_idx].add_to_forces_struct_vec(
                            rij, rik, cos_theta, dirs, i, j, k)

        #print()
        #all_objects = muppy.get_objects()
        #summ = summary.summarize(all_objects)
        #summary.print_(summ)

        # convert arrays to avoid having to convert on call
        self.type_of_each_atom = np.array(self.type_of_each_atom)

        for rho in self.rhos:
            rho.forces_struct_vec = rho.forces_struct_vec.tocsr()

        for ffg_list in self.ffgs:
            for ffg in ffg_list:

                ffg.energy_struct_vec =lil_matrix(ffg.energy_struct_vec).tocsr()
                ffg.forces_struct_vec =lil_matrix(ffg.forces_struct_vec).tocsr()

    @classmethod
    def from_hdf5(cls, hdf5_file, name):
        worker_data = hdf5_file[name]

        w = Worker(None, None, None, None, load_file=True)

        w.natoms = int(worker_data.attrs['natoms'])
        w.ntypes = int(worker_data.attrs['ntypes'])
        w.nphi = int(worker_data.attrs['nphi'])
        w.len_param_vec = int(worker_data.attrs['len_param_vec'])

        w.type_of_each_atom = np.array(worker_data['type_of_each_atom'])

        w.phis = [WorkerSpline.from_hdf5(worker_data["phis"], str(i)) for i in
                range(w.nphi)]

        w.rhos = [RhoSpline.from_hdf5(worker_data["rhos"], str(i)) for i in
                range(w.ntypes)]

        w.us = [USpline.from_hdf5(worker_data["us"], str(i)) for i in
                range(w.ntypes)]

        w.fs = [WorkerSpline.from_hdf5(worker_data["fs"], str(i)) for i in
                range(w.ntypes)]

        w.gs = [WorkerSpline.from_hdf5(worker_data["gs"], str(i)) for i in
                range(w.nphi)]

        w.ffgs = [[ffgSpline.from_hdf5(worker_data["ffgs"][str(i)],
                str(j)) for j in range(w.ntypes)] for i in range(w.ntypes)]

        return w

    def add_to_hdf5(self, hdf5_file, name):
        """Adds a worker to an existing HDF5 file

        Args:
            hdf5_file (h5py.File): file to write to
            name (str): name of worker
        """

        new_group = hdf5_file.create_group(name)

        new_group.attrs['natoms'] = self.natoms
        new_group.attrs['ntypes'] = self.ntypes
        new_group.attrs['nphi'] = self.nphi
        new_group.attrs['len_param_vec'] = self.len_param_vec

        new_group.create_dataset("type_of_each_atom",
                data=self.type_of_each_atom)

        phis_group = new_group.create_group("phis")
        for i,sp in enumerate(self.phis): sp.add_to_hdf5(phis_group, str(i))

        rhos_group = new_group.create_group("rhos")
        for i,sp in enumerate(self.rhos): sp.add_to_hdf5(rhos_group, str(i))

        us_group = new_group.create_group("us")
        for i,sp in enumerate(self.us): sp.add_to_hdf5(us_group, str(i))

        fs_group = new_group.create_group("fs")
        for i,sp in enumerate(self.fs): sp.add_to_hdf5(fs_group, str(i))

        gs_group = new_group.create_group("gs")
        for i,sp in enumerate(self.gs): sp.add_to_hdf5(gs_group, str(i))

        ffgs_group = new_group.create_group("ffgs")
        for i,ffg_list in enumerate(self.ffgs):
            mini_group = ffgs_group.create_group(str(i))
            for j,sp in enumerate(ffg_list):
                sp.add_to_hdf5(mini_group, str(j))

    def build_spline_lists(self, knot_xcoords, x_indices):
        """
        Builds lists of phi, rho, u, f, and g WorkerSpline objects

        Args:
            knot_xcoords: joined array of knot coordinates for all splines
            x_indices: starting index in knot_xcoords of each spline

        Returns:
            splines: list of lists of splines; [phis, rhos, us, fs, gs]
        """

        knots_split = np.split(knot_xcoords, x_indices[1:])

        # TODO: could specify bc outside of Worker and pass in
        bc_type = ('fixed', 'fixed')

        splines = []

        for i, knots in enumerate(knots_split):
            if (i < self.nphi) or (i >= self.nphi + 2*self.ntypes):
                s = WorkerSpline(knots, bc_type, self.natoms)
            elif (self.nphi + self.ntypes <= i < self.nphi + 2 *self.ntypes):
                s = USpline(knots, bc_type, self.natoms)
            else:
                s = RhoSpline(knots, bc_type, self.natoms)

            s.index = x_indices[i]
            splines.append(s)

        split_indices = [self.nphi, self.nphi + self.ntypes,
                         self.nphi + 2*self.ntypes, self.nphi + 3*self.ntypes]

        return np.split(splines, split_indices)

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

    # @profile
    def compute_energy(self, parameters):
        """Calculates energies for all potentials using information
        pre-computed during initialization.

        Args:
            parameters (np.arr):
                2D list of all parameters for all splines; each row
                corresponds to a unique potential. Each group in a
                single row should have K+2 elements where K is the number
                of knot points, and the 2 additional are boundary conditions.
                The first K in each group are the knot y-values
        """
        parameters = np.array(parameters)
        parameters = np.atleast_2d(parameters)
        # print(parameters)

        self.n_pots = parameters.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        energy = np.zeros(self.n_pots)

        # Pair interactions
        for y, phi in zip(phi_pvecs, self.phis, ):
            if phi.energy_struct_vec.shape[0] > 0:
                energy += phi.calc_energy(y)

        # Embedding terms
        energy += self.embedding_energy(self.compute_ni(rho_pvecs, f_pvecs,\
                                                        g_pvecs), u_pvecs)

        return energy

    # @profile
    def compute_ni(self, rho_pvecs, f_pvecs, g_pvecs):
        """
        Computes ni values for all atoms

        Args:
            rho_pvecs: parameter vectors for rho splines
            f_pvecs: parameter vectors for f splines
            g_pvecs: parameter vectors for g splines

        Returns:
            ni: potential energy
        """
        ni = np.zeros((self.n_pots, self.natoms))

        # Rho contribution
        for y, rho in zip(rho_pvecs, self.rhos):
            ni += rho.calc_energy(y).T
            np.set_printoptions(precision=16)
            # logging.info("WORKER: rho =\n{0}".format(rho.calc_energy(y)))

        # Three-body contribution
        for j, (y_fj,ffg_list) in enumerate(zip(f_pvecs, self.ffgs)):
            for k, (y_fk,ffg) in enumerate(zip(f_pvecs, ffg_list)):

                y_g = g_pvecs[src.meam.ij_to_potl(j + 1, k + 1, self.ntypes)]

                ni += ffg.calc_energy(y_fj, y_fk, y_g)

        return ni

    # @profile
    def embedding_energy(self, ni, u_pvecs):
        """
        Computes embedding energy

        Args:
            ni: per-atom ni values
            u_pvecs: parameter vectors for U splines

        Returns:
            u_energy: total embedding energy
        """

        for i, u in enumerate(self.us):
            u.energy_struct_vec = np.zeros((self.n_pots, 2*u.x.shape[0]+4))
            u.add_to_energy_struct_vec(ni[:, self.type_of_each_atom - 1 == i])

        # Evaluate U, U', and compute zero-point energies
        u_energy = np.zeros(self.n_pots)
        for y, u in zip(u_pvecs, self.us):

            if len(u.energy_struct_vec) > 0:
                u_energy -= u.compute_zero_potential(y).ravel()
                u_energy += u.calc_energy(y)
                # logging.info("WORKER: U = {0}".format(u.calc_energy(y)))
                # logging.info("WORKER: zero_pot = {0}".format(u.compute_zero_potential(y)))

            u.reset()

        return u_energy

    # @profile
    def evaluate_uprimes(self, ni, u_pvecs):
        """
        Computes U' values for every atom

        Args:
            ni: per-atom ni values
            u_pvecs: parameter vectors for U splines

        Returns:
            uprimes: per-atom U' values
        """

        tags = np.arange(self.natoms)

        # -1 because LAMMPS-style atom type is 1 indexed
        shifted_types = self.type_of_each_atom - 1

        for i, u in enumerate(self.us):
            u.struct_vecs = np.zeros((self.natoms, 2*len(u.x)+4))

            # get atom ids of type i
            indices = tags[shifted_types == i]

            u.deriv_struct_vec = np.zeros(
                (self.n_pots, self.natoms, 2*u.x.shape[0]+4))

            if indices.shape[0] > 0:
                u.add_to_deriv_struct_vec(ni[:, shifted_types == i], indices)

        # Evaluate U, U', and compute zero-point energies
        uprimes = np.zeros((self.n_pots, self.natoms))
        for y, u in zip(u_pvecs, self.us):
            uprimes += u.calc_deriv(y)

        return uprimes

    # @profile
    def compute_forces(self, parameters):
        """Calculates the force vectors on each atom using the given spline
        parameters.

        Args:
            parameters (np.arr): the 1D array of concatenated parameter
                vectors for all splines in the system
        """
        parameters = np.atleast_2d(parameters)
        self.n_pots = parameters.shape[0]

        phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs = \
            self.parse_parameters(parameters)

        forces = np.zeros((self.n_pots, self.natoms, 3))

        # Pair forces (phi)
        for phi_idx, (phi, y) in enumerate(zip(self.phis, phi_pvecs)):

            if len(phi.forces_struct_vec) > 0:
                forces += phi.calc_forces(y)

        ni = self.compute_ni(rho_pvecs, f_pvecs, g_pvecs)
        uprimes = self.evaluate_uprimes(ni, u_pvecs)

        # Electron density embedding (rho)
        for rho_idx, (rho, y) in enumerate(zip(self.rhos, rho_pvecs)):

            if rho.forces_struct_vec.shape[0] > 0:
                rho_forces = rho.calc_forces(y)

                rho_forces = rho_forces.reshape(
                    (self.n_pots, 3, self.natoms, self.natoms))

                forces += np.einsum('pijk,pk->pji', rho_forces, uprimes)

        # Angular terms (ffg)
        for j, ffg_list in enumerate(self.ffgs):
            for k, ffg in enumerate(ffg_list):

                if ffg.forces_struct_vec[0].shape[0] > 0:

                    y_fj = f_pvecs[j]
                    y_fk = f_pvecs[k]
                    y_g = g_pvecs[src.meam.ij_to_potl(j+1, k+1, self.ntypes)]

                    ffg_forces = ffg.calc_forces(y_fj, y_fk, y_g)

                    ffg_forces = ffg_forces.reshape(
                        (self.n_pots, 3, self.natoms, self.natoms))

                    forces += np.einsum('pijk,pk->pji', ffg_forces, uprimes)

        return forces

    # @profile
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

        splines = self.phis + self.rhos + self.us + self.fs + self.gs

        # Parse parameter vector
        x_indices = [s.index for s in splines]
        y_indices = [x_indices[i] + 2 * i for i in range(len(x_indices))]

        params_split = np.split(parameters, y_indices[1:], axis=1)

        nphi = self.nphi
        ntypes = self.ntypes

        phi_pvecs = params_split[:nphi]
        rho_pvecs = params_split[nphi: nphi + ntypes]
        u_pvecs = params_split[nphi + ntypes:nphi + 2*ntypes]
        f_pvecs = params_split[nphi + 2*ntypes:nphi + 3*ntypes]
        g_pvecs = params_split[nphi + 3*ntypes:]

        return phi_pvecs, rho_pvecs, u_pvecs, f_pvecs, g_pvecs

if __name__ == "__main__":
    pass
