import numpy as np
import logging
import os
import sys

from ase.calculators.lammpsrun import LAMMPS
from ase.neighborlist import NeighborList

import lammpsTools
from spline import Spline, ZeroSpline

logger = logging.getLogger(__name__)


# TODO: will need to be able to set individual splines, not just the whole set

class MEAM:
    """An object for organizing the information needed for a spline meam
    potential in various different representations

    Properties:
        types (list):
            list of str representations of unique elements in the system (
            e.g. ['H', 'He'])
        ntypes (int):
            len(types)
        nphi (int):
            the number of bond types in the system (e.g. H-H, H-He, He-He)

            nphi = (ntypes+1)*ntypes/2)

        phis (list):
            pair interaction splines, nphi in total
        rhos (list):
            electron density splines, ntypes in total
        us (list):
            full embedding energy splines, ntypes in total
        fs (list):
            three-body correction splines, ntypes in total
        gs (list):
            angular contribution splines, nphi in total
    """

    def __init__(self, splines, types):
        """Base constructor"""

        ntypes = len(types)

        if ntypes < 1:
            raise ValueError("must specify at least one atom type")
        elif ntypes > 2:
            raise NotImplementedError(
                "only unary and binary systems are supported")
        elif len(splines) != ntypes * (ntypes + 4):
            raise ValueError("incorrect number of splines for given number of"
                             "atom types")

        self.types = types
        self.ntypes = ntypes
        nphi = int((ntypes + 1) * ntypes / 2)
        self.nphi = nphi

        split_indices = [nphi, nphi + ntypes, nphi + 2 * ntypes,
                         nphi + 3 * ntypes]
        self.phis, self.rhos, self.us, self.fs, self.gs = \
            np.split(splines, split_indices)

        self.phis = list(self.phis)
        self.rhos = list(self.rhos)
        self.us = list(self.us)
        self.fs = list(self.fs)
        self.gs = list(self.gs)

        self.splines = self.phis + self.rhos + self.us + self.fs + self.gs

        radials = [self.phis, self.rhos, self.fs]
        endpoints = [max(fxn.x) for ftype in radials for fxn in ftype]
        self.cutoff = max(endpoints)

        self.zero_atom_energies = [None] * self.ntypes
        for i in range(self.ntypes):
            self.zero_atom_energies[i] = self.us[i](0.0)

        self.uprimes = None
        self.forces = None
        self.energies = None

    @classmethod
    def from_pvec(cls, x_pvec, y_pvec, x_indices, types):
        """Builds a MEAM object from 1D vectors of parameters.

        Args:
            see WorkerManyPotentialsOneStruct.__init()__ for details
            concerning formatting of input arguments

        Returns:
            MEAM object
        """

        x_pvec = np.array(x_pvec)
        y_pvec = np.array(y_pvec)

        x_indices = np.array(x_indices)
        y_indices = [x_indices[i - 1] - 2 * i for i in
                     range(1, len(x_indices) + 1)]

        split_x = np.split(x_pvec, x_indices)
        split_y = np.split(y_pvec, y_indices)

        nsplines = len(types) * (len(types) + 4)
        splines = [None] * nsplines

        for i in range(len(split_x)):
            x, bc = np.split(split_x[i], [-2])

            splines[i] = Spline(x, split_y[i], end_derivs=bc)

        return cls(splines, types)

    @classmethod
    def from_file(cls, fname):
        """Builds MEAM potential using spline information from the given file

        Args:
            fname (str):
                the name of the input file

        Returns:
            MEAM object
        """

        try:
            f = open(fname, 'r')

            f.readline()  # Remove header
            temp = f.readline().split()  # 'meam/spline ...' line
            types = temp[2:]
            ntypes = len(types)

            # Based on how fxns in MEAM are defined
            nsplines = ntypes * (ntypes + 4)

            # Build all splines; separate into different types later
            splines = []
            for i in range(nsplines):
                f.readline()  # throwaway 'spline3eq' line
                nknots = int(f.readline())

                d0, dN = [float(el) for el in f.readline().split()]

                xcoords = []  # x-coordinates of knots
                ycoords = []  # y-coordinates of knots
                for j in range(nknots):
                    # still unsure why y2 is in the file... why store if it's
                    # recalculated again later??
                    x, y, y2 = [np.float(el) for el in f.readline().split()]
                    xcoords.append(x)
                    ycoords.append(y)

                # TODO: think this is wrong; TiO.meam.splines is an anomaly
                # if (i < nphi+ntypes) or ((i >= nphi+2*ntypes) and (
                #         i<nphi+3*ntypes)):
                #     bc_type = ((), (1,0))

                bc_type = ((1, d0), (1, dN))
                splines.append(Spline(xcoords, ycoords, bc_type=bc_type,
                                      end_derivs=(d0, dN)))

            return cls(splines, types)

        except IOError:
            print("Potential file does not exist")
            sys.exit()

    def compute_energy(self, atoms):

        """Evaluates the energies for the given system using the MEAM potential,
        following the notation specified in the LAMMPS spline/meam documentation
        Args:
            atoms (Atoms):
                ASE Atoms object containing all atomic information

        Note that splines are indexed as follows:
            phis, gs: 11,12,...,1N,22,...,2N,...,NN
            rhos, us, fs: 1,2,...,N

            e.g. for Ti-O where Ti is atom type 1, O atom type 2
                phis, gs: Ti-Ti, Ti-O, O-O
                rhos, us, fs: Ti, O
        """

        # nl allows double-counting of bonds, nl_noboth does not
        nl = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2),
                          self_interaction=False, bothways=True, skin=0.0)

        nl_noboth = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2),
                                 self_interaction=False, bothways=False,
                                 skin=0.0)
        nl.build(atoms)
        nl_noboth.build(atoms)

        total_pe = 0.0
        natoms = len(atoms)
        self.energies = np.zeros((natoms,))
        self.uprimes = [None] * natoms

        for i in range(natoms):
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            # Pull atom-specific neighbor lists
            neighbors = nl.get_neighbors(i)
            neighbors_noboth = nl_noboth.get_neighbors(i)

            num_neighbors = len(neighbors[0])
            num_neighbors_noboth = len(neighbors_noboth[0])

            # Build the list of shifted positions for atoms outside of unit cell
            neighbor_shifted_positions = []

            #    neighbor_shifted_positions.append(neigh_pos)
            indices, offsets = nl.get_neighbors(i)
            for idx, offset in zip(indices, offsets):
                neigh_pos = atoms.positions[idx] + np.dot(offset,
                                                          atoms.get_cell())
                neighbor_shifted_positions.append(neigh_pos)
            # end shifted positions loop

            # TODO: workaround for this if branch; do we even need it??
            if len(neighbors[0]) > 0:
                tripcounter = 0
                total_phi = 0.0
                total_ni = 0.0

                u = self.us[i_to_potl(itype)]

                # Calculate pair interactions (phi)
                for j in range(num_neighbors_noboth):
                    jtype = lammpsTools.symbol_to_type(
                        atoms[neighbors_noboth[0][j]].symbol, self.types)

                    r_ij = np.linalg.norm(
                        ipos - neighbor_shifted_positions[j])

                    phi = self.phis[ij_to_potl(itype, jtype, self.ntypes)]

                    total_phi += phi(r_ij)
                # end phi loop

                for j in range(num_neighbors):
                    jtype = lammpsTools.symbol_to_type(
                        atoms[neighbors[0][j]].symbol, self.types)
                    r_ij = np.linalg.norm(
                        ipos - neighbor_shifted_positions[j])

                    rho = self.rhos[i_to_potl(jtype)]
                    fj = self.fs[i_to_potl(jtype)]

                    # Used for triplet calculations
                    a = neighbor_shifted_positions[j] - ipos
                    na = np.linalg.norm(a)

                    partialsum = 0.0
                    for k in range(j, num_neighbors):
                        if k != j:
                            ktype = lammpsTools.symbol_to_type(
                                atoms[neighbors[0][k]].symbol, self.types)

                            r_ik = np.linalg.norm(ipos -
                                                  neighbor_shifted_positions[k])

                            b = neighbor_shifted_positions[k] - ipos

                            fk = self.fs[i_to_potl(ktype)]
                            g = self.gs[
                                ij_to_potl(jtype, ktype, self.ntypes)]

                            nb = np.linalg.norm(b)

                            cos_theta = np.dot(a, b) / na / nb

                            fk_val = fk(r_ik)
                            g_val = g(cos_theta)

                            partialsum += fk_val * g_val
                            tripcounter += 1
                    # end triplet loop

                    fj_val = fj(r_ij)
                    total_ni += fj_val * partialsum
                    total_ni += rho(r_ij)
                # end u loop

                atom_e = total_phi + u(total_ni) - self.zero_atom_energies[
                    i_to_potl(itype)]

                self.energies[i] = atom_e
                total_pe += atom_e

                self.uprimes[i] = u(total_ni, 1)
            # end atom loop

        return total_pe

    def compute_forces(self, atoms):
        """Evaluates the energies for the given system using the MEAM potential,
        following the notation specified in the LAMMPS spline/meam documentation
        Args:
            atoms (Atoms):
                ASE Atoms object containing all atomic information

        Note that splines are indexed as follows:
            phis, gs: 11,12,...,1N,22,...,2N,...,NN
            rhos, us, fs: 1,2,...,N

            e.g. for Ti-O where Ti is atom type 1, O atom type 2
                phis, gs: Ti-Ti, Ti-O, O-O
                rhos, us, fs: Ti, O
        """

        self.compute_energy(atoms)

        nl = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2),
                          self_interaction=False, bothways=True, skin=0.0)

        nl_noboth = NeighborList(np.ones(len(atoms)) * (self.cutoff / 2),
                                 self_interaction=False, bothways=False,
                                 skin=0.0)
        nl.build(atoms)
        nl_noboth.build(atoms)

        natoms = len(atoms)

        self.forces = np.zeros((natoms, 3))
        cellx, celly, cellz = atoms.get_cell()

        for i in range(natoms):
            itype = lammpsTools.symbol_to_type(atoms[i].symbol, self.types)
            ipos = atoms[i].position

            Uprime_i = self.uprimes[i]

            # Pull atom-specific neighbor lists
            neighbors = nl.get_neighbors(i)
            neighbors_noboth = nl_noboth.get_neighbors(i)

            num_neighbors = len(neighbors[0])
            num_neighbors_noboth = len(neighbors_noboth[0])

            # Build the list of shifted positions for atoms outside of unit cell
            neighbor_shifted_positions = []
            for l in range(num_neighbors):
                shiftx, shifty, shiftz = neighbors[1][l]
                neigh_pos = atoms[neighbors[0][l]].position + shiftx * cellx\
                            + shifty * celly + shiftz * cellz

                neighbor_shifted_positions.append(neigh_pos)
            # end shifted positions loop

            forces_i = np.zeros((3,))
            if len(neighbors[0]) > 0:
                for j in range(num_neighbors):
                    jtype = lammpsTools.symbol_to_type(
                        atoms[neighbors[0][j]].symbol, self.types)

                    jpos = neighbor_shifted_positions[j]
                    jdel = jpos - ipos
                    r_ij = np.linalg.norm(jdel)
                    jdel /= r_ij

                    fj_val = self.fs[i_to_potl(jtype)](r_ij)
                    fj_prime = self.fs[i_to_potl(jtype)](r_ij, 1)

                    forces_j = np.zeros((3,))

                    # Used for triplet calculations
                    a = neighbor_shifted_positions[j] - ipos
                    na = np.linalg.norm(a)

                    for k in range(j, num_neighbors):
                        if k != j:
                            ktype = lammpsTools.symbol_to_type(
                                atoms[neighbors[0][k]].symbol, self.types)
                            kpos = neighbor_shifted_positions[k]
                            kdel = kpos - ipos
                            r_ik = np.linalg.norm(kdel)
                            kdel /= r_ik

                            b = neighbor_shifted_positions[k] - ipos
                            nb = np.linalg.norm(b)

                            cos_theta = np.dot(a, b) / na / nb

                            fk_val = self.fs[i_to_potl(ktype)](r_ik)
                            g_val = self.gs[
                                ij_to_potl(jtype, ktype, self.ntypes)
                            ](cos_theta)

                            fk_prime = self.fs[i_to_potl(ktype)](r_ik, 1)
                            g_prime = self.gs[ij_to_potl(jtype, ktype,
                                                         self.ntypes)](
                                cos_theta, 1)

                            fij = -Uprime_i * g_val * fk_val * fj_prime
                            fik = -Uprime_i * g_val * fj_val * fk_prime

                            prefactor = Uprime_i * fj_val * fk_val * g_prime
                            prefactor_ij = prefactor / r_ij
                            prefactor_ik = prefactor / r_ik
                            fij += prefactor_ij * cos_theta

                            fik += prefactor_ik * cos_theta

                            fj = jdel * fij - kdel * prefactor_ij
                            forces_j += fj

                            fk = kdel * fik - jdel * prefactor_ik
                            forces_i -= fk

                            self.forces[neighbors[0][k]] += fk
                    # end triplet loop

                    self.forces[i] -= forces_j
                    self.forces[neighbors[0][j]] += forces_j
                # end pair loop

                self.forces[i] += forces_i

                # Calculate pair interactions (phi)
                for j in range(
                        num_neighbors_noboth):  # j = index for neighbor list
                    jtype = lammpsTools.symbol_to_type(
                        atoms[neighbors_noboth[0][j]].symbol, self.types)
                    jpos = neighbor_shifted_positions[j]
                    jdel = jpos - ipos
                    r_ij = np.linalg.norm(jdel)

                    rho_prime_i = self.rhos[i_to_potl(itype)](r_ij, 1)
                    rho_prime_j = self.rhos[i_to_potl(jtype)](r_ij, 1)

                    fpair = rho_prime_j * self.uprimes[i] + rho_prime_i * \
                        self.uprimes[neighbors_noboth[0][j]]

                    phi_prime = self.phis[ij_to_potl(itype, jtype,
                                                     self.ntypes)](r_ij, 1)

                    fpair += phi_prime
                    fpair /= r_ij

                    self.forces[i] += jdel * fpair
                    self.forces[neighbors_noboth[0][j]] -= jdel * fpair

                # end phi loop

            # end atom loop

        return self.forces

    def get_lammps_results(self, struct):

        types = ['H', 'He']

        params = {'units': 'metal', 'boundary': 'p p p', 'mass': ['1 1.008',
                                                                  '2 4.0026'],
                  'pair_style': 'meam/spline',
                  'pair_coeff': ['* * test.meam.spline ' + ' '.join(types)],
                  'newton': 'on'}

        self.write_to_file('test.meam.spline')

        calc = LAMMPS(no_data_file=True, parameters=params,
                      keep_tmp_files=False, specorder=types,
                      files=['test.meam.spline'])

        energy = calc.get_potential_energy(struct)
        forces = calc.get_forces(struct)

        calc.clean()
        os.remove('test.meam.spline')

        results = {'energy': energy, 'forces': forces}

        return results

    def write_to_file(self, fname):
        """Writes the potential to a file

        Args:
            fname (str):
                name of file to write to

        Returns:
            None; writes to file with name <fname>
        """

        types = self.types
        ntypes = len(types)

        with open(fname, 'w') as f:
            # Write header lines
            f.write("# meam/spline potential parameter file produced by MEAM "
                    "object\n")

            f.write("meam/spline %d %s\n" % (ntypes, " ".join(types)))

            def write_spline(s):

                # Write additional spline info
                nknots = len(s.x)
                f.write("spline3eq\n")
                f.write("%d\n" % nknots)

                d0 = s(s.x[0], 1)
                dN = s(s.x[-1], 1)

                str1 = ("%.16f" % d0).rstrip('0').rstrip('.')
                str2 = ("%.16f" % dN).rstrip('0').rstrip('.')
                f.write(str1 + ' ' + str2 + '\n')

                # Write knot info
                for i in range(nknots):
                    str1 = ("%.16f" % s.x[i]).rstrip('0').rstrip('.')
                    str2 = ("%.16f" % s(s.x[i])).rstrip('0').rstrip('.')
                    str3 = ("%.16f" % s(s.x[i], 2)).rstrip('0').rstrip('.')
                    f.write(str1 + ' ' + str2 + ' ' + str3 + '\n')

            # Output all splines
            for fxn in self.phis:
                write_spline(fxn)
            for fxn in self.rhos:
                write_spline(fxn)
            for fxn in self.us:
                write_spline(fxn)
            for fxn in self.fs:
                write_spline(fxn)
            for fxn in self.gs:
                write_spline(fxn)

    def plot(self, fname=None):
        """Generates plots of all splines"""

        splines = self.phis + self.rhos + self.us + self.fs + self.gs

        for i, s in enumerate(splines):
            s.plot(saveName=fname + str(i + 1) + '.png')
            # TODO: finish this for generic system, not just binary/unary

    def phionly_subtype(self):
        """Returns the phionly version of pot

        Returns:
            phionly (MEAM):
                the phionly version of pot
        """

        N = self.ntypes  # number of components in the system
        nphi = int((N + 1) * N / 2)  # number of each phi and u splines

        original = self.phis + self.rhos + self.us + self.fs + self.gs

        splines = [original[i] if (i < nphi) else ZeroSpline(original[i].x)
                   for i in range(N * (N + 4))]

        return MEAM(splines=splines, types=self.types)

    def nophi_subtype(self):
        """Returns the nophi version of pot

        Args:
            self (MEAM):
                the original potential

        Returns:
            nophi (MEAM):
                the nophi version of pot
        """

        N = self.ntypes  # number of components in the system
        nphi = int((N + 1) * N / 2)  # number of each phi and u splines

        original = self.phis + self.rhos + self.us + self.fs + self.gs

        splines = [original[i] if (i >= nphi) else ZeroSpline(original[i].x)
                   for i in range(N * (N + 4))]

        return MEAM(splines=splines, types=self.types)

    def rhophi_subtype(self):
        """Returns the rhophi version of pot

        Args:
            self (MEAM):
                the original potential

        Returns:
            rhophi (MEAM):
                the rhophi version of pot
        """

        N = self.ntypes  # number of components in the system
        nphi = int((N + 1) * N / 2)  # number of each phi and u splines

        original = self.phis + self.rhos + self.us + self.fs + self.gs

        splines = [original[i] if (i < (nphi + N + N)) else ZeroSpline(original[
                                                                           i].x)
                   for i in range(N * (N + 4))]

        return MEAM(splines=splines, types=self.types)

    def norhophi_subtype(self):
        """Returns the norhophi version of pot

        Args:
            self (MEAM):
                the original potential

        Returns:
            norhophi (MEAM):
                the norhophi version of pot
        """

        N = self.ntypes  # number of components in the system
        nphi = int((N + 1) * N / 2)  # number of each phi and u splines

        original = self.phis + self.rhos + self.us + self.fs + self.gs

        splines = [original[i] if (i >= (nphi + N)) else ZeroSpline(original[
                                                                        i].x)
                   for i in range(N * (N + 4))]

        return MEAM(splines=splines, types=self.types)

    def norho_subtype(self):
        """Returns the norhos version of pot

        Args:
            self (MEAM):
                the original potential

        Returns:
            norhos (MEAM):
                the norhos version of pot
        """

        N = self.ntypes  # number of components in the system
        nphi = int((N + 1) * N / 2)  # number of each phi and u splines

        original = self.phis + self.rhos + self.us + self.fs + self.gs

        splines = [
            original[i] if ((i < nphi) or (i >= (nphi + N))) else ZeroSpline(
                original[i].x) for i in range(N * (N + 4))]

        return MEAM(splines=splines, types=self.types)

    def rho_subtype(self):
        """Returns the rhos version of pot

        Args:
            self (MEAM):
                the original potential

        Returns:
            rhos (MEAM):
                the rhos version of pot
        """

        N = self.ntypes  # number of components in the system
        nphi = int((N + 1) * N / 2)  # number of each phi and u splines

        original = self.phis + self.rhos + self.us + self.fs + self.gs

        splines = [
            original[i] if ((i >= nphi) and (i < nphi + N + N)) else ZeroSpline(
                original[i].x) for i in range(N * (N + 4))]

        return MEAM(splines=splines, types=self.types)

    def nog_subtype(self):
        """Returns the nog version of pot

        Args:
            self (MEAM):
                the original potential

        Returns:
            nog (MEAM):
                the nog version of pot
        """

        N = self.ntypes  # number of components in the system
        nphi = int((N + 1) * N / 2)  # number of each phi and u splines

        original = self.phis + self.rhos + self.us + self.fs + self.gs

        splines = [
            original[i] if ((i >= nphi) and (i < nphi + N + N)) else ZeroSpline(
                original[i].x) for i in range(N * (N + 4))]

        return MEAM(splines=splines, types=self.types)


def ij_to_potl(itype, jtype, ntypes):
    """Maps i and j element numbers to a single index of a 1D list; used for
    indexing spline functions. Currently only works for binary systems;
    ternary systems causes overlap with current algorithm (e.g. 2-1 bond maps to
    same as 1-3 bond)
    
    Args:
        itype (int):
            the i-th element in the system (e.g. in Ti-O, Ti=1)
        jtype (int):
            the j-the element in the system (e.g. in Ti-O, O=2)
        ntypes (int):
            the number of unique element types in the system
        
    Returns:
        The mapping of ij into an index of a 1D 0-indexed list
    """

    if (itype < 1) or (jtype < 1):
        raise ValueError("atom types must be positive and non-zero")
    elif ntypes != 2:
        # remove the unit test in meamTests.py once you implement this
        raise NotImplementedError("currently, only binary systems are "
                                  "supported")
    else:
        return int(jtype - 1 + (itype - 1) * ntypes - (itype - 1) * itype / 2)


def i_to_potl(itype):
    """Maps element number i to an index of a 1D list; used for indexing spline
    functions. Taken directly from pair_meam_spline.h
    
    Args:
        itype (int):
            the i-the element in the system (e.g. in Ti-O, Ti=1)
        
    Returns:
        The array index for the given element
    """

    if itype < 1:
        raise ValueError("atom types must be positive and non-zero")
    else:
        return int(itype - 1)


def splines_to_pvec(splines):
    """Converts a list of splines into a large 1D vector of parameters.
    Spline ordering is unchanged.

    Note:
        this formatting for the output was chosen to correspond with how a
        Worker object is built and evaluated

    Args:
        splines (list [Spline]):
            an ordered list of Spline objects

    Returns:
        x_pvec (np.arr):
            a group of parameters corresponding to a single spline consists
            of the x-positions of all knot points

        y_pvec (np.arr):
            y-positions of all knot points plus two additional boundary
            conditions

        x_indices (list [int]):
            indices deliminating spline knots
    """

    x_pvec = np.array([])
    y_pvec = np.array([])
    x_indices = []

    idx_tracker = 0
    for s in splines:
        x_pvec = np.append(x_pvec, s.x)

        der = []
        for i in range(len(s.x)):
            y_pvec = np.append(y_pvec, s(s.x[i]))
            der.append(s(s.x[i], 1))

        bc = [der[0], der[-1]]

        y_pvec = np.append(y_pvec, bc)

        x_indices.append(idx_tracker)
        idx_tracker += len(s.x)

    return x_pvec, y_pvec, x_indices


def splines_from_pvec(x_pvec, y_pvec, x_indices):
    """Builds splines out of the given knot coordinates and boundary
    conditions.

    Note:
        this formatting for the output was chosen to correspond with how a
        Worker object is built and evaluated

    Args:
        x_pvec (np.arr):
            a group of parameters corresponding to a single spline consists
            of the x-positions of all knot points as well as two additional
            boundary conditions

        y_pvec (np.arr):
            y-positions of all knot points

        x_indices (list [int]):
            indices deliminating spline parameters in vector

    Returns:
        splines (list [Spline]):
            an ordered list of Spline objects
    """

    x_indices = np.array(x_indices)
    y_indices = [x_indices[i - 1] - 2 * i for i in range(1, len(x_indices) + 1)]
    y_indices = np.array(y_indices)

    x_split = np.split(x_pvec, x_indices)
    y_split = np.split(y_pvec, y_indices)

    splines = []

    for i in range(len(x_split)):
        x_knots, bc = np.split(x_split[i], [-2])
        y_knots = y_split[i]

        splines.append(Spline(
            x_knots, y_knots, bc_type=((1, bc[0]), (1, bc[1])), end_derivs=bc))

    return splines


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import lammpsTools

    p = MEAM('TiO.meam.spline', ['Ti', 'O'])
    atoms = lammpsTools.atoms_from_file('Ti_only_crowd.Ti', ['Ti'])
