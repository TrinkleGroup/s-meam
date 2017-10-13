import os
import unittest
import lammpsTools
import numpy as np
import spline

from meam import MEAM
from spline import Spline
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS

#### Prepares global variables ####
###################################

class MEAMTestSuite(unittest.TestCase):
    """Testing object for MEAM potential. Uses a randomly generated potential
    for testing Python vs. LAMMPS calculated energies and forces. Uses an
    arbitrary binary system in various useful configurations including dimers,
    trimers, sc, fcc, bcc, hcp, etc. with and without vacuum
    
    LAMMPS results are stored in a dictionaries of dictionaries, orderd as
    
        energies = { potential : {subtype : [float for each struct]}}

        forces = { potential : {subtype : [np.arry for each struct]} }

    where 'potential' is the randomly generated potential, subtype is one of
    ['meam','nophi','phionly','rho','norho','norhophi','rhophi'], and
    'struct' is one of the pre-defined test structures form
    build_structures()."""

    class ResultsLAMMPS(object):
        """Helper object for organizing results of LAMMPS calculations"""

        def __init__(self,name,struct,potentials):
            self.name = name

            self.energies, self.forces = computeLAMMPS(struct, potentials)
            # TODO: need dimers/trimers/etc. to be these objects

        def computeLAMMPS(self,structs,potentials):
            """Computes energies and forces for the given structures using the given
            potentials.
            
            Args:
                structs (list[Atoms]):
                    the list of ASE Atoms objects defining the structures
                    
                potentials (list[MEAM]):
                    the list of MEAM objects defining the potentials
                
            Returns:
                energies (dict):
                    an ordered dictionary where energies[i][j] is the computed
                    energy of structure i using potential j (i,j are list indices
                    corresponding to 'structs' and 'potentials')
                    
                forces (dict):
                    an dictionary of computed forces, ordered in the same way as
                    'energies'"""

            energies = {}
            forces = {}

            for p in potentials:
                for s in structs:
                    self.meam = MEAM(splines=splines)
                    self.meam.write_to_file('test.meam.spline')

                    mass = [str(el) for el in set(atoms.get_masses())]
                    pbc = ['p' if atoms.pbc[i]==True else 'f' for i in range(3)]
                    types = ['H','He']

                    params = {}
                    params['boundary'] = ' '.join(pbc)
                    params['mass'] =  ' '.join(mass)
                    params['pair_style'] = 'meam/spline'
                    params['pair_coeff'] = '* * test.meam.spline ' + ' '.join(types)
                    params['newton'] = 'on'

                    calc = LAMMPS(no_data_file=True, parameters=params,\
                            keep_tmp_files=False,specorder=types,\
                            tmp_dir=os.path.join('tmp',fname),files=[meam], keep_alive=True)

        @property
        def results(self, i):
            """Returns the full results of the i-th potential, where the results
            are ordered as ['meam','nophi','phionly','rho','norho','norhophi',
            'rhophi']"""
            return self._results[i]

    def setUp(self):
        # TODO: generate N MEAM potentials and specific testing structs.
        # Produce results from LAMMPS. Each test case will compare with a
        # different worker

        N = 100

        # Random potential cutoff in range [1,10]; atomic distances 
        # Default atomic spacing is set as a0/2
        # Default vacuum is a0*2
        a0 = 1 + np.random.rand()*9

        self.build_structures(a0)

        for n in N:
            # Generate splines with 5-10 knots, random y-coords of knots, equally
            # spaced x-coords ranging from 0 to a0, random d0 dN
            splines = []
            for i in range(12): # 2-component system has 12 total splines
                num_knots = np.random.randint(low=5,high=11)

                knot_x = np.linspace(0,a0, num=num_knots)
                knots_y = np.random.random(num_knots)

                d0 = np.random.rand()
                dN = np.random.rand()

                temp = Spline(knot_x, knot_y, bc_type=((1,d0),(1,dN)),\
                        derivs=(d0,dN))

                temp.cutoff = (knot_x[0],knot_x[len(knot_x)-1])
                splines.append(temp)

            self.potentials[n] = MEAM(splines=splines, types=['H','He'])

        # TODO: meam, nophi, phionly, rho, norho, norhophi, rhophi

    def tearDown(self):
        self.struct.dispose()
        self.struct = None

        self.meam.dispose()
        self.meam = None

    def test_energies_meam(self):
        pass

    def test_forces_meam(self):
        pass

    def test_meam(self):
        """Tests all potentials for all structs"""

        for s in self.structs:
            for p in self.potentials:
                pass
                
    def test_dimers(self):
        """Tests dimers for all potentials"""

        for s in self.dimers:
            for p in 

