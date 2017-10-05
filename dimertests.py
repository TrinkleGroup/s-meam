import unittest
import os

from ase.calculators.lammpsrun import LAMMPS
from teststructs import dimers
from testpotentials import meams,nophis,phionlys,rhos,norhos,norhophis,rhophis

class DimerTests(unittest.TestCase):

    def test_meam(self):
        for atoms in dimers:
            for p in meams:
                p.write_to_file('test.meam.spline')

                mass = [str(el) for el in set(atoms.get_masses())]
                pbc = ['p' if atoms.pbc[i]==True else 'f' for i in range(3)]
                types = ['H','He']

                params = {}
                params['boundary'] = ' '.join(pbc)
                params['mass'] =  ['1 ' + str(mass[0]), '2 ' + str(mass[1])]
                params['pair_style'] = 'meam/spline'
                params['pair_coeff'] = ['* * test.meam.spline ' + '\
                        '.join(types)]
                params['newton'] = 'on'

                # rzm: forces from ASE; are they ordered and %.16f?

                calc = LAMMPS(no_data_file=True, parameters=params,\
                        keep_tmp_files=True,specorder=types,\
                        files=['test.meam.spline'], keep_alive=True)

                expected = calc.get_potential_energy(atoms)
                val = p.compute_energies(atoms)

                self.assertAlmostEquals(val, expected, 6)
