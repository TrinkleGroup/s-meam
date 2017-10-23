"""Tests bulk structures with vacuum"""
import unittest
import os
import lammpsTools

from ase.calculators.lammpsrun import LAMMPS
from .structs import bulk_vac, bulk_vac_rhombo
from .potentials import meams,nophis,phionlys,rhos,norhos,norhophis,rhophis
from .globalVars import DIGITS

class BulkVacTests(unittest.TestCase):

    def runner(self, pots):

        for p in pots:
     
            p.write_to_file('test.meam.spline')

            types = ['H','He']

            params = {}
            # TODO: all calc args should be in [] format, dummy
            params['boundary'] = 'p p p'
            params['mass'] =  ['1 1.008', '2 4.0026']
            params['pair_style'] = 'meam/spline'
            params['pair_coeff'] = ['* * test.meam.spline ' + '\
                    '.join(types)]
            params['newton'] = 'on'

            calc = LAMMPS(no_data_file=True, parameters=params,\
                    keep_tmp_files=False,specorder=types,files=['test.meam.spline'])

            for atoms in bulk_vac:
                expected = calc.get_potential_energy(atoms)
                val = p.compute_energies

                self.assertAlmostEquals(val, expected, DIGITS)

            for atoms in bulk_vac_rhombo:
                expected = calc.get_potential_energy(atoms)
                val = p.compute_energies

                self.assertAlmostEquals(val, expected, DIGITS)

            calc.clean()

    def test_meam(self):
        self.runner(meams)

    def test_nophis(self):
        self.runner(nophis)

    def test_phionlys(self):
        self.runner(phionlys)

    def test_rhos(self):
        self.runner(rhos)

    def test_norhos(self):
        self.runner(norhos)

    def test_norhophis(self):
        self.runner(norhophis)

    def test_rhophis(self):
        self.runner(rhophis)
