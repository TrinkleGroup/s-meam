import unittest
import os
import lammpsTools
import numpy as np

from meam import MEAM
from workers import WorkerManyPotentialsOneStruct as worker
from ase.calculators.lammpsrun import LAMMPS
from structs import allstructs
from potentials import meams,nophis,phionlys,rhos,norhos,norhophis,rhophis
from globalVars import ATOL

class WorkerManyPotentialsOneStructTests(unittest.TestCase):

    def setUp(self):
        print("Building dictionaries of LAMMPS results")

        # Builds dictionaries where dict[i]['ptype'] is the energy or force matrix of
        # struct i using potential j, according to LAMMPS
        energies = {}
        forces = {}

        types = ['H','He']

        params = {}
        params['boundary'] = 'p p p'
        params['mass'] =  ['1 1.008', '2 4.0026']
        params['pair_style'] = 'meam/spline'
        params['pair_coeff'] = ['* * test.meam.spline ' + ' '.join(types)]
        params['newton'] = 'on'

        for i,atoms in enumerate(allstructs):
            energies[i] = {}

            for ptype in ['meams', 'nophis', 'phionlys', 'rhos', 'norhos',\
                    'norhophis', 'rhophis' ]:

                exec('pots = ' + ptype)
                energies[i][ptype] = np.zeros(len(pots))

                for j,p in enumerate(pots):
                    p.write_to_file('test.meam.spline')
                    calc = LAMMPS(no_data_file=True, parameters=params,\
                            keep_tmp_files=False,specorder=types,files=['test.meam.spline'])
                    energies[i][ptype][j] = calc.get_potential_energy(atoms)
                    calc.clean()

        self.energies = energies
        self.forces = forces

    def runner(self, ptype):
        print("Calculating with worker")

        exec('pots = ' + ptype)

        for i,atoms in enumerate(allstructs):

            w = worker(atoms,pots)
            allVals = w.compute_energies()

            
            print('expected', self.energies[i][ptype])
            print('computed', allVals)
            #np.testing.assert_allclose(allVals, self.energies[i], atol=ATOL)

        calc.clean()

    #def test_meam(self):
    #    self.runner('meams')

    #def test_nophis(self):
    #    self.runner('nophis')

    def test_phionlys(self):
        self.runner('phionlys')

    #def test_rhos(self):
    #    self.runner('rhos')

    #def test_norhos(self):
    #    self.runner('norhos')

    #def test_norhophis(self):
    #    self.runner('norhophis')

    #def test_rhophis(self):
    #    self.runner('rhophis')
