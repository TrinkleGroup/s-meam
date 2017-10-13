import unittest
import numpy as np
import time
import pickle
import os
import logging

from workers import WorkerManyPotentialsOneStruct as worker
from ase.calculators.lammpsrun import LAMMPS
from .structs import allstructs
from .potentials import meams,nophis,phionlys,rhos,norhos,norhophis,rhophis, N
from .globalVars import ATOL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkerManyPotentialsOneStructTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        if os.path.isfile('lammps_energies.dat'):
            duration, cls.energies = pickle.load(open('lammps_energies.dat', 'rb'))
            #cls.forces = pickle.load(open('lammps_forces.dat', 'r'))
        else:
            duration, cls.energies,_  = getLammpsResults()

            pickle.dump((duration, cls.energies), open('lammps_energies.dat', 'wb'))
            #pickle.dump(cls.forces, open('lammps_forces.dat', 'w'))

        logger.info('LAMMPS build time (accounting for file writing time): ({}s)'.format(round(duration,3)))

    def setUp(self):
        self.start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.start
        logger.info('{} ({}s)'.format(self.id(), round(elapsed,3)))

    def runner(self, ptype):
        exec('pots = ' + ptype, globals())

        #pots[0].plot()
        #print(len(pots))
        #for p in pots:
        #    p.plot()

        for i,atoms in enumerate(allstructs):
            w = worker(atoms,pots)
            allVals = w.compute_energies()

            np.testing.assert_allclose(allVals, self.energies[i][ptype], atol=ATOL)

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

def getLammpsResults():
    # TODO: store with json; only rebuild if necessary

    start = time.time()

    # Builds dictionaries where dict[i]['ptype'][j] is the energy or force matrix of
    # struct i using potential j for the given 'ptype', according to LAMMPS
    energies = {}
    forces = {}

    types = ['H','He']

    params = {}
    params['boundary'] = 'p p p'
    params['mass'] =  ['1 1.008', '2 4.0026']
    params['pair_style'] = 'meam/spline'
    params['pair_coeff'] = ['* * test.meam.spline ' + ' '.join(types)]
    params['newton'] = 'on'

    #writeduration = 0
    calcduration = 0

    ptypes = ['meams', 'nophis', 'phionlys', 'rhos', 'norhos', 'norhophis', 'rhophis' ]

    energies = {k:v for (k,v) in [(i,{}) for i in range(len(allstructs))]}

    for i in range(len(allstructs)):
        for ptype in ptypes:
            energies[i][ptype] = np.array([])

    for ptype in ptypes:

        exec('pots = ' + ptype, globals())

        j = 0
        for p in pots:

            p.write_to_file('test.meam.spline')

            calc = LAMMPS(no_data_file=True, parameters=params, \
                          keep_tmp_files=False,specorder=types,files=['test.meam.spline'])

            for i,atoms in enumerate(allstructs):

                cstart = time.time()
                energies[i][ptype] = np.append(energies[i][ptype], calc.get_potential_energy(atoms))
                calcduration += float(time.time() - cstart)

                # TODO: LAMMPS runtimes are inflated due to ASE internal read/write
                # TODO: need to create shell script to get actual runtimes

            j += 1
            calc.clean()

    #logging.info("Time spent writing potentials: {}s".format(round(writeduration,3)))
    #logging.info("Time spent calculating in LAMMPS: {}s".format(round(calcduration,3)))

    return calcduration, energies, forces
