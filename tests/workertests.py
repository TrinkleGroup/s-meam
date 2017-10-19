import unittest
NOSE_PARAMETERIZED_NO_WARN=1
from nose_parameterized import parameterized
import numpy as np
import time
import pickle
import os
import logging

np.random.seed(42)

from workers import WorkerManyPotentialsOneStruct as worker
from ase.calculators.lammpsrun import LAMMPS
from .structs import allstructs
from .potentials import meams,nophis,phionlys,rhos,norhos,norhophis,rhophis
from .globalVars import ATOL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#def setUp(self):
#    self.start = time.time()
#
#def tearDown(self):
#    elapsed = time.time() - self.start
#    logger.info('{} ({}s)'.format(self.id(), round(elapsed,3)))

def runner(ptype):
    exec('pots = ' + ptype, globals())

    calculated = {}
    for s,atoms in enumerate(allstructs):
        name = atoms.get_chemical_formula() + '_' + str(s)

        w = worker(atoms,pots)
        calculated[name] = w.compute_energies()

    return calculated

def getLammpsResults():

    start = time.time()

    # Builds dictionaries where dict[i]['ptype'][j] is the energy or force matrix of
    # struct i using potential j for the given 'ptype', according to LAMMPS
    energies = {}
    forces = {}

    types = ['H','He']
    #types = ['He','H']

    params = {}
    params['boundary'] = 'p p p'
    params['mass'] =  ['1 1.008', '2 4.0026']
    params['pair_style'] = 'meam/spline'
    params['pair_coeff'] = ['* * test.meam.spline ' + ' '.join(types)]
    params['newton'] = 'on'

    #writeduration = 0
    calcduration = 0

    ptypes = ['meams', 'nophis', 'phionlys', 'rhos', 'norhos', 'norhophis', 'rhophis' ]

    #energies = {k:v for (k,v) in [(i.get_chemical_formula()+'_'+str(i),{}) for
    #                                                        i in allstructs]}

    for s,atoms in enumerate(allstructs):
        key = atoms.get_chemical_formula() + '_' + str(s)
        energies[key] = {}
        for ptype in ptypes:
            energies[key][ptype] = np.array([])

    temp = rhophis
    for ptype in ptypes:

        exec('pots = ' + ptype, globals())

        j = 0
        for p in pots:

            p.write_to_file('test.meam.spline')

            calc = LAMMPS(no_data_file=True, parameters=params, \
                          keep_tmp_files=False,specorder=types,files=['test.meam.spline'])

            # counter for unique keys
            for s,atoms in enumerate(allstructs):

                cstart = time.time()
                key = atoms.get_chemical_formula() + '_' + str(s)
                energies[key][ptype] = np.append(energies[key][ptype],
                                               calc.get_potential_energy(atoms))
                calcduration += float(time.time() - cstart)

                # TODO: LAMMPS runtimes are inflated due to ASE internal read/write
                # TODO: need to create shell script to get actual runtimes

            j += 1
            calc.clean()
            os.remove('test.meam.spline')

    #logging.info("Time spent writing potentials: {}s".format(round(writeduration,3)))
    #logging.info("Time spent calculating in LAMMPS: {}s".format(round(calcduration,3)))

    return calcduration, energies, forces

def load_energies():
    if os.path.isfile('lammps_energies.dat'):
        # TODO: check if energies matches number of potentials; else rebuild
        duration, energies = pickle.load(open('lammps_energies.dat', 'rb'))
        #cls.forces = pickle.load(open('lammps_forces.dat', 'r'))
    else:
        duration, energies,_  = getLammpsResults()

        pickle.dump((duration, energies), open('lammps_energies.dat', 'wb'))
        #pickle.dump(cls.forces, open('lammps_forces.dat', 'w'))

    #logger.info('LAMMPS build time (accounting for file writing time):({'
    #            '}s)'.format(round(duration,3)))

    return energies

def load_phionlys():
    energies = load_energies()

    calculated = runner('phionlys')

    tests = []
    for name in calculated.keys():
        test_name = 'test_phionly_' + name
        tests.append((test_name, calculated[name], energies[name]['phionlys']))

    return tests

@parameterized.expand(load_phionlys)
def test_phionly(name,a,b):
    np.testing.assert_allclose(a,b,atol=ATOL)

#def test_rhos(self):
#    self.runner('rhos')

#def test_norhos(self):
#    self.runner('norhos')

#def test_norhophis(self):
#    self.runner('norhophis')

#def test_rhophis(self):
#    self.runner('rhophis')

