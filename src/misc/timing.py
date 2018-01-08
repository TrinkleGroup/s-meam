import numpy as np
import time
import logging
import sys
import tests.potentials

from meam import MEAM
from workers import WorkerManyPotentialsOneStruct as worker
from ase.calculators.lammpsrun import LAMMPS
from tests.structs import allstructs
from tests.globalVars import ATOL

def runner(pots, structs):

    global py_calcduration

    calculated = {}
    for name in structs.keys():
        atoms = structs[name]

        start = time.time()
        w = worker(atoms,pots)
        calculated[name] = w.compute_energies(pots)
        py_calcduration += time.time() - start

    return calculated

def getLammpsResults(pots, structs):

    start = time.time()

    # Builds dictionaries where dict[i]['ptype'][j] is the energy or force matrix of
    # struct i using potential j for the given 'ptype', according to LAMMPS
    energies = {}
    forces = {}

    types = ['H','He']
    #types = ['He','H']

    params = {}
    params['units'] = 'metal'
    params['boundary'] = 'p p p'
    params['mass'] =  ['1 1.008', '2 4.0026']
    params['pair_style'] = 'meam/spline'
    params['pair_coeff'] = ['* * test.meam.spline ' + ' '.join(types)]
    params['newton'] = 'on'

    global lammps_calcduration

    for key in structs.keys():
        energies[key] = np.zeros(len(pots))

    for pnum,p in enumerate(pots):

        p.write_to_file('test.meam.spline')

        calc = LAMMPS(no_data_file=True, parameters=params, \
                      keep_tmp_files=False,specorder=types,files=[
                'test.meam.spline'])

        for name in structs.keys():
            atoms = structs[name]

            cstart = time.time()
            energies[name][pnum] = calc.get_potential_energy(atoms)
            lammps_calcduration += float(time.time() - cstart)

            # TODO: LAMMPS runtimes are inflated due to ASE internal read/write
            # TODO: need to create shell script to get actual runtimes

        calc.clean()
        #os.remove('test.meam.spline')

    #logging.info("Time spent writing potentials: {}s".format(round(writeduration,3)))
    #logging.info("Time spent calculating in LAMMPS: {}s".format(round(
    # calcduration,3)))

    return energies, forces

N = int(sys.argv[1])
#N = 1

potentials = tests.potentials.get_random_pots(N)['meams']
#potentials = MEAM("HHe.meam.spline")
#potentials = [potentials]*N

# rzm: timing tests; figure out how to deal with lammps_setup_time

key = 'bulk_periodic_ortho_mixed'
struct = {key:allstructs['bulk_periodic_ortho_mixed']}

lammps_calcduration = 0.0
py_calcduration     = 0.0

energies, _ = getLammpsResults(potentials, struct)
calculated  = runner(potentials, struct)

lammps_setup_time = 0.0025125
lammps_calcduration -= N*lammps_setup_time

res_lammps  = energies[key]
res_py      = calculated[key]
np.testing.assert_allclose(energies[key], calculated[key], atol=ATOL)

print("{0}\t{1}\t{2}".format(N, round(lammps_calcduration,3),
                             round(py_calcduration,3)))
