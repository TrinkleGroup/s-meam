import numpy as np
import os
import pickle

import ase.build

import src.meam
from src.spline import Spline
from src.meam import MEAM
from src.worker import Worker
from tests.testStructs import extra, bulk_periodic_rhombo

np.random.seed(42)

a0 = 5.5  # Potential cutoff distance
r0 = 2.8  # Default atomic spacing
vac = 2 * a0  # Vacuum size used in all directions for certain structures

# Build test potential (grid knots, random y-values, random end_derivs)
num_elements = 2
num_splines = num_elements*(num_elements + 4)
num_knots = 10

knots_x = np.linspace(0, a0, num=num_knots)

splines = []
for i in range(num_splines):
    knots_y = np.random.random(num_knots)

    d_0 = np.random.rand()
    d_N = np.random.rand()

    tmp_spline = Spline(knots_x, knots_y,
                        bc_type=((1, d_0), (1, d_N)),
                        end_derivs=(d_0, d_N))

    tmp_spline.cutoff = (knots_x[0], knots_x[len(knots_x) - 1])

    splines.append(tmp_spline)

potential = MEAM(splines=splines, types=['H', 'He'])

atoms = ase.build.bulk('H', crystalstructure='fcc', a=a0, orthorhombic=True)
atoms = atoms.repeat((5, 5, 5))
atoms.rattle()
atoms.center(vacuum=0)
atoms.set_pbc(True)
atoms.set_chemical_symbols(np.random.randint(1, 3, size=len(atoms)))

atoms.center(vacuum=vac)

# atoms = bulk_periodic_rhombo['bulk_periodic_rhombo_mixed']
# atoms = extra['8_atoms']

x_pvec, y_pvec, indices = src.meam.splines_to_pvec(potential.splines)

file_name = 'data/workers/line_prof_worker.pkl'

force_remake = False

if force_remake:
    worker = Worker(atoms, x_pvec, indices, potential.types)
    pickle.dump(worker, open(file_name, 'wb'))
else:
    if os.path.isfile(file_name):
        print("Loading worker from file")
        worker = pickle.load(open(file_name, 'rb'))
    else:
        worker = Worker(atoms, x_pvec, indices, potential.types)
        pickle.dump(worker, open(file_name, 'wb'))

n = len(y_pvec)

@profile
def call():
    worker.compute_energies(np.random.random(n))
    worker.compute_forces(np.random.random(n))

for i in range(10000):
    call()
