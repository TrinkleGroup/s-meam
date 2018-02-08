import os
import numpy as np
import cProfile
import time
import datetime

import ase.build
from pympler.classtracker import ClassTracker

import meam
from spline import Spline
from meam import MEAM
from worker import Worker

allow_overwrite = True

output_msg = "5-ffg-call-np-matmul"

date = datetime.datetime.now()

mem_filename = "../data/profiling/" \
           "{0}-{1}-{2}_profiling-mem_{3}.txt".format(date.year, date.month,
                                                      date.day, output_msg)

stats_filename = "../data/profiling/" \
           "{0}-{1}-{2}_profiling-stats_{3}.dat".format(date.year, date.month,
                                                      date.day, output_msg)

if not allow_overwrite:
    if (os.path.isfile(mem_filename)
        or os.path.isfile(stats_filename)):

        raise FileExistsError("Change file name to avoid overwriting previous "
                              "results")

np.random.seed(42)

a0 = 5.5  # Potential cutoff distance
r0 = 2.8  # Default atomic spacing
vac = 2 * a0  # Vacuum size used in all directions for certain structures

start = time.time()

# Build test potential (grid knots, random y-values, random end_derivs)
print("Building test potential ... ", end="", flush=True)
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

print("done", flush=True)

# Build test structure (bulk 250 atoms, orthorhombic, periodic, mixed types)
print("Building test structure ... ", end="")

atoms = ase.build.bulk('H', crystalstructure='fcc', a=a0, orthorhombic=True)
atoms = atoms.repeat((5, 5, 5))
atoms.rattle()
atoms.center(vacuum=0)
atoms.set_pbc(True)
atoms.set_chemical_symbols(np.random.randint(1, 3, size=len(atoms)))

atoms.center(vacuum=vac)

print("{0} atoms ... ".format(len(atoms)), end="", flush=True)

print("done", flush=True)

# Set up memory tracker
print("Setting up tracker ... ", end="", flush=True)

tracker = ClassTracker(open(mem_filename, 'w'))
tracker.track_class(Worker)

print("done", flush=True)

# Perform worker force evaluation
print("Performing calculations ... ", end="", flush=True)

x_pvec, y_pvec, indices = meam.splines_to_pvec(potential.splines)
worker = Worker(atoms, x_pvec, indices, potential.types)

cProfile.run('_ = worker.compute_energies(y_pvec)', filename=stats_filename)
# cProfile.run('_ = worker.compute_forces(y_pvec)', filename=stats_filename)

print("done", flush=True)

tracker.create_snapshot()
tracker.stats.print_summary()

print("Total runtime ... {0}".format(time.time() - start))
