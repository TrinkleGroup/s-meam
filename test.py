"""Testing suite for all structures and potentials"""
import unittest
import numpy as np

import tests.dimertests
import tests.trimertests
import tests.bulkvactests
import tests.bulkperiodictests
import tests.workertests

loader = unittest.TestLoader()
suite = unittest.TestSuite()

#suite.addTests(loader.loadTestsFromModule(tests.dimertests))
#suite.addTests(loader.loadTestsFromModule(tests.trimertests))
#suite.addTests(loader.loadTestsFromModule(tests.bulkvactests))
#suite.addTests(loader.loadTestsFromModule(tests.bulkperiodictests))
suite.addTests(loader.loadTestsFromModule(tests.workertests))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# TODO: NOT within machine precision; THINK precision is being affected by the
# read/write methods in ASE LAMMPS() calculator

# TODO: add warnings that LAMMPS must be pre-installed
