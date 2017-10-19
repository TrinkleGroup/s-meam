"""Testing suite for all structures and potentials"""
import nose
import unittest
import numpy as np

np.random.seed(42)

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

runner = unittest.TextTestRunner()
#runner = unittest.TextTestRunner()
#result = runner.run(suite)

config = nose.config.Config(verbosity=2, stopOnError=False)
result = nose.run(module=tests.workertests, config=config)



# TODO: NOT within machine precision; THINK precision is being affected by the
# read/write methods in ASE LAMMPS() calculator

# TODO: add warnings that LAMMPS must be pre-installed
# TODO: build LAMMPS results outside of individual tests; call from tests