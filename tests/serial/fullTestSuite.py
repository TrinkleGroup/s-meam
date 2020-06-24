"""Testing suite for all structures and potentials"""
import nose
import os
import unittest
import numpy as np

# config = nose.config.Config(verbosity=2, stopOnError=False)
# nose.run(config=config)

# np.random.seed(42)
#
# import tests.splineTests
# import tests.meamTests
# import tests.workertests
#
# loader = unittest.TestLoader()
# suite = unittest.TestSuite()
#
# suite.addTests(loader.loadTestsFromModule(tests.workertests))
# suite.addTests(loader.loadTestsFromModule(tests.splineTests))
# suite.addTests(loader.loadTestsFromModule(tests.meamTests))
#
# runner = unittest.TextTestRunner()
# runner = unittest.TextTestRunner()
# result = runner.run(suite)
#
# config = nose.config.Config(verbosity=2, stopOnError=False)
# result = nose.run(module=tests.splineTests, config=config)
# result = nose.run(module=tests.meamTests, config=config)

# TODO: NOT within machine precision; THINK precision is being affected by the
# read/write methods in ASE LAMMPS() calculator

# TODO: add warnings that LAMMPS must be pre-installed
# TODO: build LAMMPS results outside of individual tests; call from tests

if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    # tests_path = os.path.join(os.path.abspath(os.path.dirname(file_path)), "tests")
    tests_path = file_path
    config = nose.config.Config(verbosity=2, stopOnError=False, withCov=True)
    # result = nose.run(argv=[os.path.abspath(__file__),
    #                         "--with-cov", "--verbosity=3",
    #                         "--cover-package=tests", tests_path])
    result = nose.run(config=config)
