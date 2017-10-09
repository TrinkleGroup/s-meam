"""Testing suite for all structures and potentials"""
import unittest
import numpy as np

import dimertests
import trimertests
import bulkvactests
import bulkperiodictests

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(dimertests))
suite.addTests(loader.loadTestsFromModule(trimertests))
suite.addTests(loader.loadTestsFromModule(bulkvactests))
suite.addTests(loader.loadTestsFromModule(bulkperiodictests))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# TODO: NOT within machine precision; THINK precision is being affected by the
# read/write methods in ASE LAMMPS() calculator
