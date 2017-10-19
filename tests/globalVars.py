"""Global testing parameters"""

a0 = 2.6    # Potential cutoff distance
r0 = a0/4   # Default atomic spacing
vac = 2*a0  # Vacuum size used in all directions for certain structures

DIGITS = 6  # Desired number of digits for test accuracy (assertAlmostEquals)
ATOL = 1e-5 # Absolute tolerance for test accuracy (np.assert_allclose)
