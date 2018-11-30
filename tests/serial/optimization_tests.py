"""
Intended to compare runtimes for various key functions:
    1) Outer products
    2) Indexing
    3) Einsum
"""

import os
import time
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

N = 128
M = 10

# vectors for outer products
v1 = v2 = v3 = np.arange(M)

# array, indices, and vec for indexing and einsum
A = np.random((N, N, M*M*M))
indices = np.arange(0, M*M*M, M // 10).tolist()
b = np.arange(M*M*M).tolist()

################################################################################
""" Define functions for Numba direct Python"""

def python_outer(u1, u2, u3):
    n1 = u1.shape[0]
    n2 = u2.shape[0]
    n3 = u3.shape[0]

    output = np.zeros(n1*n2*n3)

    for i in range(n1):
        u1_i = u1[i]
        for j in range(n2):
            u2_j = u2[j]
            ij = i*j
            for k in range(n3):
                output[ij*k] = u1_i*u2_j*u3[k]

    return output

def numpy_outer(u1, u2, u3):
    tmp = np.einsum('i,j', u1, u2).ravel()
    return np.einsum('i,j', tmp, u3).ravel()

numba_outer = jit(python_outer)

################################################################################
"""Run the tests"""

py_start = time.time()
python_outer(v1, v2, v3)
py_time = time.time() - py_start

np_start = time.time()
numpy_outer(v1, v2, v3)
np_time = time.time() - np_start

nb_start = time.time()
numba_outer(v1, v2, v3)
nb_time = time.time() - nb_start

################################################################################
"""Plot the results"""


