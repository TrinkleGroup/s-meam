import numpy as np
from numba import jit

@jit(nopython=True)
def jit_add_at_1D(indices, values, minlength):
    xmax = values.shape[0]

    A = np.zeros(minlength)

    for i in range(xmax):
            A[indices[i]] += values[i]

    return A

@jit(nopython=True)
def jit_add_at_2D(indices, values, minlength):
    ymax = values.shape[0]
    xmax = 3

    A = np.zeros((minlength, 3))

    for i in range(ymax):
        for j in range(xmax):
            A[indices[i]][j] += values[i][j]

    return A

# numba_add_at_2D = jit(jit_add_at_2D)