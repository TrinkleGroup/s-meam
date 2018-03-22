import numpy as np
from numba import jit
# from numba.pycc import CC
#
# cc = CC('fast')
#
# @jit(nopython=True)
# def jit_add_at_1D(indices, values, minlength):
#     xmax = values.shape[0]
#
#     A = np.zeros(minlength)
#
#     for i in range(xmax):
#             A[indices[i]] += values[i]
#
#     return A
#
# @jit(nopython=True)
# def jit_add_at_2D(indices, values, minlength):
#     ymax = values.shape[0]
#     xmax = 3
#
#     A = np.zeros((minlength, 3))
#
#     for i in range(ymax):
#         for j in range(xmax):
#             A[indices[i]][j] += values[i][j]
#
#     return A

# @cc.export('onepass_min_max', 'UniTuple(f8, 2)(f8[:])')
@jit(nopython=True, cache=True)
def onepass_min_max(a):
    min = a[0]
    max = a[0]

    for x in a[1:]:
        if x < min: min = x
        if x > max: max = x


    return min, max

# @cc.export('mat_vec_mult', 'f8[:](f8[:], i4[:], i4[:], f8[:], i4)')
@jit(nopython=True, cache=True)
def mat_vec_mult(val, row, col, vec, N):

    results = np.zeros(N)

    for i in range(N):
        for j in range(row[i], row[i+1]):
            results[i] += val[j]*vec[col[j]]

    return results

# if __name__ == '__main__':
#     cc.compile()
