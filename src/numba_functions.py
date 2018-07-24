import numpy as np
from numba import jit
from numba.pycc import CC

cc = CC('fast')
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

#@jit(nopython=True, cache=True)
@jit(nopython=True, cache=True)
def outer_prod(arr1, arr2):
    n_pots = arr1.shape[0]

    N = arr1.shape[1]
    M = arr2.shape[1]

    results = np.zeros((n_pots, N*M))

    # for p in range(n_pots):
    #     for i in range(N):
    #         a = arr1[p, i]
    #
    #         for j in range(M):
    #             results[p, i*M + j] += a*arr2[p, j]
    for i in range(N):
        v1 = arr1[:, i]

        for j in range(M):
            results[:, i*M + j] += v1*arr2[:,j]

    return results

@jit(cache=True)
def outer_prod_simple(arr1, arr2):
    """Normal outer product; not for multiple potentials"""
    N = arr1.shape[0]
    M = arr2.shape[0]

    results = np.zeros(N*M)

    for i in range(N):
        a = arr1[i]

        for j in range(M):
            results[i*M + j] += a*arr2[j]

    return results

@jit
def jit_einsum(A, B):
    return np.einsum('zp,aiz->iap', A, B)

@jit(nopython=True, cache=True)
def contract_extend_end_transpose(S, forces, N, len_cart):
    """Equivalent to einsum('zp,aiz->iap'); used for multiplying the scaled
    energy structure vector with the forces, summing along one dimension,
    and reshaping

    Args:
        S (np.ndarray): (Z x P) array; the scaled structure vector
        forces (np.ndarray): (A x I x Z) array; evaluated forces, uncontracted
        N (int): the number of atoms in the system
        len_cart (int): the length of the parameter vector

    Returns:
        A (np.ndarray): (I x A x P) gradient of forces w.r.t parameters
    """

    A = np.zeros((N, 3, len_cart))

    for i in np.arange(N):
        for a in np.arange(3):
            for p in np.arange(len_cart):
                for z in np.arange(N):
                    A[i,a,p] += S[z,p]*forces[a,i,z]

    return A

# @jit(nopython=True, cache=True)
@cc.export('mat_vec_mult', 'f8[:](f8[:], i4[:], i4[:], f8[:], i4)')
def mat_vec_mult(val, row, col, vec, N):

    results = np.zeros(N)

    for i in range(N):
        for j in range(row[i], row[i+1]):
            results[i] += val[j]*vec[col[j]]

    return results

if __name__ == '__main__':
    cc.compile()
