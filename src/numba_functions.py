import numpy as np
from numba import jit
from numba.pycc import CC

cc = CC('fast')

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

    for i in range(N):
        v1 = arr1[:, i]

        for j in range(M):
            results[:, i*M + j] += v1*arr2[:,j]

    return results

# @jit(cache=True, nopython=True)
def outer_prod_1d_2vecs(u1, u2, n1, n2, output):

    for i in range(n1):
        in1 = i*n1
        u1_i = u1[i]
        for j in range(n2):
            output[in1 + j] = u1_i*u2[j]

    # return output

@jit(cache=True, nopython=True)
def outer_prod_1d(u1, u2, u3, n1, n2, n3, output):

    for i in range(n1):
        in1 = i*n1
        u1_i = u1[i]
        for j in range(n2):
            jn2 = j*n2
            u2_j = u2[j]
            for k in range(n3):
                output[n3*(i*n2 + j) + k] = u1_i*u2_j*u3[k]

    # return output

@jit
def jit_einsum(A, B):
    return np.einsum('zp,aiz->iap', A, B)

# @jit(nopython=True, cache=True)
# def contract_extend_end_transpose(S, forces, N, len_cart):
#     """Equivalent to einsum('zp,aiz->iap'); used for multiplying the scaled
#     energy structure vector with the forces, summing along one dimension,
#     and reshaping
#
#     Args:
#         S (np.ndarray): (Z x P) array; the scaled structure vector
#         forces (np.ndarray): (A x I x Z) array; evaluated forces, uncontracted
#         N (int): the number of atoms in the system
#         len_cart (int): the length of the parameter vector
#
#     Returns:
#         A (np.ndarray): (I x A x P) gradient of forces w.r.t parameters
#     """
#
#     A = np.zeros((N, 3, len_cart))
#
#     for i in np.arange(N):
#         for a in np.arange(3):
#             for p in np.arange(len_cart):
#                 for z in np.arange(N):
#                     A[i,a,p] += S[z,p]*forces[a,i,z]
#
#     return A

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
