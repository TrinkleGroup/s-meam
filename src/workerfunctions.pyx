import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def add_at(np.ndarray[DTYPE_t, ndim=2] A, indices,
        np.ndarray[DTYPE_t, ndim=2] values):
    """Adds values[i] to A[indices[i]]

    Args:
        A (np.arr):
            original matrix

        indices (list):
            list of indices specifying where to add values

        values (np.arr):
            array to be added; second dimensions of A and values should match

    Returns:
        A_new (np.arr):
            the updated matrix
        """

    cdef int ymax = values.shape[0]
    cdef int xmax = values.shape[1]
    #cdef np.ndarray B = A.copy().astype(DTYPE)
    cdef i, j

    for i in range(ymax):
        for j in range(xmax):

            A[indices[i]][j] += values[i][j]

    return A
