import numpy as np
from cla_utils import solve_U


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k dimensional numpy array.
    :return Lk: an mxm dimensional numpy array.

    """
    L = np.identity(m)
    k = m - lvec.size
    L[k:, k-1] = -lvec
    return L


def LU_inplace(A, bl=0, bu=0):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array
    :param bl: an integer, the lower bandwidth for matrix A.
    Defaults to zero.
    :param bu: an integer, the upper bandwidth for matrix A.
    Defaults to zero.

    """
    m, _ = A.shape
    if not bl:
        bl = m
    if not bu:
        bu = m
    for k in range(m-1):
        idx1 = min(k+bl+1, m)
        idx2 = min(k+bu+1, m)
        temp_vector = A[k+1:idx1, k] / A[k, k]
        A[k+1:idx1, k:idx2] -= np.outer(temp_vector, A[k, k:idx2])
        A[k+1:idx1, k] = temp_vector


def solve_L(L, b, ones_diagonal=False, bl=0):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,...,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :param bl: an integer, the lower bandwidth for matrix A.
    Defaults to zero.
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m, _ = L.shape
    if len(b.shape) == 1:
        b = np.reshape(b, (b.shape[0], 1))
    _, k = b.shape
    x = np.zeros((m, k))
    if ones_diagonal:
        diagonal_entries = np.ones(m)
    else:
        diagonal_entries = L.diagonal()
    if bl:
        for i in range(m):
            j = max(0, i-bl)
            x[i, :] = (b[i, :] - L[i, j:i].dot(x[j:i, :])) / diagonal_entries[i]
    else:
        for i in range(m):
            x[i, :] = (b[i, :] - L[i, :i].dot(x[:i, :])) / diagonal_entries[i]
    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """
    m, _ = A.shape
    LU_inplace(A)
    U_Ainv = solve_L(A, np.identity(m), ones_diagonal=True)
    Ainv = solve_U(A, U_Ainv)
    return Ainv
