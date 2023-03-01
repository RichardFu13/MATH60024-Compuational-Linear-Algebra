import numpy as np
from cla_utils import solve_U, solve_L

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """

    p[i], p[j] = p[j], p[i]


def LUP_inplace(A, return_swaps = False):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """
                     
    m, _ = A.shape
    p = np.arange(m)
    swap_counter = 0
    for k in range(m-1):
        i = np.argmax(np.abs(A[k:, k])) + k
        if i != k and return_swaps:
            swap_counter += 1
        A[[k, i]] = A[[i, k]]
        perm(p, i, k)
        temp_vector = A[k+1:, k] / A[k, k]
        A[k+1:, k:] -= np.outer(temp_vector, A[k, k:])
        A[k+1:, k] = temp_vector
    if return_swaps:
        return p, swap_counter
    else:
        return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
                     
    p = LUP_inplace(A)
    y = solve_L(A, b[p], ones_diagonal=True)
    x = solve_U(A, y)
    return x.flatten()


def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
                     
    _, swap_counter = LUP_inplace(A, return_swaps=True)
    return np.product(np.diagonal(A)) * (-1)**(swap_counter)
