import numpy as np

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    x = A[:, 0] # slice appropriate x
    v = x.copy()
    # create householder
    if x[0] == 0:
        v[0] += np.linalg.norm(x)
    else:
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
    if np.linalg.norm(v) != 0:
        v /= np.linalg.norm(v)
    A -= 2 * np.outer(v, v.conj().dot(A)) # apply householder to the sliced matrix
    A_star = A.T.conj()
    A_star -= 2 * np.outer(A_star.dot(v), v.conj())
    A = A_star.T.conj()
    
    return A


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """
    m, _ = A.shape
    for k in range(m-2):
        x = A[k+1:, k]
        v_k = x.copy()
        if x[0] == 0:
            v_k[0] += np.linalg.norm(x)
        else:
            v_k[0] += np.sign(x[0]) * np.linalg.norm(x)
        if np.linalg.norm(v_k) != 0:
            v_k /= np.linalg.norm(v_k)
        A[k+1:, k:] -= 2 * np.outer(v_k, v_k.conj().dot(A[k+1:, k:]))
        A[:, k+1:] -= 2 * np.outer(A[:, k+1:].dot(v_k), v_k.conj())


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    m, _ = A.shape
    Q = np.identity(m)
    for k in range(m-2):
        x = A[k+1:, k]
        v_k = x.copy()
        if x[0] == 0:
            v_k[0] += np.linalg.norm(x)
        else:
            v_k[0] += np.sign(x[0]) * np.linalg.norm(x)
        if np.linalg.norm(v_k) != 0:
            v_k /= np.linalg.norm(v_k)
        A[k+1:, k:] -= 2 * np.outer(v_k, v_k.conj().dot(A[k+1:, k:]))
        Q[k+1:, :] -= 2 * np.outer(v_k, v_k.conj().dot(Q[k+1:, :]))
        A[:, k+1:] -= 2 * np.outer(A[:, k+1:].dot(v_k), v_k.conj())

    return Q.T.conj()


def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H

    Do not change this function.
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """
    Q = hessenbergQ(A)
    V = hessenberg_ev(A)
    
    return Q @ V
