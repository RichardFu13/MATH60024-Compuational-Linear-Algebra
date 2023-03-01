import numpy as np
from cw3.q1 import QR_shift
from cla_utils import inverse_it, householder_qr


### 3c
def get_D(A, return_H=False):
    """
    Construct the diagonal matrix D in the SVD of A.
    :param A: mxm dimensional numpy array
    :param return_H: a bool, determines whether to our matrix H as described in question. Defaults to False.

    :return D: an mxm dimensional diagonal numpy array,
    :return H: a 2mx2m dimensional numpy array
    """
    A_copy = A.copy()
    m, _ = A_copy.shape
    H = np.zeros((2*m,2*m))
    H[:m, m:] = A_copy.conj().T
    H[m:, :m] = A_copy
    eigs_H = QR_shift(H, 10000, 1.0e-6)
    eigs_H = np.sort(eigs_H)[::-1]
    D = np.diag(eigs_H[:m], k=0)
    if return_H:
        return D, H
    else:
        return D


### 3d
def get_evecs(D, H, return_eigs=False):
    """
    Use inverse_it to find the eigenvectors of H.
    :param D: an mxm dimensional diagonal numpy array
    :param H: a 2mx2m dimensional numpy array
    :param return_eigs:, a bool, determines whether to return the eigenvalues of H

    :return evecs_H: a 2mx2m dimensional numpy array containing the eigenvectors of H in its columns
    :return eigs_H: a 2m dimensional numpy array containing the corresponding eigenvalues
    """
    two_m, _ = H.shape
    eigs_H = np.append(np.diag(D), - np.diag(D))
    evecs_H = np.zeros((two_m,two_m))
    for idx, eig in enumerate(eigs_H):
        evecs_H[:, idx], _ = inverse_it(H, np.ones(two_m), eig, 1.0e-6, 10000)
    if return_eigs:
        return evecs_H, eigs_H
    else:
        return evecs_H


### 3e
def create_rd_matrix(m, remove_amount):
    """
    Create an mxm rank-deficient matrix by setting some rows to 0.
    :param m: integer of the height of our matrix
    :param remove_amount: integer<=m of the number of rows to set to 0

    :return B: an mxm rank deficient numpy array
    """
    A = np.random.randn(m,m)
    Q, R = householder_qr(A)
    idxs = np.random.choice(np.arange(m), replace=False, size=remove_amount)
    R[idxs] = np.zeros(m) # set rows to 0
    B = Q @ R
    return B


def rank_def_LS(A, b):
    """
    Given a real rank deficient mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    m, _ = A.shape
    D, H = get_D(A, return_H=True)
    r = 0 # initialize rank counter
    for i in range(m):
        if D[i, i] > 1.0e-6: # non-zero diagonal entries (larger than 1.0e-6)
            r += 1
    evecs_H = get_evecs(D, H)
    V = np.sqrt(2) * evecs_H[:m, :m]
    U = np.sqrt(2) * evecs_H[m:, :m]
    #calculate x
    x = np.zeros(m)
    for i in range(r):
        x += U.conj()[:, i].dot(b) / D[i, i] * V[:, i]

    return x
