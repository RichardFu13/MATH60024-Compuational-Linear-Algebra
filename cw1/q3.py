import numpy as np
from cla_utils import householder_qr, solve_U, householder_ls


#3)b)
def rq_decomposition(A):
    """
    Given a real nxm matrix A, use the full Householder QR factorisation to find
    the full RQ factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return R: an nxm-dimensional numpy array
    :return Q: an mxm-dimensional numpy array
    """
    m, n = A.shape
    A_hat = A[::-1]
    Q_hat, R_hat = householder_qr(A_hat.T)
    Q = Q_hat.T[::-1]
    R = R_hat.T[::-1][:, ::-1]
    return R, Q


#3)d)
def simultaneous_fact(A, B):
    """
    Given a real nxm matrix A, nxp matrix B, find the simulatenous factorization described
    in 3)c).

    :param A: an nxm-dimensional numpy array
    :param B: a nxp-dimensional numpy array

    :return Q: an nxn-dimensional numpy array
    :return U: an mxm-dimensional numpy array
    :return R: an nxm-dimensional numpy array
    :return S: an nxp-dimensional numpy array
    """
    Q, S = householder_qr(B)
    R, U_t = rq_decomposition(Q.T @ A)
    U = U_t.T
    return Q, U, R, S


#3)f)
def solve_constrained_ls(A, B, vector_b, vector_d):
    """
    Given a real mxn matrix A, pxn matrix B, an m dimensional vector b, and a p dimensional
    vector d, find the least squares solution to Ax = b with constraint Bx = d.

    :param A: an mxn-dimensional numpy array
    :param B: a pxn-dimensional numpy array
    :param b: an m-dimensional numpy array
    :param d: a p-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    m, n = A.shape
    p, _ = B.shape
    Q, U, R, S = simultaneous_fact(A.T, B.T) # get simultaneous factorisations for A^T, B^T
    S_11 = S[:p] # slice required parts of S
    R_12 = R[:p, m-n+p:] # slice required parts of R
    R_22 = R[p:, m-n+p:] # ^^
    print(U.T, vector_b)
    c = U.T @ vector_b # transform constraint constant
    y_1 = solve_U(S_11.T[::-1][:, ::-1], vector_d[::-1]).flatten()[::-1] # convert system to upper triangular and then reverse output vector
    y_2 = householder_ls(R_22.T, (c[m-n+p:] - R_12.T @ y_1)) # solve truncated LSE problem
    y = np.append(y_1, y_2) # create y
    return Q @ y # return x
