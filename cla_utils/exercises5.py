import numpy as np
from cla_utils import operator_2_norm, solve_U, householder_solve


def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = np.linalg.qr(np.random.randn(m, m))
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = np.random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)
        A = Q1 @ R1
        Q2, R2 = np.linalg.qr(A)
        print(operator_2_norm(Q2-Q1), operator_2_norm(R2-R1), operator_2_norm(A-Q2@R2)/ operator_2_norm(A))


def back_stab_solve_U(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = np.random.randn(m, m)
        R = np.triu(A)
        x = np.random.randn(m)
        b = R @ x
        x_tilde = solve_U(R, b).flatten()
        b_tilde = R @ x_tilde
        print(np.linalg.norm(b_tilde - b)/np.linalg.norm(b))


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = np.random.randn(m, m)
        x = np.random.randn(m)
        b = A @ x
        x_tilde = householder_solve(A, b)
        b_tilde = A @ x_tilde
        print(np.linalg.norm(b_tilde - b)/np.linalg.norm(b))
        

# backward_stability_householder(50)
# back_stab_solve_U(10)
# back_stab_householder_solve(10)
