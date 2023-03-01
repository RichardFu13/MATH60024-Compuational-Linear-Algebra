import pytest
import numpy as np
from numpy import random
from q3 import rq_decomposition, simultaneous_fact, solve_constrained_ls


#3)b)
@pytest.mark.parametrize('m, n', [(20, 20), (40, 40), (87, 87)])
def test_rq(m, n):
    """
    Test RQ decomposition method on some random square matrices
    :param m: integer height of test matrix
    :param n: integer width of test matrix
    """
    random.seed(4732*m + 1238*n)
    A = random.randn(m, n)
    A0 = 1*A
    R, Q = rq_decomposition(A0)

    # check orthonormality
    assert(np.linalg.norm(np.dot(np.conj(Q.T), Q) - np.eye(m)) < 1.0e-6)
    # check upper triangular
    assert(np.allclose(R, np.triu(R)))
    # check RQ factorisation
    assert(np.linalg.norm(np.dot(R, Q) - A) < 1.0e-6)


#3)d)
@pytest.mark.parametrize('m, n, p', [(20, 18, 16), (40, 37, 36), (87, 85, 81)])
def test_simultaneous_fact(m, n, p):
    """
    Test simultaneous factorisation method on some random matrices
    :param m: integer width of random test matrix A
    :param n: integer height of random test matrix A and also height of randomtest matrix B
    :param p: integer width of random test matrix B
    """
    random.seed(4732*m + 1238*n + p)
    A = random.randn(n, m)
    A0 = 1*A
    B = random.randn(n, p)
    B0 = 1*B

    Q, U, R, S = simultaneous_fact(A0, B0)

    # check orthonormality of Q
    assert(np.linalg.norm(np.dot(np.conj(Q.T), Q) - np.eye(n)) < 1.0e-6)
    # check orthonormality of U
    assert(np.linalg.norm(np.dot(np.conj(U.T), U) - np.eye(m)) < 1.0e-6)
    # check R
    assert(np.allclose(R, np.triu(R, k=m-n)))
    # check S
    assert(np.allclose(S, np.triu(S)))
    # check factorisation of A
    assert(np.linalg.norm(Q.T @ A @ U - R) < 1.0e-6)
    # check factorisation of B
    assert(np.linalg.norm(Q.T @ B - S) < 1.0e-6)


@pytest.mark.parametrize('m, n, p', [(5, 4, 3), (4, 4, 2), (3, 2, 2)])
def test_ls_constraint(m, n, p):
    """Test the constraint is satisfied for a constrained LS problem"""
    A = random.randn(m, n)
    B = random.randn(p, n)
    vector_b = random.randn(m)
    vector_d = random.randn(p)
    x = solve_constrained_ls(A, B, vector_b, vector_d)
    assert(np.linalg.norm(B@x - vector_d) < 1.0e-6)


def test_ls_error():
    """Test using constrained least squares problem with known solution"""
    A = np.array([[1,1,1],[1,3,1],[1,-1,1],[1,1,1]])
    B = np.array([[1,1,1],[1,1,-1]])
    vector_b = np.array([1,2,3,4])
    vector_d = np.array([7,4])
    x = solve_constrained_ls(A, B, vector_b, vector_d) #calculated x
    actual_ls_x = np.array([5.75, -0.25, 1.5]) #known x_ls
    assert(np.linalg.norm(x - actual_ls_x) < 1.0e-6)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
