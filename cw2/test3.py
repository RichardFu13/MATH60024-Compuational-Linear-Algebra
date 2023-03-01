import pytest
import numpy as np
from cla_utils import solve_L, solve_U, LU_inplace
from q3 import solve_LU, iterative_solve, mat_mul_banded, run_iter, construct_D


@pytest.mark.parametrize('m', [20, 204, 18])
def test_LU_inplace_banded(m):
    np.random.seed(8564*m)
    A = np.random.randn(m, m)
    #randomly select bandwidths
    bu = np.random.randint(1, m-1)
    bl = np.random.randint(1, m-1)
    #set entries outside bandwiths to zero
    A = np.triu(A, -bl)
    A = np.tril(A, bu)
    A0 = 1.0*A
    LU_inplace(A, bl=bl, bu=bu)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    A1 = np.dot(L, U)
    err = A1 - A0
    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m, k', [(20, 4), (204, 100), (18, 7)])
def test_solve_L_banded(m, k):
    np.random.seed(1002*m + 2987*k)
    b = np.random.randn(m, k)
    Q, R = np.linalg.qr(np.random.randn(m,m))
    L = R.T
    #randomly select a bandwidth
    n = np.random.randint(1, m-1)
    #set entries lower than bandwidth to zero
    A = np.triu(L, -n)
    x = solve_L(A, b, bl=n)
    err = b - np.dot(A, x)
    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m, k', [(20, 4), (204, 100), (18, 7)])
def test_solve_U_banded(m, k):
    np.random.seed(1002*m + 2987*k)
    b = np.random.randn(m, k)
    Q, R = np.linalg.qr(np.random.randn(m,m))
    #randomly select a bandwidth
    n = np.random.randint(1, m-1)
    #set entries higher than bandwidth to zero
    A = np.tril(R, n)
    x = solve_U(A, b, bu=n)
    err = b - np.dot(A, x)
    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_solve_LU_banded(m):
    np.random.seed(8364*m)
    A = np.random.randn(m, m)
    #randomly select bandwidths
    bu = np.random.randint(1, m-1)
    bl = np.random.randint(1, m-1)
    #set entries outside bandwiths to zero
    A = np.triu(A, -bl)
    A = np.tril(A, bu)
    A0 = 1.0*A
    b = np.random.randn(m)
    x = solve_LU(A, b, bl=bl, bu=bu)
    assert(np.linalg.norm(b - np.dot(A0, x)) < 1.0e-6)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_mat_mul_banded(m):
    np.random.seed(8364*m)
    A = np.random.randn(m, m)
    #randomly select bandwidths
    bu = np.random.randint(1, m-1)
    bl = np.random.randint(1, m-1)
    #set entries outside bandwiths to zero
    A = np.triu(A, -bl)
    A = np.tril(A, bu)
    B = np.random.randn(m, m)
    assert(np.linalg.norm(A @ B - mat_mul_banded(A, B, bu=bu, bl=bl)) < 1.0e-6)


###HELPER FUNCTIONS FOR PYTEST
def construct_H(N):
    """
    Construct the square matrix H as described in question.
    :param N: an integer.
    :return H: an N^2xN^2-dimensional numpy array.
    """
    H = np.zeros((N**2, N**2))
    #fill upper triagular entries first
    for i in range(N**2):
        for j in range(i+1, min(i+1+N, N**2)):
            if np.abs(j-i) == N:
                H[i, j] = -1
    #fill lower triangular entries first
    H += H.T
    #fill diagonal entries
    np.fill_diagonal(H, 2)
    return H


def construct_V(N):
    """
    Construct the square matrix V as described in question.
    :param N: an integer.
    :return V: an N^2xN^2-dimensional numpy array.
    """
    V = np.zeros((N**2, N**2))
    #fill upper triagular entries first
    for i in range(N**2):
        for j in range(i+1, min(i+1+N, N**2)):
            if j-i == 1:
                if (i+1) % N != 0:
                    V[i, j] = -1
    #fill lower triangular entries first
    V += V.T
    #fill diagonal entries
    np.fill_diagonal(V, 2)
    return V


@pytest.mark.parametrize('m, s, rho, nu', [(5, 0.1, 1, 1), (10, 0.5, 0.5, 1), (20, 1, 1, 0.5)])
def test_run_iter(m, s, rho, nu):
    u_n = np.zeros(m**2)
    b = np.random.randn(m**2)
    inter_u_n_iter = run_iter(m, u_n, s, rho, nu, b, return_step_1=True) #n+1/2th iter
    u_n_iter = run_iter(m, u_n, s, rho, nu, b, return_step_1=False) #n+1th iter
    V = construct_V(m)
    H = construct_H(m)
    #check equation 9
    assert(np.linalg.norm(((1+rho)*np.identity(m**2) + s**2*H) 
           @ inter_u_n_iter - ((rho*np.identity(m**2) - s**2*V)@ u_n + b)) < 1.0e-6)
    #check equation 10
    assert(np.linalg.norm(((1+nu)*np.identity(m**2) + s**2*V)
           @ u_n_iter - ((nu*np.identity(m**2) - s**2*H) @ inter_u_n_iter + b)) < 1.0e-6)


@pytest.mark.parametrize('m, s', [(20, 0.1), (40, 0.5), (60, 1)])
def test_iterative_solve(m, s):
    u_0 = np.zeros(m**2)
    D = construct_D(m, s)
    b = np.random.randn(m**2)
    u_n_iter = iterative_solve(m, u_0, s, 1, 1, b, 1.0e-8)
    #check convergence
    assert(np.linalg.norm(D @ u_n_iter - b) < 1.0e-6)
