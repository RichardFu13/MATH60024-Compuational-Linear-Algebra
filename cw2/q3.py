import numpy as np
import matplotlib.pyplot as plt
from cla_utils import solve_L, solve_U, LU_inplace
import time


### 3b
def solve_LU(A, b, bl=0, bu=0):
    """
    Solve Ax=b using LU factorisation, with support for
    optimising this process when matrix is banded.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array
    :param bl: an integer, the lower bandwidth for matrix A.
    Defaults to zero.
    :param bu: an integer, the upper bandwidth for matrix A.
    Defaults to zero.

    :return x: an m-dimensional numpy array
    """
    LU_inplace(A, bl=bl, bu=bu)
    #solve lower triangular system by forward sub
    y = solve_L(A, b, ones_diagonal=True, bl=bl)
    #solve upper triangular system by back sub
    x = solve_U(A, y, bu=bu)
    if x.shape[-1] == 1:
        return x.flatten()
    else:
        return x


### 3c
def construct_D(N, s):
    """
    Construct the square matrix D as described in question 1.

    :param N: an integer, where N^2 is the dimension of the matrix to generate.
    :param s: a positive float, the strength of correlations between nearby points.

    :return A: an N^2xN^2-dimensional numpy array.
    """
    D = np.zeros((N**2, N**2))
    #fill upper triagular entries first
    for i in range(N**2):
        for j in range(i+1, min(i+1+N, N**2)):
            if np.abs(j-i) == N:
                D[i, j] = -s**2
            elif j-i == 1:
                if (i+1) % N != 0:
                    D[i, j] = -s**2
    #fill lower triangular entries first
    D += D.T
    #fill diagonal entries
    np.fill_diagonal(D, 1+4*s**2)
    return D


def image_simulator(N, s, w, banded_solve=False, return_time=False):
    """
    Simulate a random image by solving the system Du=w.

    :param N: an integer, where N is the size of our image grid.
    :param s: a positive float, the strength of correlations between nearby points.
    :param w: an N^2-dimensional numpy array.
    :param banded_solve: a boolean to indicate whether to use the banded solver.
    :param return_time: a boolean to indicate whether to return the time elapsed
    of the solve.

    :return u: an N^2-dimensional numpy array.
    :return elapsed_time: a float, the time elapsed in seconds.
    """
    D = construct_D(N, s)
    start_time = time.time()
    if banded_solve:
        u = solve_LU(D, w, bl=N, bu=N)
    else:
        u = solve_LU(D, w)
    end_time = time.time()
    if return_time:
        elapsed_time = end_time - start_time
        return u, elapsed_time
    else:
        return u

###plot located in plotting_scripts/q3_plots.py


### 3d
###plot located in plotting_scripts/q3_plots.py


### 3g
def construct_V_block(N):
    """
    Construct a block of V of size N, a banded matrix with 2's on the
    main diagonal and 1's on the sub and super diagonals.

    :param N: an integer, the size of the block V.
    
    :return V_block: an NxN-dimensional numpy array.
    """
    V_block = -np.identity(N-1)
    V_block = np.hstack((np.zeros(N-1).reshape(N-1, 1), V_block))
    V_block = np.vstack((V_block, np.zeros(N).reshape(1, N)))
    V_block += V_block.T
    np.fill_diagonal(V_block, 2)
    return V_block


def mat_mul_banded(A, B, bl=0, bu=0):
    """
    Perform efficient matrix multiplication of a square banded matrix A with square matrix B.
    :param A: An nxn-dimensional banded numpy array.
    :param B: An nxn-dimensional numpy array.
    :param bl: an integer, the lower bandwidth for matrix A.
    Defaults to zero.
    :param bu: an integer, the upper bandwidth for matrix A.
    Defaults to zero.

    :return C: an nxn-dimensional numpy array.
    """
    n, _ = A.shape
    if not bl:
        bl = n
    if not bu:
        bu = n
    C = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            #define appropriate slicing indices
            idx1 = max(i-bl, 0)
            idx2 = min(i+bu+1, n)
            C[i, j] = np.inner(A[i, idx1:idx2], B[idx1:idx2, j])
    return C


def run_iter(N, u_n, s, rho, nu, w, return_step_1=False):
    """
    Run a single iteration of solving the iterative system described in (9) and (10)

    :param N: an integer, where N is the size of our image grid.
    :param u_n: an N^2-dimensional numpy array, n-th iteration of u.
    :param s: a positive float, the strength of correlations between nearby points.
    :param rho: a positive float.
    :param nu: a positive float.
    :param w: an N^2-dimensional numpy array.
    :param return_step_1: a boolean to indicate whether to return n+1/2th iteration of u
    (ONLY USED IN PYTEST)

    :return next_u_n: an N^2-dimensional numpy array, n+1st iteration of u.
    """
    V_block = construct_V_block(N)
    #reshapes vector to matrices columnwise
    u_n_mat = u_n.reshape(N, N, order="F")
    w_mat = w.reshape(N, N, order="F")
    #calculate RHS of eq1 system before permutation
    eq1_rhs = mat_mul_banded((rho * np.identity(N) - s**2 * V_block), u_n_mat, bl=1, bu=1) + w_mat
    #calculate RHS of eq1 system after permutation by transposing
    eq1_rhs_perm = eq1_rhs.T
    #calculate the LHS of eq1 system
    eq1_lhs = ((1+rho) * np.identity(N) + s**2 * V_block)
    #solve all the systems of eq1
    inter_u_n_mat = solve_LU(eq1_lhs, eq1_rhs_perm, bl=1, bu=1)

    #ONLY FOR PYTEST TO RETURN n+1/2th ITERATION
    if return_step_1:
        return inter_u_n_mat.flatten()

    #calculate RHS of eq2 system
    eq2_rhs = (mat_mul_banded((nu * np.identity(N) - s**2 * V_block), inter_u_n_mat, bl=1, bu=1)).T + w_mat
    #calculate LHS of eq2 system
    eq2_lhs = ((1+nu) * np.identity(N) + s**2 * V_block)
    #solve all the systems of eq2
    next_u_n_mat = solve_LU(eq2_lhs, eq2_rhs, bl=1, bu=1)
    #flatten the matrix columnwise to get vector
    next_u_n = next_u_n_mat.flatten("F")

    return next_u_n


def iterative_solve(N, u_0, s, rho, nu, w, eps, return_iter = False):
    """
    Run our iterative algorithm using run_iter until our stopping criteria is met.

    :param N: an integer, where N is the size of our image grid.
    :param u_0: an N^2-dimensional numpy array, the initial u.
    :param s: a positive float, the strength of correlations between nearby points.
    :param rho: a positive float.
    :param nu: a positive float.
    :param w: an N^2-dimensional numpy array.
    :param eps: a positive float, used in our stopping criteria.
    :param return_iter: a boolean to indicate whether to return iteration count.

    :return u_n: an N^2-dimensional numpy array, our iterative solution to Du = w
    :return iter_count: an integer, the number of iterations taken.
    """
    u_n = u_0
    D = construct_D(N, s)
    iter_count = 0
    #check stopping condition
    while np.linalg.norm(D @ u_n - w) >= eps * np.linalg.norm(w):
        iter_count += 1
        #iterate u
        u_n = run_iter(N, u_n, s, rho, nu, w)
    if return_iter:
        return u_n, iter_count
    else:
        return u_n


### 3h
###plot located in plotting_scripts/q3_plots.py
