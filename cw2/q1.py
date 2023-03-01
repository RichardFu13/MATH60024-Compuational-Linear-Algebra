import numpy as np
import matplotlib.pyplot as plt
from cla_utils import LUP_inplace, operator_2_norm, solve_LUP


### 1a
def generate_A(n):
    """
    Generate the square matrix A^(n) as described in question 1.

    :param n: an integer, the dimensions of the matrix to generate.

    :return A: an nxn-dimensional numpy array.
    """
    A = np.tril(-np.ones((n,n-1)), -1)
    np.fill_diagonal(A, 1)
    A = np.hstack((A, np.ones((n, 1))))
    return A


def get_rho(A):
    """
    Calculate the growth factor rho as defined in section 4.3.

    :param A: an nxn-dimensional numpy array.

    :return rho: a float, the growth factor of matrix A.
    """
    A_copy = A.copy()
    #find the largest absolute entry of matrix A
    a_max = np.max(np.abs(A_copy.flat))
    LUP_inplace(A_copy)
    #find the largest absolute entry of matrix U
    u_max = np.max(np.abs(np.triu((A_copy)).flat))
    rho = u_max / a_max
    return rho

#print the growth rate of A^(6)
A6 = generate_A(6)
print(get_rho(A6))


### 1c
def error_LUP(A):
    """
    Calculate the forward error in the LUP factorisation of A using LUP_inplace.

    :param A: an nxn-dimensional numpy array.

    :return err: a float, the forward error in our LUP factorisation.
    """
    A_copy = A.copy()
    p = LUP_inplace(A_copy)
    L = np.tril(A_copy, -1)
    np.fill_diagonal(L, 1)
    U = np.triu(A_copy)
    err = operator_2_norm(A[p, :] - L @ U)/operator_2_norm(A)
    return err


def error_solve_LUP(A):
    """
    Calculate the forward error from using solve_LUP to solve Ax = b.

    :param A: an nxn-dimensional numpy array

    :return err: a float, the forward error from using solve_LUP
    """
    A_copy = A.copy()
    x = np.random.randn(A.shape[0])
    b = A_copy @ x
    x_tilde = solve_LUP(A_copy, b)
    err = np.linalg.norm(x_tilde - x)/np.linalg.norm(x)
    return err

#print the errors for A^(60)
A60 = generate_A(60).astype(dtype=np.float64)
print(error_LUP(A60))
print(error_solve_LUP(A60))


### 1d
def random_matrix(n):
    """
    Generate a random square matrix with entries sampled from a
    Uniform(-1/n, 1/n) distribution.

    :param n: an integer, the dimensions of the matrix.

    :return A: an nxn-dimensional numpy array, with entries as described.
    """
    A = np.random.uniform(-1/n, 1/n, (n, n))
    return A

###plot located in plotting_scripts/q1_plots.py
