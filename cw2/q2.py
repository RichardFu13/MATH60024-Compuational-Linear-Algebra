import numpy as np
import matplotlib.pyplot as plt
from cla_utils import GS_modified, solve_U

A2 = np.loadtxt('cw2/A2.dat') #load in text file


### 2a
def MGS_solve_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b using the algorithm described in
    question 2)a).

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    Q = A.copy()
    R = GS_modified(Q)
    z = Q.conj().T @ b
    x = solve_U(R, z).flatten()

    return x


### 2b
###plot located in plotting_scripts/q2_plots.py


### 2d
def MGS_solve_ls_modified(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b using my proposed modified algorithm.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    m, n = A.shape
    #add b as a column to the end of A
    A_plus = np.hstack((A, b.reshape((m, 1))))
    R_plus = GS_modified(A_plus)
    #slice out R
    R = R_plus[:n, :n]
    #slice out z
    z = R_plus[:n, n]
    x = solve_U(R, z).flatten()
    return x


### 2e
###plot located in plotting_scripts/q1_plots.py


### 2f
###plot located in plotting_scripts/q2_plots.py
