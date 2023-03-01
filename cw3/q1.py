import numpy as np
import matplotlib.pyplot as plt
from cla_utils import pure_QR, inverse_it, householder_qr, hessenberg

A3 = np.loadtxt('cw3/A3.dat') #load in text file
A4 = np.loadtxt('cw3/A4.dat') #load in text file


### 1c
Ak_c = pure_QR(A4, 10000, 1.0e-6)
print(f"resulting Ak from pure_QR on A4: {Ak_c}")


### 1d
#demonstration of new method on A4
Ak_d = pure_QR(A4, 10000, 1.0e-6, special_criteria=True)
eigs = np.zeros(6, dtype=complex)
#get idxs of all non zero entries on lower subdiagonal
non_zero_idxs = np.argwhere(np.abs(np.diag(Ak_d, -1)) > 1.0e-6).flatten()
for k in range(6):
    if k in non_zero_idxs:
        #solve 2x2 blocks for eigenvalues
        tr = Ak_d[k,k] + Ak_d[k+1,k+1]
        det = Ak_d[k,k] * Ak_d[k+1,k+1] - Ak_d[k,k+1] * Ak_d[k+1,k]
        eigs[k] = (tr + np.emath.sqrt(tr**2 - 4*det))/2
        eigs[k+1] = (tr - np.emath.sqrt(tr**2 - 4*det))/2
    else:
        #1x1 block is exactly the eigenvalue
        if k-1 in non_zero_idxs:
            pass
        else:
            eigs[k] = Ak_d[k,k]
print(f"eigenvalues of A4 are: {eigs}")

#verify these are eigenvalues
evecs = np.zeros((6, 6), dtype=complex)
eig_errors = np.zeros(6)
for idx, eig in enumerate(eigs):
    v_k, _ = inverse_it(A4, np.ones(6), eig, 1.0e-6, 10000)
    eig_errors[k] = np.linalg.norm(A4 @ v_k - eig * v_k)
#print norm of these errors
print(f"norm of residual errors is {np.linalg.norm(eig_errors)}")


### 1e
def block_matrix_eigs(A, maxit, tol):
    """
    Find the eigenvalues of a matrix of matrices with complex eigenvalues by
    getting into A' form and solving diagonal blocks for eigenvalues.

    :param A: mxm numpy array
    :param maxit: integer, max number of iterations of pure_QR
    :param tol: a positive float, the tolerance passed into pure_QR

    :return eigs: an m dimensional numpy array containing the numerical eigenvalues of A
    """
    m, _ = A.shape
    eigs = np.zeros(m, dtype=complex)
    Ak = pure_QR(A, maxit, 1.0e-6, special_criteria=True)
    #get idxs of all non zero entries on lower subdiagonal
    non_zero_idxs = np.argwhere(np.abs(np.diag(Ak, -1)) > tol).flatten()
    for k in range(m):
        if k in non_zero_idxs:
            #solve 2x2 blocks for eigenvalues
            tr = Ak[k,k] + Ak[k+1,k+1]
            det = Ak[k,k] * Ak[k+1,k+1] - Ak[k,k+1] * Ak[k+1,k]
            eigs[k] = (tr + np.emath.sqrt(tr**2 - 4*det))/2
            eigs[k+1] = (tr - np.emath.sqrt(tr**2 - 4*det))/2
        else:
            #1x1 block is exactly the eigenvalue
            if k-1 in non_zero_idxs:
                pass
            else:
                eigs[k] = Ak[k,k]
    return eigs

#demonstration on some matrices
#3x3 example
M1 = np.array([[2, 5, 1],[-1, 0, 2], [0, 0, 3]])
print(f"eigenvalues of M1: {block_matrix_eigs(M1, 10000, 1.0e-6)}")
#5x5 example
M2 = np.zeros((5, 5))
M2[2:, 2:] = M1
M2[:2, :2] = np.array([[0, 1], [-1, 0]])
print(f"eigenvalues of M2: {block_matrix_eigs(M2, 10000, 1.0e-6)}")
#7x7 example
M3 = np.zeros((7, 7))
M3[:5, :5] = M2
M3[5:, 5:] = np.array([[0, 5], [-1, 4]])
print(f"eigenvalues of M3: {block_matrix_eigs(M3, 10000, 1.0e-6)}")


### 1h
def QR_shift(A, maxit, tol, return_iterations=False):
    """
    For matrix A, apply the QR algorithm with shifting and deflation.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param return_iterations: bool defaulted to False

    :return eigs: an m dimensional numpy array containing the numerical eigenvalues of A
    :return iter_arr: an kxm dimensional numpy array containing the iterations of the eigenvalues on its rows,
    where k is the number of iterationstaken to converge.
    """
    Ak = A.copy()
    #transform A to hessenberg form
    hessenberg(Ak)
    m, _ = Ak.shape
    j = m-1
    iter = maxit
    if return_iterations:
        iter_arr = np.zeros((maxit, m))
    for k in range(maxit):
        #pick shift
        mu_k = Ak[j, j]
        #perform algorithm on submatrix
        Qk, Rk = householder_qr(Ak[:j+1, :j+1] - mu_k * np.identity(j+1))
        Ak[:j+1, :j+1] = Rk @ Qk + mu_k * np.identity(j+1)
        #move onto submatrix if current entry has converged enough
        if return_iterations:
            iter_arr[k, :] = np.diag(Ak)
        if np.abs(Ak[j-1, j]) < tol and j > 1:
            j -= 1
        elif np.abs(Ak[j-1, j]) < tol and j == 1:
            iter = k
            break
    eigs = np.diag(Ak)
    if return_iterations:
        iter_arr = iter_arr[:iter+1, :]
        return eigs, iter_arr
    else:
        return eigs


### 1k
print(QR_shift(A4, 1000, 1.0e-6))
print(np.linalg.eig(A4)[1])
