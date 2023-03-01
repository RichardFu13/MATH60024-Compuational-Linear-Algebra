import numpy as np
import numpy.random as random
from cla_utils import householder_ls
from cw3.q2 import get_callback


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    m, _ = A.shape
    H = np.zeros((k+1, k), dtype=A.dtype)
    Q = np.zeros((m, k+1), dtype=A.dtype)
    Q[:, 0] = b / np.linalg.norm(b)
    for n in range(k):
        v = A @ Q[:, n]
        H[:n+1, n] = Q[:m, :n+1].conj().T @ v
        v -= Q[:, :n+1] @ H[:n+1, n]
        H[n+1, n] = np.linalg.norm(v)
        Q[:, n+1] = v / H[n+1, n]
    
    return Q, H


def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False, return_residuals=False, callback=None):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical
    :param callback: a callback function

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    if callback is not None:
        f = open("cw3/callback.dat", "w")
        f.close()

    if x0 is None:
        x0 = b
    m, _ = A.shape
    H = np.zeros((maxit+1, maxit), dtype=A.dtype)
    Q = np.zeros((m, maxit+1), dtype=A.dtype)
    Q[:, 0] = b / np.linalg.norm(b)
    x = np.zeros(m)
    nits = 0
    if return_residual_norms:
        rnorms = np.array([])
    if return_residuals:
        r = np.zeros((m, maxit))
    while nits < maxit:
        v = A @ Q[:, nits]
        H[:nits+1, nits] = Q[:m, :nits+1].conj().T @ v
        v -= Q[:, :nits+1] @ H[:nits+1, nits]
        H[nits+1, nits] = np.linalg.norm(v)
        Q[:, nits+1] = v / H[nits+1, nits]
        rhs = np.zeros(nits+2)
        rhs[0] = np.linalg.norm(b)
        lhs = H[:nits+2, :nits+1].copy()
        y = householder_ls(lhs, rhs)
        x = Q[:, :nits+1] @ y
        if callback is not None:
            callback(x)
        residual_n = H[:nits+2, :nits+1] @ y - rhs
        if return_residual_norms:
            rnorms = np.append(rnorms, np.linalg.norm(residual_n))
        if return_residuals:
            r[:, nits] = A @ x - b
        nits += 1
        if np.linalg.norm(residual_n) < tol:
            if return_residual_norms and return_residuals:
                r = r[:, :nits]
                return x, nits, rnorms, r
            elif return_residual_norms:
                return x, nits, rnorms
            elif return_residuals:
                r = r[:, :nits]
                return x, nits, r
            else:
                return x, nits

    nits = -1
    if return_residual_norms and return_residuals:
        return x, nits, rnorms, r
    elif return_residual_norms:
        return x, nits, rnorms
    elif return_residuals:
        return x, nits, r
    else:
        return x, nits


def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
