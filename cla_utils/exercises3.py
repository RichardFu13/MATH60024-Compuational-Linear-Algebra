import numpy as np


def householder(A, kmax=None, swap=None, reduced_tol=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations. The reduction should be done "in-place",
    so that A is transformed to R.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """
    
    m, n = A.shape
    if kmax==None:
        kmax = n
    for k in range(kmax):
        if swap==True:
            if k == 0:
                norm_squared_array = np.array([np.inner(A[0:m, i], A[0:m, i]) for i in range(kmax)]) # initialize squared norm array
            else:
                norm_squared_array[k:] -= np.array([A[k-1, i]**2 for i in range(k, kmax)]) # subtract the squared norm of the k-1th entry
            idx = max([idx for idx in range(k, kmax)], key = lambda x: norm_squared_array[x]) # index for largest norm is same as index for largest norm squared
            if reduced_tol:
                if np.sqrt(np.abs(norm_squared_array[idx])) < reduced_tol: # if the largest norm less than tolerance, end alg and return sliced matrix
                    return A[:k, :]
            A.T[[k, idx]] = A.T[[idx, k]] # swap the columns k and max norm index
            norm_squared_array[k], norm_squared_array[idx] = norm_squared_array[idx], norm_squared_array[k] # swap the k and max norm index entries in norm squared array
        x = A[k:m, k] # slice appropriate x
        v_k = x.copy()
        # create householder
        if x[0] == 0:
            v_k[0] += np.linalg.norm(x)
        else:
            v_k[0] += np.sign(x[0]) * np.linalg.norm(x)
        if np.linalg.norm(v_k) != 0:
            v_k /= np.linalg.norm(v_k)
        A[k:m, k:n] -= 2 * np.outer(v_k, v_k.conj().dot(A[k:m, k:n])) # apply householder to the sliced matrix

    return A


def solve_U(U, b, bu=0):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :param bu: an integer, the upper bandwidth for matrix A.
    Defaults to zero.
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m, _ = U.shape
    if len(b.shape) == 1:
        b = np.reshape(b, (b.shape[0], 1))
    _, k = b.shape
    x = np.zeros((m, k))
    if bu:
        for i in range(m-1, -1, -1):
            j = min(m, bu+i+1)
            x[i, :] = (b[i, :] - U[i, i+1:j].dot(x[i+1:j, :])) / U[i, i]
    else:
        for i in range(m-1, -1, -1):
            x[i, :] = (b[i, :] - U[i, i+1:].dot(x[i+1:, :])) / U[i, i]

    return x


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    m, n = A.shape
    if len(b.shape) == 1:
        b = np.reshape(b, (b.shape[0], 1))
    A_hat = np.hstack((A, b))
    R_hat = householder(A_hat, kmax=m)
    x = solve_U(R_hat[:, :m], R_hat[:, m:])

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    m, n = A.shape
    A_hat = np.hstack((A, np.identity(m)))
    R_hat = householder(A_hat, kmax=m)
    R = R_hat[:, :n]
    Q = R_hat[:, n:].conj().T

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    m, n = A.shape
    A_hat = np.hstack((A, b.reshape((m, 1))))
    R_hat = householder(A_hat)
    reduced_R = R_hat[:n,:n]
    reduced_Q_adj_b = R_hat[:n, n]
    x = solve_U(reduced_R, reduced_Q_adj_b).flatten()

    return x
