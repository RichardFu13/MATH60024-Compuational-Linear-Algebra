import numpy as np
import time
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(100, 100)
A1 = random.randn(200, 200)
A2 = random.randn(400, 400)
Q0, _ = np.linalg.qr(A0)
Q1, _ = np.linalg.qr(A1)
Q2, _ = np.linalg.qr(A2)
x0 = random.randn(100)
x1 = random.randn(200)
x2 = random.randn(400)

def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    m, n = Q.shape
    r = v.copy()
    u = np.zeros(n, dtype=Q.dtype)
    for i in range(n):
        q_i = Q[:, i]
        u_i = q_i.conj().dot(v)
        r -= u_i * q_i
        u[i] = u_i

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """
    x = Q.conj().T.dot(b)

    return x


def time_solveQ():
    """
    Compare timings of solveQ with numpy.linalg.solve for unitary matrices Q
    """
    start_time = time.time()
    solveQ(Q0, x0)
    end_time = time.time()
    print(f"Timing using solveQ for 100x100 matrix: {end_time - start_time}")
    start_time = time.time()
    np.linalg.solve(Q0, x0)
    end_time = time.time()
    print(f"Timing using numpy.linalg.solve for 100x100 matrix: {end_time - start_time}")

    start_time = time.time()
    solveQ(Q1, x1)
    end_time = time.time()
    print(f"Timing using solveQ for 200x200 matrix: {end_time - start_time}")
    start_time = time.time()
    np.linalg.solve(Q1, x1)
    end_time = time.time()
    print(f"Timing using numpy.linalg.solve for 200x200 matrix: {end_time - start_time}")

    start_time = time.time()
    solveQ(Q2, x2)
    end_time = time.time()
    print(f"Timing using solveQ for 400x400 matrix: {end_time - start_time}")
    start_time = time.time()
    np.linalg.solve(Q2, x2)
    end_time = time.time()
    print(f"Timing using numpy.linalg.solve for 400x400 matrix: {end_time - start_time}")


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """
    P = Q @ Q.conj().T

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """
    m, n = V.shape
    Q, _ = np.linalg.qr(V, mode="complete")
    Q = Q[:, n:m]

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.zeros((n, n), dtype=A.dtype)
    # for j in range(n):
    #     R[:j, j] = orthog_proj(A)[:j, j] #fill in columns of R with projectors
    #     if j>0:
    #         A[:, j] -= (A @ R)[:, j] #project out the previous columns
    #     R[j, j] = np.linalg.norm(A[:, j])
    #     A[:, j] /= R[j, j]

    for j in range(n):
        R[:j, j] = (A.T.conj()).dot(A)[:j, j]
        A[:, j] -= (A @ R)[:, j]
        R[j, j] = np.linalg.norm(A[:, j])
        A[:, j] /= R[j, j]

    return R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] /= R[i, i]
        # for j in range(i+1, n):
        #     R[i, j] = A[:, i].conj().dot(A[:, j])
        #     A[:, j] -= R[i, j] * A[:, i]

        R[i, i+1:] = A.T.conj().dot(A)[i, i+1:]
        A[:, i+1:] -= np.outer(A[:,i],(R[i, i+1:]))
    return R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.identity(n, dtype=A.dtype)
    R[k, k] = 1 / np.linalg.norm(A[:, k])
    for j in range(k+1, n):
        R[k, j] = A[:, k].dot(A[:, j]) / R[k, k]

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
