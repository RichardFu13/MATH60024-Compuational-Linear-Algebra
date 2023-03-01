import numpy as np
import timeit
import numpy.random as random
import time

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)
u0 = random.randn(400)
v0 = random.randn(400)
A1 = np.identity(400) + np.outer(u0, v0.conj())


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """
    m, n = A.shape
    b = np.zeros(m)
    for i in range(m):
        b_i = 0
        for j in range(n):
            b_i += A[i, j] * x[j]
        b[i] = b_i

    return b


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """
    m, n = A.shape
    b = np.zeros(m)
    for j in range(n):
        b += x[j] * A[:, j]
    
    return b


def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v1^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """
    B = np.column_stack((u1, u2))
    C = np.vstack((v1, v2)).conjugate()
    A = B.dot(C)

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """
    m = len(u)
    alpha = -1/(1+v.conj().dot(u))
    Ainv = np.identity(m) + alpha * np.outer(u, v.conj())

    return Ainv


def time_rank1pert_inv():
    """
    Compare timings of rank1pert_inv with numpy.linalg.inv for 400x400 matrix
    """
    start_time = time.time()
    rank1pert_inv(u0, v0)
    end_time = time.time()
    print(f"Timing using rank1pert_inv for 400x400 matrix: {end_time - start_time}")

    start_time = time.time()
    np.linalg.inv(A1)
    end_time = time.time()
    print(f"Timing using numpy.linalg.inv for 400x400 matrix: {end_time - start_time}")


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i>=j and Ahat[i,j] = C[i,j] for i<j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    m, _ = Ahat.shape
    zr = np.zeros(m)
    zi = np.zeros(m)
    #zr = Bx_r - Cx_i, zi = Bx_i + Cx_r
    for j in range(m):
        #column multiplying xr and xi with appropriate slices of B
        zr += xr[j] * np.concatenate((Ahat[j, 0:j], Ahat[j:m, j]))
        zi += xi[j] * np.concatenate((Ahat[j, 0:j], Ahat[j:m, j]))
    
        #column multiplying xr and xi with appropriate slices of C
        zr -= xi[j] * np.concatenate((Ahat[0:j, j], np.zeros(1), -Ahat[j, j+1:m]))
        zi += xr[j] * np.concatenate((Ahat[0:j, j], np.zeros(1), -Ahat[j, j+1:m]))

    return zr, zi
