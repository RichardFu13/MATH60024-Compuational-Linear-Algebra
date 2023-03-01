import numpy as np
import numpy.random as random
from cla_utils import solve_LUP, householder_qr

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.76505141, -0.03865876,  0.42107996],
                     [-0.03865876,  0.20264378, -0.02824925],
                     [ 0.42107996, -0.02824925,  0.23330481]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.76861909,  0.01464606,  0.42118629],
                     [ 0.01464606,  0.99907192, -0.02666057],
                     [ 0.42118629, -0.02666057,  0.23330798]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    if store_iterations:
        x = np.array([x0])
        for k in range(maxit):
            x_k = A @ x[-1]
            x_k /= np.linalg.norm(x_k)
            lambda0 = x_k.dot(A @ x_k)
            x = np.append(x, x_k)
            if np.linalg.norm(A @ x_k - lambda0 * x_k):
                break
    else:
        x = x0
        for k in range(maxit):
            x = A @ x
            x /= np.linalg.norm(x)
            lambda0 = x.dot(A @ x)
            if np.linalg.norm(A @ x - lambda0 * x) < tol:
                break

    return x, lambda0


# print(pow_it(get_A3(), np.array([1,1,1]), 0.001, 10000))
# print(pow_it(get_B3(), np.array([1,1,1]), 0.001, 10000))


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, a maxit dimensional numpy array containing \
    all the iterates.
    """
    m, _ = A.shape
    if store_iterations:
        x = np.array([x0])
        l = np.array([])
        for k in range(maxit):
            x_k = solve_LUP(A-mu*np.identity(m), x[-1])
            x_k /= np.linalg.norm(x_k)
            x = np.append(x, x_k)
            l_k = x_k.dot(A @ x_k)
            l = np.append(l, l_k)
            if np.linalg.norm(A @ x_k - l_k * x_k) < tol:
                break
    else:
        if np.imag(mu) == 0:
            mu = np.real(mu)
            x = x0
            for k in range(maxit):
                x = solve_LUP(A-mu*np.identity(m), x)
                x /= np.linalg.norm(x)
                l = x.dot(A @ x)
                if np.linalg.norm(A @ x - l * x) < tol:
                    break
        else:
            x_hat0 = np.append(np.real(x0), np.imag(x0))
            mu_r, mu_i = np.real(mu), np.imag(mu)
            B = np.zeros((2*m, 2*m))
            B[:m, :m] = A
            B[m:, m:] = A
            B[:m, m:] = mu_i * np.identity(m)
            B[m:, :m] = -mu_i * np.identity(m)
            x_hat, l = inverse_it(B, x_hat0, mu_r, 1.0e-6, 1000)
            x = x_hat[:m] + 1j*x_hat[m:]
            # for k in range(maxit):
            #     x_hat = solve_LUP(B-mu_r*np.identity(m), x_hat)
            #     x_hat /= np.linalg.norm(x)
            #     l = (x_hat[:m] + 1j*x_hat[m:]).dot(A @ (x_hat[:m] + 1j*x_hat[m:]))
            #     if np.linalg.norm(B @ x_hat - l * )
    return x, l


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    m, _ = A.shape
    if store_iterations:
        x = np.array([x0])
        l = np.array([x0.dot(A @ x0)])
        for k in range(maxit):
            x_k = solve_LUP(A-l*np.identity(m), x[-1])
            x_k /= np.linalg.norm(x_k)
            x = np.append(x, x_k)
            l_k = x_k.dot(A @ x_k)
            l = np.append(l, l_k)
            if np.linalg.norm(A @ x_k - l_k * x_k) < tol:
                break
    else:
        x = x0
        l = x0.dot(A @ x0)
        for k in range(maxit):
            x = solve_LUP(A-l*np.identity(m), x)
            x /= np.linalg.norm(x)
            l = x.dot(A @ x)
            if np.linalg.norm(A @ x - l * x) < tol:
                break
    
    return x, l


def pure_QR(A, maxit, tol, return_norms=False, return_iterations=False, special_criteria=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param norms: bool, determines whether to return the norms in 1a of coursework.
    False by default
    :param return_iterations: bool, determines whether to return the array containing eigenvalue iterations
    False by default
    :param special_criteria: bool, determines whether to use the convergence criteria for arrays with complex
    eigenvalues.

    :return Ak: the result
    :return matrix_norms: k dimensional numpy array, containing the norm described in 1a at each iteration,
    where k is the number of iterationstaken to converge..
    :return iter_arr: kxm dimensional numpy array, containing the eigenvalues at each iteration on its rows
    where k is the number of iterationstaken to converge.
    """
    m, _ = A.shape
    Ak = A.copy()
    iter = maxit
    if return_norms:
        matrix_norms = np.zeros(maxit)
    if return_iterations:
        iter_arr = np.zeros((maxit, m))
    for k in range(maxit):
        Q, R = householder_qr(Ak)
        Ak = R @ Q
        if return_norms:
            As = np.tril(Ak, k=-1)
            matrix_norms[k] = np.linalg.norm(As)
        if return_iterations:
            iter_arr[k, :] = np.diag(Ak)
        if special_criteria:
            #check all entries below lower subdiagonal are zero
            if np.linalg.norm(Ak[np.tril_indices(m, -2)])/m**2 < tol:
                #get idxs of all non zero entries on lower subdiagonal
                non_zero_idxs = np.argwhere(np.abs(np.diag(Ak, -1)) > tol).flatten()
                #find difference between idxs
                idx_diff = non_zero_idxs[1:] - non_zero_idxs[:-1]
                #check for no consecutive non zero entries
                if not 1 in idx_diff:
                    iter = k
                    break

        else:
            if np.linalg.norm(Ak[np.tril_indices(m, -1)])/m**2 < tol:
                iter = k
                break
    if return_norms and return_iterations:
        matrix_norms = matrix_norms[:iter+1]
        iter_arr = iter_arr[:iter+1, :]
        return Ak, matrix_norms, iter_arr
    elif return_norms:
        matrix_norms = matrix_norms[:iter+1]
        return Ak, matrix_norms
    elif return_iterations:
        iter_arr = iter_arr[:iter+1, :]
        return Ak, iter_arr
    else:
        return Ak
