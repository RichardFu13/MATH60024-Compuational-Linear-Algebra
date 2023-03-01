import numpy as np
from cla_utils import householder, GS_modified, solve_U, GS_classical
import matplotlib.pyplot as plt


#1)a)
def construct_A(points, n):
    """
    Given a set of m points construct the matrix A required to set up a least
    problem described in question 1.

    :param points: an m-dimensional numpy array of points
    :param n: an integer, the number of basis functions to use in the least squares
    problem

    :return A: an mxn-dimensional numpy array where A[i, j] is the ith point in
    points evaluated by the jth basis function.
    """
    m = points.size
    delta_x = 1/(n-1)
    A = np.zeros((m, n))
    for i in range(n):
        x_i = i*delta_x
        phi_i = lambda x: np.exp(-(x-x_i)**2 / delta_x**2)
        A[:, i] = phi_i(points)

    return A


def basis_coeffs(points, n, fx, use_mgs = False):
    """
    Given a set of m points and their observed function values, find the basis
    coefficients obtained from the least squares estimate fitting the basis of size n.

    :param points: an m-dimensional numpy array of points
    :param n: an integer, the number of basis functions to fit our function to
    :param fx: an m-dimensional numpy array of the observed function values at each
    point in points
    :param use_mgs: a boolean to indicate whether to use mgs when True or householder
    when False. Defaults to False

    :return x: an n-dimensional numpy array of basis coefficients where the ith entry
    corresponds to the ith basis function.
    """
    m = points.size
    A = construct_A(points, n)

    if use_mgs: # set up Q_adj and R using mgs
        A_copy = A.copy()
        R = GS_modified(A_copy)
        Q_adj = A_copy.T.conj()
    
    else: # set up Q_adj and R using householder
        A_hat = np.column_stack((A, np.identity(m)))
        R_hat = householder(A_hat, kmax=m)
        R = R_hat[:n,:n]
        Q_adj = R_hat[:n, n:n+m]

    x = solve_U(R, Q_adj @ fx).flatten() # solve the least squares problem using back substitution
    
    return x


#1)b)
def draw_plots(m, n, random=True):
    """
    Draw the plots for the function f(x) as well as our least squares estimate
    fitted function over domain (0,1).
    :param m: an integer, the number of points to estimate with
    :param n: an integer, the number of basis functions to fit our function to
    :param random: a boolean to indicate whether to use a random set of points
    if True, or an equi-distant set of points if False
    """
    if random:
        points = np.sort(np.random.uniform(0, 1, m))
    else:
        points = np.linspace(0, 1, m)
    def fn(x):
        """
        Define as (2) is in the questions.
        """
        if np.abs(x-0.5) < 0.25:
            return 1
        else:
            return 0

    fx = np.array([fn(x) for x in points])
    A = construct_A(points, n)

    x_householder = basis_coeffs(points, n, fx, use_mgs=False)
    x_mgs = basis_coeffs(points, n, fx, use_mgs=True)

    #uniform plot
    fig, axs = plt.subplots(1, 2, figsize=(30,7))
    if random:
        fig.suptitle(f"Random points between 0 and 1, M={m}, N={n}", fontsize=20)
    else:
        fig.suptitle(f"Equidistant points between 0 and 1, M={m}, N={n}", fontsize=20)
    axs[0].plot(points, fx, "k-", label="$f(x)$")
    axs[1].plot(points, fx, "k-", label="$f(x)$")
    axs[0].plot(points, np.dot(A, x_householder), "r-", label="householder fitted")
    axs[1].plot(points, np.dot(A, x_mgs), "b-", label="mgs fitted")
    axs[0].set_title(f"householder error = {np.linalg.norm(np.dot(A, x_householder) - fx)}", fontsize=15)
    axs[1].set_title(f"mgs error = {np.linalg.norm(np.dot(A, x_mgs) - fx)}", fontsize=15)
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    plt.show()


###UNCOMMENT BELOW TO PLOT FOR 1)b)
# draw_plots(100, 10, random=False)
# draw_plots(100, 50, random=False)
# draw_plots(100, 10, random=True)
# draw_plots(100, 50, random=True)
