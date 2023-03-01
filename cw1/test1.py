'''Tests for the question 1 of the coursework 1.'''
import pytest
from q1 import basis_coeffs, construct_A
import numpy as np
import matplotlib.pyplot as plt


@pytest.mark.parametrize('m, n, use_mgs', [(20, 20, True), (50, 50, True), (100, 100, True), (20, 20, False), (50, 50, False), (100, 100, False)])
def test_basis_coeffs(m, n, use_mgs):
    """
    test basis_coeffs function by fitting a function to a linear combination of basis vectors
    :param m: integer height of test matrix
    :param n: integer width of test matrix
    :param use_mgs: boolean to indicate whether to use mgs if True, or householder if False
    """
    points = np.linspace(0, 1, m)
    A = construct_A(points, n)
    # define a function fn that is a linear combination of basis functions phi_i
    def fn(x):
        coeffs = np.random.randint(1, 10, n)
        delta_x = 1/(n-1)
        sum_functions = sum([coeffs[i] * np.exp(-(x-i*delta_x)**2 / delta_x**2) for i in range(n)])
        return sum_functions
    fx = np.array([fn(x) for x in points])
    # find approximations to weights of basis functions
    x = basis_coeffs(points, n, fx, use_mgs=use_mgs)

    ###UNCOMMENT BELOW TO SEE PLOT OF TEST FUNCTION AND APPROXIMATED FUNCTION
    # plt.figure()
    # plt.plot(points, fx, "k-", label="$f(x)$", linewidth = 8)
    # if use_mgs:
    #     label = "mgs fitted"
    # else:
    #     label = "householder fitted"
    # plt.plot(points, np.dot(A, x), "r-", label=label, alpha=1)
    # plt.title(f"error = {np.linalg.norm(np.dot(A, x) - fx)}")
    # plt.legend(loc="upper right")
    # plt.show()
    
    assert(np.linalg.norm(np.dot(A, x) - fx)) < 1.0e-6 # check the least squares approximation is close to real values


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
