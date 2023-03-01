'''Tests for the tenth exercise set.'''
import pytest
import cla_utils
from numpy import random
import numpy as np
from cw3.q2 import get_callback


@pytest.mark.parametrize('m, k', [(20, 4), (40, 20), (70, 13)])
def test_arnoldi(m, k):
    A = random.randn(m, m) + 1j*random.randn(m, m)
    b = random.randn(m) + 1j*random.randn(m)

    Q, H = cla_utils.arnoldi(A, b, k)
    assert(Q.shape == (m, k+1))
    assert(H.shape == (k+1, k))
    assert(np.linalg.norm((Q.conj().T)@Q - np.eye(k+1)) < 1.0e-6)
    assert(np.linalg.norm(A@Q[:,:-1] - Q@H) < 1.0e-6)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_GMRES(m):
    A = random.randn(m, m)
    b = random.randn(m)

    x, _ = cla_utils.GMRES(A, b, maxit=1000, tol=1.0e-3)
    assert(np.linalg.norm(np.dot(A, x) - b) < 1.0e-3)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_GMRES_callback(m):
    #clear the file
    f = open("cw3/callback.dat", "w")
    f.close()
    A = random.randn(m, m)
    x = random.randn(m)
    b = A @ x
    cla_utils.GMRES(A, b, maxit=1000, tol=1.0e-3, callback=get_callback(x))
    #load the text file
    text_array = np.loadtxt("cw3/callback.dat")
    #check file isn't empty
    assert(text_array.size > 0)

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
