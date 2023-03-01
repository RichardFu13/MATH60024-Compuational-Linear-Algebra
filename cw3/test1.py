import pytest
import numpy as np
import numpy.random as random
from q1 import QR_shift, block_matrix_eigs
from cla_utils import inverse_it

A3 = np.loadtxt('cw3/A3.dat') #load in text file

### 1e
@pytest.mark.parametrize('m', [10, 20, 30])
def test_block_matrix_eigs(m):
    #construct Z of form A'
    Z = np.triu(random.randn(m, m))
    #stepsize
    step = random.randint(2, m)
    idxs = np.arange(0, m-1, step)
    #fill sub diagonal
    for idx in idxs:
        Z[idx+1, idx] = random.randn()
    #form an orthogonal Q
    Q, _ = np.linalg.qr(random.randn(m,m))
    #create A of desired form
    A = Q.T.conj() @ Z @ Q
    eigs = block_matrix_eigs(A, 10000, 1.0e-6)
    for k in range(m):
        v, _ = inverse_it(A, np.ones(m), eigs[k], 1.0e-6, 10000)
        assert(np.linalg.norm(A @ v - eigs[k] * v) < 1.0e-6)


### 1g
def test_QR_shift_A3():
    m, _ = A3.shape
    pure_QR_eigs = QR_shift(A3, 10000, 1.0e-6)
    # iter_eigs = np.zeros(m)
    for k in range(m):
        v, _ = inverse_it(A3, np.ones(m), pure_QR_eigs[k], 1.0e-6, 10000)
        assert(np.linalg.norm(A3 @ v - pure_QR_eigs[k] * v) < 1.0e-6)


@pytest.mark.parametrize('m', [10, 20, 30])
def test_QR_shift(m):
    #construct Z
    Z = np.zeros((m,m))
    np.fill_diagonal(Z, random.randn(m))
    #form an orthogonal Q
    Q, _ = np.linalg.qr(random.randn(m,m))
    #create A of desired form
    A = Q.T.conj() @ Z @ Q
    pure_QR_eigs = QR_shift(A, 10000, 1.0e-6)
    for k in range(m):
        v, _ = inverse_it(A, np.ones(m), pure_QR_eigs[k], 1.0e-6, 10000)
        assert(np.linalg.norm(A @ v - pure_QR_eigs[k] * v) < 1.0e-6)
