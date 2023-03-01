import pytest
import numpy as np
import numpy.random as random
from cw3.q3 import get_D, get_evecs, rank_def_LS, create_rd_matrix
from cla_utils import householder_qr


### 3c
@pytest.mark.parametrize('m', [10, 20, 30])
def test_get_D(m):
    U, _ = np.linalg.qr(random.randn(m,m))
    V, _ = np.linalg.qr(random.randn(m,m))
    diag = np.sort(np.abs(np.random.randn(m)))[::-1]
    D = np.diag(diag)
    A = U @ D @ V.conj().T
    D0 = get_D(A)
    #check diagonality of D0
    assert(np.linalg.norm(D0[np.tril_indices(m, -1)]) < 1.0e-6)
    assert(np.linalg.norm(D0[np.triu_indices(m, 1)]) < 1.0e-6)
    #check D0 close to D
    assert(np.linalg.norm(np.diag(D)-np.diag(D0)) < 1.0e-6)


### 3d
@pytest.mark.parametrize('m', [10, 20, 30])
def test_get_evecs(m):
    #construct A
    U, _ = np.linalg.qr(random.randn(m,m))
    V, _ = np.linalg.qr(random.randn(m,m))
    diag = np.sort(np.abs(np.random.randn(m)))[::-1]
    D = np.diag(diag)
    A = U @ D @ V.conj().T
    #find U, D, and V
    D0, H = get_D(A, return_H=True)
    evecs_H = get_evecs(D0, H)
    V0 = np.sqrt(2) * evecs_H[:m, :m]
    U0 = np.sqrt(2) * evecs_H[m:, :m]
    #check unitary
    assert(np.linalg.norm(V0 @ V0.conj().T - np.identity(m)) < 1.0e-6)
    assert(np.linalg.norm(U0 @ U0.conj().T - np.identity(m)) < 1.0e-6)
    #check our algorithm for U, D, V correct
    assert(np.linalg.norm(U0 @ D0 @ V0.conj().T - A) < 1.0e-6)


@pytest.mark.parametrize('m', [10, 20, 30])
def test_get_evecs_2(m):
    #construct A
    U, _ = np.linalg.qr(random.randn(m,m))
    V, _ = np.linalg.qr(random.randn(m,m))
    diag = np.sort(np.abs(np.random.randn(m)))[::-1]
    D = np.diag(diag)
    A = U @ D @ V.conj().T
    D0, H = get_D(A, return_H=True)
    # get evectors and eigs of H
    evecs_H, eigs_H = get_evecs(D0, H, return_eigs=True)
    for k in range(2*m):
        v_k = evecs_H[:, k]
        #check they are indeed evectors of H
        assert(np.linalg.norm(H @ v_k - eigs_H[k] * v_k) < 1.0e-6)


### 3e
@pytest.mark.parametrize('m', [10, 20, 30])
def test_rank_def_LS(m):
    #create rank deficient matrix
    remove_amount = random.randint(1, m)
    A = create_rd_matrix(m, remove_amount)
    b = random.randn(m)
    x = rank_def_LS(A, b)
    Q, _ = householder_qr(A.T)
    null_space = Q[:,-remove_amount:]
    #check x is orthgonal to null_space
    for idx in range(null_space.shape[1]):
        assert(null_space[:, idx].dot(x) < 1.0e-4)
