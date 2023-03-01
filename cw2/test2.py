import pytest
import numpy as np
import matplotlib.pyplot as plt
from q2 import MGS_solve_ls

@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)])
def test_MGS_solve_ls(m, n):
    np.random.seed(8473*m + 9283*n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = MGS_solve_ls(A, b)
    #!!!change test param to b

    #check normal equation residual
    assert(np.linalg.norm(np.dot(A.T, np.dot(A, x) - b)) < 1.0e-6)

@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)])
def test_MGS_solve_ls_modified(m, n):
    np.random.seed(8473*m + 9283*n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = MGS_solve_ls(A, b)
    #!!!change test param to b

    #check normal equation residual
    assert(np.linalg.norm(np.dot(A.T, np.dot(A, x) - b)) < 1.0e-6)
