from cla_utils import householder
from q2 import rank_from_householder_swap
import pytest
import numpy as np

A1 = np.loadtxt('cw1/A1.dat') #read in A1

#2)c)
def test_householder_swap_1():
    """test for householder with column swapping for 2x2 known QR decomp"""
    matrix = np.array([[1.0,2.0], [2.0,4.0]]) # test matrix
    correct_R = np.array([[-4.47213595, -2.23606798], [0, 0]]) # hand calculated R
    # check R is correct when using swap
    assert(np.allclose(correct_R, householder(matrix, swap=True)))


def test_householder_swap_2():
    """test for householder with column swapping for 3x3 known QR decomp"""
    matrix = np.array([[1.0,3.0,2.0], [2.0,1.0,4.0],[1.0,1.0,2.0]]) # test matrix
    correct_R = np.array([[-2*np.sqrt(6), -np.sqrt(6), -np.sqrt(6)], [0, np.sqrt(5), 0], [0, 0, 0]]) # hand calculated R
    # check R is correct when using swap
    assert(np.allclose(correct_R, householder(matrix, swap=True)))


#2)d)
def test_rank_from_householder_swap():
    """test for householder with column swapping and tolerance calculating the rank of A1"""
    Q = A1.copy()
    actual_rank = 4
    assert(actual_rank == rank_from_householder_swap(Q, 1.0e-6))


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
