import numpy as np


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    eigs = np.linalg.eig(A.conj().T @ A)[0]
    o2norm = np.sqrt(max(eigs))

    return o2norm


def verify_inequality(A):
    m, n = A.shape
    x = np.random.random(n)

    return np.linalg.norm(A@x) <= operator_2_norm(A) * np.linalg.norm(x)

###UNCOMMENT TO VERIFY
# A0 = np.random.randn(50, 10)
# print(verify_inequality(A0))
# A1 = np.random.randn(5, 15)
# print(verify_inequality(A1))
# A2 = np.random.randn(30, 30)
# print(verify_inequality(A2))


def verify_theorem(A, B):
    return operator_2_norm(A@B) <= operator_2_norm(A) * operator_2_norm(B)

###UNCOMMENT TO VERIFY
# B0 = np.random.randn(10, 20)
# print(verify_theorem(A0, B0))
# B1 = np.random.randn(15, 5)
# print(verify_theorem(A1, B1))
# B2 = np.random.randn(30, 30)
# print(verify_theorem(A2, B2))


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """

    eigs = sorted(np.linalg.eig(A.conj().T @ A)[0])
    ncond = np.sqrt(eigs[-1]/eigs[0])

    return ncond


#EXERCISE 3.4)
def lower_triag_multiplication(A, B):
    m, _ = A.shape
    C = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            C[j:m, i] += B[j, i]*A[j:m, j]
    return C
