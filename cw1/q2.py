import numpy as np
from cla_utils import householder, householder_qr
import matplotlib.pyplot as plt

A1 = np.loadtxt('cw1/A1.dat') #load in text file

#2a)
# PRINT THE QR DECOMPOSITION OF A1
print(f"A1 = {A1}")
Q, R = householder_qr(A1)
print(f"Q = {Q}")
print(f"R={R}")

# CALCULATE THE RANK OF A1 BY COUNTING NON-ZERO DIAGONAL ENTRIES OF R
m, n = R.shape
non_zero_count = 0 # initialize counter
for i in range(n):
    if np.abs(R[i, i]) > 1.0e-6: # non-zero diagonal entries (larger than 1.0e-6)
        non_zero_count += 1
print(f"non-zero diagonal entries of R = {non_zero_count}")


#2d)
def rank_from_householder_swap(A, s):
    """
    Calculate the rank of a matrix by finding R' using householder with swaps and reduced_tol.
    :param A: mxn matrix we want to find the rank of
    :s: float to determine the value of reduced_tol to pass into householder

    :return r: the rank (height) of R' and therefore the rank of A
    """
    R = householder(A.copy(), swap=True, reduced_tol=s)
    r, n = R.shape # number of rows of R is the r, the rank of A
    return r


#2e)
def create_rd_matrix(m, n, remove_amount):
    """
    Create an mxn rank-deficient matrix by setting some rows to 0.
    :param m: integer of the height of our matrix
    :param n: integer of the width of our matrix
    :remove_amount: integer<=m of the number of rows to set to 0

    :return B: an mxn rank deficient matrix
    """
    A = np.random.random((m, n))
    Q, R = householder_qr(A)
    idxs = np.random.choice(np.arange(m), replace=False, size=remove_amount)
    R[idxs] = np.zeros(n) # set rows to 0
    B = Q @ R
    return B


def rd_ranks_and_errors(n, tolerance_array):
    """
    Create size n rank-deficient matrices, and calculate the ranks of these and the errors
    when estimating the rank for specific tolerances. Assists with plotting.
    :param n: integer number of deficient matrices to create
    :tolerance_array: a numpy array of tolerances to estimate rank with

    :return ranks: an n-array with entries corresponding to the ranks of the deficient matrics
    :return errors: an pxn numpy array where p is the len(tolerance_array), with i,jth entry
    the error estimating matrix j's rank with the ith tolerance in tolerance_array.
    """
    ranks = np.zeros(n)
    errors = np.zeros((len(tolerance_array), n))
    for i in range(n):
        m = np.random.randint(100, 300)
        n = np.random.randint(50, 100)
        remove_amount = int(np.random.uniform(0.1, 0.9)*m)
        B = create_rd_matrix(m, n, remove_amount)
        real_rank = np.linalg.matrix_rank(B)
        ranks[i] = real_rank
        for j in range(len(tolerance_array)):
            calculated_rank = rank_from_householder_swap(B, tolerance_array[j])
            errors[j, i] = calculated_rank - real_rank
    return ranks, errors


##UNCOMMENT BELOW TO SEE PLOTS FOR 2)e)
# fig, axs = plt.subplots(1, 4, figsize=(40,6), constrained_layout=True)
# tolerances = np.array([1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8])
# ranks, errors= rd_ranks_and_errors(300, tolerances)
# for i in range(len(tolerances)): #plot for each tolerance
#     non_zero = np.count_nonzero(errors[i])
#     axs[i].scatter(ranks, errors[i], marker="o", alpha=0.5)
#     axs[i].set_xlabel("matrix rank")
#     axs[i].set_ylabel("calculated rank - actual rank")
#     axs[i].set_title(f"tol = {tolerances[i]}, accuracy = {np.around((len(errors[i]) - non_zero)*100 / len(errors[i]),2)}%")
# plt.show()
