import numpy as np
import matplotlib.pyplot as plt
from cw2.q2 import MGS_solve_ls_modified, MGS_solve_ls
from cla_utils import GS_modified, householder_ls

A2 = np.loadtxt('cw2/A2.dat') #load in text file

### 2b
MGS_errors = np.array([])
householder_errors = np.array([])
#repeat 100 times to find errors for MGS and householder
for n in range(100):
    #set seed
    np.random.seed(n)
    x_star = np.random.randn(20)
    b = A2 @ x_star
    error_MGS = np.linalg.norm(MGS_solve_ls(A2, b) - x_star)
    error_householder = np.linalg.norm(householder_ls(A2, b) - x_star)
    MGS_errors = np.append(MGS_errors, error_MGS)
    householder_errors = np.append(householder_errors, error_householder)


#plot the errors
plt.figure(figsize=(20,7))
plt.title("Errors in numerical least squares solution", fontsize=15)
plt.plot(MGS_errors, "r-", label="MGS_solve_ls")
plt.plot(householder_errors, "b-", label="householder_ls")
plt.yscale("log")
plt.ylabel("error")
plt.legend(loc="center right")
plt.show()

### 2e
MGS_modified_errors = np.array([])
#repeat 100 times to find errors for MGS_modified
for n in range(100):
    np.random.seed(n)
    x_star = np.random.randn(20)
    b = A2 @ x_star
    error_MGS_modified = np.linalg.norm(MGS_solve_ls_modified(A2, b) - x_star)
    MGS_modified_errors = np.append(MGS_modified_errors, error_MGS_modified)


#plot the errors in least squares solution for all three methods
plt.figure(figsize=(20,7))
plt.title("Errors in numerical least squares solution", fontsize=15)
plt.plot(MGS_errors, "r-", label="MGS_solve_ls")
plt.plot(MGS_modified_errors, "b-", label="MGS_solve_ls_modified")
plt.plot(householder_errors, "g-", label="householder_ls")
plt.yscale("log")
plt.ylabel("error")
plt.legend(loc="center right")
plt.show()


### 2f
errors_1 = np.array([])
errors_2 = np.array([])
errors_3 = np.array([])
#repeat 100 times to find errors for all three methods
for n in range(100):
    x_star = np.random.randn(20)
    #find r_2 such that it is not in range space
    r_1 = np.random.randn(100)
    A2_aug = np.hstack((A2, r_1.reshape((100, 1))))
    GS_modified(A2_aug)
    #slice out Q^* r_1 to get r_2
    r_2 = A2_aug[:,-1]
    b = A2 @ x_star + r_2
    error_MGS = np.linalg.norm(MGS_solve_ls(A2, b) - x_star)
    error_MGS_modified = np.linalg.norm(MGS_solve_ls_modified(A2, b) - x_star)
    error_householder = np.linalg.norm(householder_ls(A2, b) - x_star)
    errors_1 = np.append(errors_1, error_MGS)
    errors_2 = np.append(errors_2, error_MGS_modified)
    errors_3 = np.append(errors_3, error_householder)


#plot the errors in least squares solution for all three methods
plt.figure(figsize=(20,7))
plt.title("Errors in numerical least squares solution", fontsize=15)
plt.plot(errors_1, "r-", label="MGS_solve_ls")
plt.plot(errors_2, "b-", label="MGS_solve_ls_modified")
plt.plot(errors_3, "g-", label="householder_ls")
plt.yscale("log")
plt.ylabel("error")
plt.legend(loc="center right")
plt.show()
