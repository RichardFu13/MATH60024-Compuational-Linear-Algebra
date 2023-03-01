import numpy as np
import matplotlib.pyplot as plt
from cla_utils import pure_QR, inverse_it, hessenberg
from cw3.q1 import QR_shift
import numpy.random as random

A3 = np.loadtxt('cw3/A3.dat') #load in text file

### 1a
Ak, norms = pure_QR(A3, 10000, 1.0e-6, return_norms=True)

#semilogy plot
plt.figure()
plt.semilogy(norms, "r-")
plt.xlabel("Iteration number")
plt.ylabel("$\||A_s\||$")
plt.show()

#verify these are indeed eigenvalues
m, _ = A3.shape
pure_QR_eigs = np.diag(Ak)
eig_errors = np.zeros(m)
for k in range(m):
    #obtain evector by inverse_it
    v_k, _ = inverse_it(A3, np.ones(m), pure_QR_eigs[k], 1.0e-6, 1000)
    #store error from Av - lambda*v in eig_errors
    eig_errors[k] = np.linalg.norm(A3 @ v_k - pure_QR_eigs[k] * v_k)
#print norm of these errors
print(f"norm of residual errors is {np.linalg.norm(eig_errors)}")


### 1b
# generate iterations using pure_QR
_, iter_arr = pure_QR(A3, 10000, 1.0e-6, return_iterations=True)
colours = np.array(["red", "blue", "lawngreen", "darkorange", "black", "deeppink"])
fig, axs = plt.subplots(1, 2, figsize=(20, 7))
for k in range(m):
    iter_error_arr = np.abs(iter_arr[:, k] - pure_QR_eigs[k] * np.ones_like(iter_arr[:, k])) + 1.0e-6
    #plot errors
    axs[0].semilogy(iter_error_arr,  color=colours[k], label=f"entry {k+1}{k+1}", alpha=0.5)
    #plot absolute value
    axs[1].plot(np.abs(iter_arr[:, k]), color=colours[k], label=f"entry {k+1}{k+1}", alpha=0.5)
axs[0].set_xlabel("iteration number")
axs[0].set_ylabel("error")
axs[0].legend()
axs[1].legend()
axs[0].set_title("Errors in convergence of diagonal entries")
axs[1].set_title("Absolute value of diagonal entries")
plt.show()


### 1h
# generate iterations using QR_shift
eigs, iter_arr = QR_shift(A3, 10000, 1.0e-6, return_iterations=True)
colours = np.array(["red", "blue", "lawngreen", "darkorange", "black", "deeppink"])
fig, axs = plt.subplots(1, 2, figsize=(20, 7))
for k in range(m):
    iter_error_arr = np.abs(iter_arr[:, k] - eigs[k] * np.ones_like(iter_arr[:, k])) + 1.0e-6
    #plot errors
    axs[0].semilogy(iter_error_arr,  color=colours[k], label=f"entry {k+1}{k+1}", alpha=0.5)
    #plot absolute value
    axs[1].plot(np.abs(iter_arr[:, k]), color=colours[k], label=f"entry {k+1}{k+1}", alpha=0.5)
axs[0].set_xlabel("iteration number")
axs[0].set_ylabel("error")
axs[0].legend()
axs[1].legend()
axs[0].set_title("Errors in convergence of diagonal entries")
axs[1].set_title("Absolute value of diagonal entries")
plt.show()



### 1i
m_arr = np.arange(50, 100, 5)
iter_pure_QR = np.zeros(m_arr.size)
iter_shifted_QR = np.zeros(m_arr.size)
for idx, m in enumerate(m_arr):
    A = random.randn(m, m)
    #generate symmetric matrix
    B = A.T @ A
    #make hessenberg form
    hessenberg(B)
    #get the iterations for both methods
    _, iterations_pure_QR = pure_QR(B, 10000, 1.0e-6, return_iterations=True)
    _, iterations_shifted_QR = QR_shift(B, 10000, 1.0e-6, return_iterations=True)
    iter_pure_QR[idx] = iterations_pure_QR.shape[0]
    iter_shifted_QR[idx] = iterations_shifted_QR.shape[0]
    #determine number of iterations for all eigenvalues converged to 3dp
    for k in range(1, iterations_pure_QR.shape[0]):
        if np.all(np.around(iterations_pure_QR[k, :], 3) == np.around(iterations_pure_QR[k,:], 3)):
            iter_pure_QR[idx] = k
    for k in range(1, iterations_shifted_QR.shape[0]):
        if np.all(np.abs(iterations_shifted_QR[k, :] - iterations_shifted_QR[k,:]) < 0.001):
            iter_shifted_QR[idx] = k

#plot the number of iterations against m
plt.figure(figsize=(20,7))
plt.title("Convergence of pure_QR and QR_shift eigenvalues to 3 decimal places")
plt.semilogy(m_arr, iter_pure_QR, "r-", label="pure QR")
plt.semilogy(m_arr, iter_shifted_QR, "b-", label="shifted QR")
plt.xlabel("m")
plt.ylabel("iterations")
plt.legend()
plt.show()

