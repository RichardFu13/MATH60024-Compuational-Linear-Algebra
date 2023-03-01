import numpy as np
import matplotlib.pyplot as plt
from cw3.q2 import get_A, get_callback
from cw3.q1 import QR_shift
from cla_utils import pure_QR, GMRES


### 2c
m_arr = np.arange(4, 50, 2)
eig_err = np.zeros(m_arr.size)

for idx, m in enumerate(m_arr):
    A = get_A(m)
    #calculate eig from formula derived, smallest to largetst
    eigs = np.sort(2 * (np.cos(np.arange(1,m+1)*np.pi/(m+1)) - 1))
    Ak = pure_QR(A, 1000, 1.0e-6)
    #calculate eig using QR algorithm , smallest to largetst
    QR_eigs = np.sort(np.diag(Ak, k=0))
    QR_eigs = np.sort(QR_shift(A, 10000, 1.0e-6))
    #error in eigenvalues
    eig_err[idx] = np.linalg.norm(eigs - QR_eigs)

plt.figure()
plt.plot(m_arr, eig_err, "r-")
plt.title("Eigenvalue error from derived formula")
plt.xlabel("m")
plt.ylabel("error")
plt.show()


# ### 2d
fig, axs = plt.subplots(2, 3, figsize=(20,7))
m_arr2 = np.array([5, 50, 30, 40, 20, 60])
fig.suptitle("Distribution of eigenvalues for different $m$", fontsize=20)
for idx, m in enumerate(m_arr2):
    A = get_A(m)
    eigs = np.sort(2 * (np.cos(np.arange(1,m+1)*np.pi/(m+1)) - 1))
    axs[idx % 2, idx % 3].plot(np.zeros(eigs.size), eigs, "r.", alpha=0.3)
    axs[idx % 2, idx % 3].set_title(f"$m$ = {m}")
    axs[idx % 2, idx % 3].set_ylim(-5, 0.5)
plt.show()


### 2e
m_arr = np.arange(10, 110, 10)
colour_arr = np.array(["red", "blue", "green", "orange", "black", "aquamarine", "brown", "violet", "purple", "magenta"])
plt.figure(figsize=(20,7))
for idx, m in enumerate(m_arr):
    A = get_A(m)
    x = np.random.randn(m)
    b = A @ x
    GMRES(A, b, 1000, 1.0e-6, callback=get_callback(x))
    #plot error norm data from callback function
    plt.plot(np.loadtxt("cw3/callback.dat"), color=colour_arr[idx], label=f"m={m}")
plt.legend()
plt.title("Callback errors")
plt.xlabel("iteration number")
plt.ylabel("error")
plt.show()
