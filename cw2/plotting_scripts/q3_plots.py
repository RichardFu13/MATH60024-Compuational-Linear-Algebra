import numpy as np
import matplotlib.pyplot as plt
from cw2.q3 import image_simulator, construct_D, iterative_solve


### 3c
#plot simulated images for various s

fig, axs = plt.subplots(2, 3, figsize=(20,7))
s_array = np.array([0.1, 3, 1, 1.5, 0.5, 4])
fig.suptitle("Simulated random images for different $s$, $N=50$", fontsize=20)
for idx, s in enumerate(s_array):
    w = np.random.randn(50**2)
    u_hat = image_simulator(50, s, w, banded_solve=True)
    u = u_hat.reshape((50, 50)) #reshape to a matrix
    axs[idx % 2, idx % 3].imshow(u, cmap="gray", vmin=0, vmax=1)
    axs[idx % 2, idx % 3].set_title(f"$s$ = {s}")
plt.show()


### 3d
#plot the time elapsed against N for banded and unbanded solve.

N_array = np.arange(0, 50)
banded_times = np.array([])
unbanded_times = np.array([])
for N in N_array:
    w = np.random.randn(N**2)
    D = construct_D(N, 1)
    _, time_banded = image_simulator(N, 1, w, banded_solve=True, return_time=True)
    _, time_unbanded = image_simulator(N, 1, w, banded_solve=False, return_time=True)
    banded_times = np.append(banded_times, time_banded)
    unbanded_times = np.append(unbanded_times, time_unbanded)
plt.figure()
plt.title("Runtime of solvers against matrix size $N$", fontsize=15)
plt.plot(N_array, banded_times, "r-", label="banded")
plt.plot(N_array, unbanded_times, "b-", label="unbanded")
plt.xlabel("$N$")
plt.ylabel("time")
plt.legend(loc="upper center")
plt.show()


### 3h
rho_array = np.arange(0.2, 2.2, 0.1)
nu_array = np.arange(0.2, 2.2, 0.1)
iter_array = np.zeros((rho_array.size, nu_array.size))

#plot heatmaps for different N displaying the iteration count of rho and nu combinations
fig, axs = plt.subplots(1,3, figsize=(15,7))
fig.suptitle("Heat Map plot of iterations for varying N", fontsize=20)
for idx, N in enumerate(np.array([10, 20, 40])):
    u_0 = np.zeros(N**2)
    w = np.random.randn(N**2)
    for i, rho in enumerate(rho_array):
        for j, nu in enumerate(nu_array):
            _, iter_array[i, j] = iterative_solve(N, u_0, 1, rho, nu, w, 1.0e-6, return_iter=True)
    axs[idx].set_title(f"N={N}", fontsize=15)
    axs[idx].set_xlabel(r"$\nu$")
    axs[idx].set_ylabel(r"$\rho$")
    axs[idx].imshow(iter_array, extent=[0.2, 2, 2, 0.2], cmap="gray", vmin=0, vmax=30)
plt.show()

#using optimal choices of rho and nu = 1
N_values = np.arange(10, 110, 10)
iter_values = np.array([])
for N in N_values:
    u_0 = np.zeros(N**2)
    w = np.random.randn(N**2)
    _, iter_count = iterative_solve(N, u_0, 1, 1, 1, w, 1.0e-6, return_iter=True)
    iter_values = np.append(iter_values, iter_count)

#plot our number of iterations against N
plt.figure(figsize=(20,7))
plt.xlabel("N")
plt.ylabel("no. iterations")
plt.plot(N_values, iter_values, "r-")
plt.show()
