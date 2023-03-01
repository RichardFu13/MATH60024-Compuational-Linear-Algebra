from cw2.q1 import get_rho, random_matrix
import numpy as np
import matplotlib.pyplot as plt

#loglog plot of rho against n
n_values = np.arange(10, 1000, 10)
rho_values = np.array([get_rho(random_matrix(n)) for n in n_values])
plt.figure(figsize=(20,7))
plt.loglog(n_values, rho_values, "k.")
plt.plot(n_values, n_values**(3/4)/3, "r-")
plt.xlabel("$n$", fontsize=15)
plt.ylabel(r"$\rho$", fontsize=15)
plt.ylim(1, 10**2)
plt.show()
