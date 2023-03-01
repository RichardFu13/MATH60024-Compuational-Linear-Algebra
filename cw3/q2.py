import numpy as np

### 2a
def get_callback(x_sol):
    """
    Callback interface for GMRES.
    :param x_sol: an m dimensional numpy array, true value of solution x
    """
    def callback(x):
        """
        Open the file callback.dat and write in
        the error for the current iteration of x
        """
        f = open("cw3/callback.dat", "a")
        f.write(f"{np.linalg.norm(x-x_sol)}")
        f.write("\n")
        f.close()
    return callback


### 2c
def get_A(m):
    """
    Helper function to construct A as described in the question.
    :param m: integer, dimension of desired matrix
    :return A: an mxm dimensional numpy array
    """
    A = np.diag(np.ones(m-1), -1) + np.diag(-2*np.ones(m), 0) + np.diag(np.ones(m-1), 1)
    return A
