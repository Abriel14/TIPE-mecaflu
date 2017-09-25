cimport numpy as np
cimport cython
ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)

def apply_bound(b,np.ndarray[DTYPE_f, ndim=2] x):
    N = x.shape[0]
    if b == 0:  # champs de pression
        for i in range(int(N / 2) - 5, int(N / 2) + 5):
            for j in range(int(N / 2 - 5), int(N / 2) + 5):
                x[i, j] = 0
    if b == 1:  # horizontal
        for i in range(int(N / 2) - 5, int(N / 2 + 5)):
            for j in range(int(N / 2) - 6, int(N / 2) + 6):
                x[i, j] = 0
    if b == 2:  # vertical
        for i in range(int(N / 2) - 6, int(N / 2) + 6):
            for j in range(int(N / 2) - 5, int(N / 2) + 5):
                x[i, j] = 0


def linear_solver(b,np.ndarray[DTYPE_f, ndim=2] x,np.ndarray[DTYPE_f, ndim=2]  x0, a, c, kmax):
    N = x.shape[0]
    for k in range(kmax):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                x[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) / c
        apply_bound(b, x)
