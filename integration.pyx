cimport numpy as np
cimport cython
ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)

def apply_bound(b,np.ndarray[DTYPE_f, ndim=2] x,ta):
    N = x.shape[0]
    if b == 0:  # champs de pression
        for k in range(int(N / 2) - ta, int(N / 2) + ta + 1):
            x[int(N / 2) - ta-1,k] += x[int(N / 2) - ta,k]
            x[int(N / 2) + ta + 1,k] += x[int(N / 2) + ta,k]
            x[k,int(N / 2) - ta-1] += x[k,int(N / 2) - ta]
            x[k,int(N / 2) + ta] += x[k,int(N / 2) + ta-1]
        for i in range(int(N / 2) - ta, int(N / 2) + ta +1):
            for j in range(int(N / 2) - ta, int(N / 2) + ta +1):
                x[i, j] = 0
    if b == 1:  # horizontal
        for i in range(int(N / 2) - ta, int(N / 2) + ta +1):
            for j in range(int(N / 2) - ta-1, int(N / 2) + ta+2):
                x[i, j] = 0
    if b == 2:  # vertical
        for i in range(int(N / 2) - ta-1, int(N / 2) + ta+2):
            for j in range(int(N / 2) - ta, int(N / 2) + ta+1):
                x[i, j] = 0


def linear_solver(b,np.ndarray[DTYPE_f, ndim=2] x,np.ndarray[DTYPE_f, ndim=2]  x0, a, c, kmax,ta):
    N = x.shape[0]
    omega = 2 / (1 +3.14/N)
    for k in range(kmax):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                x[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) / c
        apply_bound(b, x,ta)


def diffuse_step(np.ndarray[DTYPE_f, ndim=2] u0,np.ndarray[DTYPE_f, ndim=2] v0,np.ndarray[DTYPE_f, ndim=2] u,np.ndarray[DTYPE_f, ndim=2] v,kmax,diff,dt,ta,f):
    N = u.shape[0]
    u0, u = u, u0
    v0, v = v, v0
    a = dt * diff * N * N * f
    linear_solver(1, u, u0, a, 1 + 4 * a, kmax,ta)
    linear_solver(2, v, v0, a, 1 + 4 * a, kmax,ta)




def advection(b, np.ndarray[DTYPE_f, ndim=2] d,np.ndarray[DTYPE_f, ndim=2] d0,np.ndarray[DTYPE_f, ndim=2] u,np.ndarray[DTYPE_f, ndim=2] v,f,dt,ta):
    N = d.shape[0]
    dt0 = f * dt * N
    for i in range(1,N-1):
        for j in range(1,N-1):
            # position de la particule à l'instant précédent
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            # conditions aux limites: taille de la grille
            if x < 0: x = 0
            if x >= N - 1: x = N - 2
            i0 = int(x)
            i1 = i0 + 1
            if y < 0: y = 0
            if y >= N - 1: y = N - 2
            j0 = int(y)
            j1 = j0 + 1
            # interpolation:
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    apply_bound(b, d,ta)


def advection_step(np.ndarray[DTYPE_f, ndim=2] u0,np.ndarray[DTYPE_f, ndim=2] v0,np.ndarray[DTYPE_f, ndim=2] u,np.ndarray[DTYPE_f, ndim=2] v,f,dt,ta):
    (u0, u) = (u, u0)
    (v0, v) = (v, v0)
    advection(1, u, u0 , u, v,f,dt,ta)
    advection(2, v, v0, u , v,f,dt,ta)


def projection(np.ndarray[DTYPE_f, ndim=2] u,np.ndarray[DTYPE_f, ndim=2] v,np.ndarray[DTYPE_f, ndim=2] p,np.ndarray[DTYPE_f, ndim=2] div, kmax,f,ta):
    N = len(p)
    ## calcul de la divergence
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            div[i, j] = -0.5 * f * (u[i + 1, j] - u[i - 1, j] + v[i, j + 1] - v[i, j - 1]) / N
            p[i,j] = 0
    apply_bound(0, div,ta)
    apply_bound(0, p,ta)

    ## Résolution du système
    linear_solver(0, p, div, 1, 4, kmax,ta);

    ## Mise à jour des vélocités
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            u[i, j] -= 0.5 * f * N * (p[i + 1, j] - p[i - 1, j])
            v[i, j] -= 0.5 * f * N * (p[i, j + 1] - p[i, j - 1])

    apply_bound(1, u,ta)
    apply_bound(2, v,ta)


def projection_step(np.ndarray[DTYPE_f, ndim=2] u,np.ndarray[DTYPE_f, ndim=2] v,np.ndarray[DTYPE_f, ndim=2] p,np.ndarray[DTYPE_f, ndim=2] div,kmax,f,ta):
    projection(u, v, p, div, kmax,f,ta)


def integration_step(np.ndarray[DTYPE_f, ndim=2] u0,np.ndarray[DTYPE_f, ndim=2] v0,np.ndarray[DTYPE_f, ndim=2] u,np.ndarray[DTYPE_f, ndim=2] v,np.ndarray[DTYPE_f, ndim=2] p,np.ndarray[DTYPE_f, ndim=2] div,f,kmax,dt,diff,ta):
    advection_step(u0,v0,u,v,f,dt,ta)
    diffuse_step(u0,v0,u,v,kmax,diff,dt,ta,f)
    projection_step(u,v,p,div,kmax,f,ta)

def apply_integration(iterations,np.ndarray[DTYPE_f, ndim=2] u0t,np.ndarray[DTYPE_f, ndim=2] v0t,np.ndarray[DTYPE_f, ndim=2] ut,np.ndarray[DTYPE_f, ndim=2] vt,np.ndarray[DTYPE_f, ndim=2] p,np.ndarray[DTYPE_f, ndim=2] div,f,kmax,dt,diff,ta):
    for k in range(iterations):
        integration_step(u0t,v0t,ut,vt,p,div,f,kmax,dt,diff,ta)
        print((k/iterations)*100,'%')
    return(ut,vt,u0t,v0t,p)