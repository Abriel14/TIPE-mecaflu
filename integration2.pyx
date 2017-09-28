cimport numpy as np
cimport cython


ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)

def set_bnd(N, b, x):
#    if b == 1:  # horizontal
#        for i in range(int(N / 2 ) - 5, int(N / 2) + 5 + 1):
#            for j in range(int(N / 2) - 5 - 1, int(N / 2) + 5 + 2):
#                x[i, j] = 0
#    if b == 2:  # vertical
#        for i in range(int(N / 2) - 5 - 1, int(N / 2) + 5 + 2):
#            for j in range(int(N / 2) - 5, int(N / 2) + 5 + 1):
#                x[i, j] = 0
#        for k in range(N+2):
#            x[0,k] = 0.1
    for k in range(N+2):
        if b == 1:
            x[0,k] = 0.1
            x[N+1,k] = -x[N,k]
        else:
            x[0,k] = x[1,k]
            x[N+1,k] = x[N,k]
        if b == 2:
            x[k,0] = -x[k,1]
            x[k,N+1] = -x[k,N]
        else:
            x[k,0] = x[k,1]
            x[k,N+1] = x[k,N]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N + 1] = 0.5 * (x[1, N + 1] + x[0, N])
    x[N + 1, 0] = 0.5 * (x[N, 0] + x[N + 1, 1])
    x[N + 1, N + 1] = 0.5 * (x[N, N + 1] + x[N + 1, N])


def add_source(N, x, s, dt):
    for i in range(N + 2):
        for j in range(N + 2):
            x[i,j] = dt * s[i,j]


def diffuse(N, b, x, x0, diff, dt):
    a = dt * diff * N * N
    for k in range(50):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                x[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) / (1 + 4 * a)
        set_bnd(N, b, x)


def advect(N, b, d, d0, u, v, dt):
    dt0 = dt * N
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            # position de la particule à l'instant précédent
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            # conditions aux limites: taille de la grille
            if x < 0.5: x = 0.5
            if x > N + 0.5: x = N + 0.5
            i0 = int(x)
            i1 = i0 + 1
            if y < 0.5: y = 0.5
            if y > N + 0.5: y = N + 0.5
            j0 = int(y)
            j1 = j0 + 1
            # interpolation:
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    set_bnd(N, b, d)


def project(N, u, v, p, div):
    h = 1 / N
    ## calcul de la divergence
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            div[i, j] = -0.5 * h * (u[i + 1, j] - u[i - 1, j] + v[i, j + 1] - v[i, j - 1])
            p[i, j] = 0
    set_bnd(N, 0, div)
    set_bnd(N, 0, p)

    ## Résolution du système
    for k in range(50):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                p[i, j] = (div[i, j] + p[i - 1, j] + p[i + 1, j] + p[i, j - 1] + p[i, j + 1]) / 4
        set_bnd(N, 0, p)

    ## Mise à jour des vélocités
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            u[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j]) / h
            v[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1]) / h
    set_bnd(N, 1, u)
    set_bnd(N, 2, v)


def dens_step(N, x, x0, u, v, diff, dt):
    add_source(N, x, x0, dt)
    x, x0 = x0, x
    diffuse(N, 0, x, x0, diff, dt)
    x, x0 = x0, x
    advect(N, 0, x, x0, u, v, dt)
    print(x)


def vel_step(N, u, v, u0, v0, visc, dt):
    add_source(N, u, u0, dt)
    add_source(N, v, v0, dt)
    u0, u = u, u0
    diffuse(N, 1, u, u0, visc, dt)
    v0, v = v, v0
    diffuse(N, 2, v, v0, visc, dt)
    project(N, u, v, u0, v0)
    u0, u = u, u0
    v0, v = v, v0
    advect(N, 1, u, u0, u0, v0, dt)
    advect(N, 2, v, v0, u0, v0, dt)
    project(N, u, v, u0, v0)

def simulate_step(N, u, v, u_prev, v_prev, dens, dens_prev, diff, visc, dt):
    vel_step(N, u, v, u_prev, v_prev, visc, dt)
    dens_step(N, dens, dens_prev, u, v, diff, dt)
    return(dens_prev,u,v)