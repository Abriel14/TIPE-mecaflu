import numpy as np
import matplotlib.pyplot as plt
from cmath import *

N = 100  # largeur et hauteur de la grille de calcul
kmax = 100  # nombre d'itérations à effectuer (solveur)
diff = 0.01  # coefficient de diffusion
u = np.zeros((N, N))  # composante horizontale des vecteurs vélocité
v = np.zeros((N, N))  # composante verticale des vecteurs vélocité
u0 = np.zeros((N, N))
v0 = np.zeros((N, N))  # tableaux temporaires relatifs à u et v
dt = 0.01


def apply_bound(b, x):
    N = len(x)
    if b == 1:  # horizontal
        for i in range(N / 2 - 5, N / 2 + 5):
            for j in range(N / 2 - 6, N / 2 + 6):
                x[i, j] = 0
    if b == 2:  # vertical
        for i in range(N / 2 - 6, N / 2 + 6):
            for j in range(N / 2 - 5, N / 2 + 5):
                x[i, j] = 0


def linear_solver(b, x, x0, a, c, kmax):
    for k in range(kmax):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                x[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) / c
        apply_bound(b, x)


def diffuse_step():
    u0, u = u, u0
    v0, v = v, v0
    a = dt * diff * N * N
    linear_solver(1, u, u0, a, 1 + 4 * a, kmax)
    linear_solver(2, v, v0, a, 1 + 4 * a, kmax)


def advection(b, d, d0):
    dt0 = dt * N
    for i in range(N):
        for j in range(N):
            # position de la particule à l'instant précédent
            x = i - dt0 * x[i, j]
            y = j - dt0 * x[i, j]
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
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]);
    apply_bound(b, d)


def advection_step():
    (u0, u) = (u, u0)
    (v0, v) = (v, v0)
    advection(1, u, u0)
    advection(2, v, v0)

