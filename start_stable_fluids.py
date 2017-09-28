import test
import matplotlib.pyplot as plt
from integration2 import *
import numpy as np

plt.figure(1)
plt.ion()
N = 50  # largeur et hauteur de la grille de calcul
diff = 0.01  # coefficient de diffusion
visc = 10  # coefficient de viscosité
vitesse = 0.1
u = np.zeros((N + 2, N + 2)) * vitesse  # composante horizontale des vecteurs vélocité
v = np.zeros((N + 2, N + 2))  # composante verticale des vecteurs vélocité
u_prev = np.zeros((N + 2, N + 2)) * vitesse
v_prev = np.zeros((N + 2, N + 2))  # tableaux temporaires relatifs à u et v
dens = np.ones((N + 2, N + 2))
dens_prev = np.ones((N + 2, N + 2))
dt = 0.0005
epsilon = 0.01
bounds = np.zeros((N+2,N+2),dtype=bool)
for i in range(N+2):
    for j in range(N+2):
        dens[i,j] = j/(N+2)
        dens_prev[i, j] = j / (N + 2)
k = 0


while k < 100:
    plt.clf()
    plt.quiver(v,u)
    u_prev = np.ones((N+2,N+2))*vitesse
    for i in range(N + 2):
        for j in range(N + 2):
            dens[i, j] = j / (N + 2)
            dens_prev[i, j] = j / (N + 2)
    dens, u, v = simulate_step(N, u, v, u_prev, v_prev, dens, dens_prev, diff, visc, dt)
    k += 1
    print(k)
    plt.pause(0.0001)
print(dens)
plt.ioff()
plt.show()
