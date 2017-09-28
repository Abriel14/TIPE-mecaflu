import test
import matplotlib.pyplot as plt
import integration as inte
import numpy as np

plt.figure(1)
plt.ion()
N = 30  # largeur et hauteur de la grille de calcul
kmax = 200  # nombre d'itérations à effectuer (solveur)
diff = 0.001  # coefficient de diffusion
vitesse = 1
ut = np.zeros((N, N)) * vitesse # composante horizontale des vecteurs vélocité
vt = np.zeros((N, N))  # composante verticale des vecteurs vélocité
u0t = np.zeros((N, N)) * vitesse
v0t = np.zeros((N, N))  # tableaux temporaires relatifs à u et v
p = np.zeros((N, N))
div = np.zeros((N, N))
dt = 1
f = 0.2
epsilon = 0.01
rang = 0
# for i in range(N):
#     for j in range(N):
#         ut[i,j] = vitesse*(abs(15-j)/15)
#         u0t[i,j] = vitesse*(abs(15-j)/15)
u, v, u0, v0, p = inte.apply_integration(1, u0t, v0t, ut, vt, p, div, f, kmax, dt, diff, 5)
while np.linalg.norm(abs(u - u0)) > epsilon:
    u, v, u0, v0, p = inte.apply_integration(1, u0, v0, u, v, p, div, f, kmax, dt, diff, 5)
    rang += 1
    print(rang, np.linalg.norm((abs(u - u0) + abs(v - v0)), 2))
    plt.clf()
    plt.quiver(u, v)
    plt.pause(0.0001)
print(p)
plt.ioff()
plt.show()
