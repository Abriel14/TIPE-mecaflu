import test
import matplotlib.pyplot as plt
N = 100  # largeur et hauteur de la grille de calcul
kmax = 100  # nombre d'itérations à effectuer (solveur)
diff = 0.01  # coefficient de diffusion
ut = np.zeros((N, N))  # composante horizontale des vecteurs vélocité
vt = np.zeros((N, N))  # composante verticale des vecteurs vélocité
u0t = np.zeros((N, N))
v0t = np.zeros((N, N))  # tableaux temporaires relatifs à u et v
p = np.zeros((N, N))
div = np.zeros((N, N))
dt = 1
f = 1
vitesse = 2

u , v, p = test.apply_integartion(iterations, u0t, v0t, ut, vt , N, kmax, diff, p , div, dt , f, vitesse)

plt.quiver(u,v)