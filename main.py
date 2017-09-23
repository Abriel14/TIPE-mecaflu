import numpy as np
import matplotlib.pyplot as plt
from cmath import *
import meshpy as mp
import scipy.spatial as scsp


def function_wing(a, c, d, theta):
    r = abs(complex(c - a, d))
    return (
    0.5 * ((complex(c, d) + r * exp(complex(0, theta))) + (a * a) / (complex(c, d) + r * exp(complex(0, theta)))))


def generate_wing(a, c, d, n):
    theta = np.linspace(0, 2 * np.pi, n)
    wing_complex = np.zeros(n) * complex(0, 1)
    x = np.zeros(n)
    y = np.zeros(n)
    wing = np.zeros((n, 2))
    for k in range(n):
        wing_complex[k] = function_wing(a, c, d, theta[k])
        x[k] = (wing_complex[k].real)
        y[k] = (wing_complex[k].imag)
        wing[k] = [x[k], y[k]]
    return (wing_complex, x, y, wing)


def generate_layers(wing_complex, nbr_of_layers):
    n = len(wing_complex)
    points_complex = np.zeros((nbr_of_layers, n)) * complex(0, 1)
    ## injecter les points de la matrice wing_complex dans la première ligne de la matrice points
    for k in range(n):
        points_complex[0, k] = wing_complex[k]
    ## calculer les points à ajouter étant des translation de chaque point selon la normale entre 2 points:
    ## créer un maillage plus précis sur les bords de l'aile
    for h in range(1, nbr_of_layers):
        rapport = np.exp(h * h / (nbr_of_layers * nbr_of_layers)) - 1
        for k in range(n - 1):
            zprime = points_complex[h - 1, k + 1] - points_complex[h - 1, k]
            argu = phase(zprime)
            z_step = exp(complex(0, np.pi / 2 + argu))
            z_moy = (points_complex[h - 1, k + 1] + points_complex[h - 1, k]) / 2
            # argu = phase(z_moy)
            # z_step = exp(complex(0, np.pi / 2 + argu))
            z0 = z_moy + z_step * rapport
            points_complex[h, k] = z0
        ## traitement du cas final: placer un point translaté entre le point de coordonée [h-1,0] et [h-1,n-1]
        zprime = points_complex[h - 1, 0] - points_complex[h - 1, n - 1]
        argu = phase(zprime)
        z_step = exp(complex(0, np.pi / 2 + argu))
        z_moy = (points_complex[h - 1, 0] + points_complex[h - 1, n - 1]) / 2
        # argu = phase(z_moy)
        # z_step = exp(complex(0, np.pi / 2 + argu))
        z0 = z_moy + z_step * rapport
        points_complex[h, n - 1] = z0
    ##transformation de la matrice points_complex en points: array de taille (n*nbr_of_layers,2) listant les coordonées dans R² des points complexes
    points_complex = np.reshape(points_complex, n * nbr_of_layers)
    points = np.zeros((n * nbr_of_layers, 2))
    for k in range(n * nbr_of_layers):
        points[k] = [points_complex[k].real, points_complex[k].imag]

    return (points)


def generate_meshes(a, c, d, n, nbr_of_layers):
    wing_complex, x, y, wing = generate_wing(a, c, d, n)
    points = generate_layers(wing_complex,nbr_of_layers)
    meshes = scsp.Delaunay(points)
    return (points,meshes)



p0, tri1 = generate_meshes(5, 0.5, 0.4, 100 , 10)
plt.triplot(p0[:, 0], p0[:, 1], tri1.simplices.copy())
plt.plot(p0[:, 0], p0[:, 1], 'o')

plt.show()
