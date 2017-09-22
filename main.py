import numpy as np
import matplotlib.pyplot as plt
from cmath import *
import meshpy as mp
import scipy.spatial as scsp


def generate_wing(a, c, d, n):
    r = abs(complex(c - a, d))
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.zeros(n) * complex(0, 1)
    x = np.zeros(n)
    y = np.zeros(n)
    wing = np.zeros((n, 2))
    for k in range(n):
        z[k] = 0.5 * (
        (complex(c, d) + r * exp(complex(0, theta[k]))) + (a * a) / (complex(c, d) + r * exp(complex(0, theta[k]))))
        x[k] = (z[k].real)
        y[k] = (z[k].imag)
        wing[k] = [x[k], y[k]]
    return (z, x, y, wing)


def generate_mesh(wing_complex, nbr_of_meshes):
    n = len(wing_complex)
    points = np.zeros((nbr_of_meshes * n, 2))
    for k in range(n):
        points[k] = [wing_complex[k].real,wing_complex[k].imag]
    for h in range(1,nbr_of_meshes):
        rapport = np.exp(h * h / (nbr_of_meshes * nbr_of_meshes)) - 1
        for k in range(n - 1):
            zprime = complex(points[n*(h-1) + k + 1,0],points[n*(h-1) + k + 1,1]) - complex(points[n*(h-1) + k,0],points[n*(h-1) + k,1])
            argu = phase(zprime)
            zc = exp(complex(0, np.pi / 2 + argu))
            zf = (complex(points[n*(h-1) + k + 1,0],points[n*(h-1) + k + 1,1]) + complex(points[n*(h-1) + k,0],points[n*(h-1) + k,1])) / 2
            z0 = zf + zc*rapport
            points[h * n + k] = [z0.real, z0.imag]
        zprime = complex(points[n*h,0],points[n*h,1]) - complex(points[n*(h-1) + n - 1,0],points[n*(h-1) + n-1,1])
        argu = phase(zprime)
        zc = exp(complex(0, np.pi / 2 + argu))
        zf = (complex(points[n*h,0],points[n*h,1]) + complex(points[n*(h-1) + n - 1,0],points[n*(h-1) + n-1,1])) / 2
        z0 = zf + zc * rapport
        points[(h + 1) * n - 1] = [z0.real, z0.imag]
    return (points)


a, b, c, p0 = generate_wing(5, 0.5, 0.4, 100)
p = generate_mesh(a, 10)
tri1 = scsp.Delaunay(p)
plt.triplot(p[:, 0], p[:, 1], tri1.simplices.copy())
plt.plot(p[:, 0], p[:, 1], 'o')


plt.show()
