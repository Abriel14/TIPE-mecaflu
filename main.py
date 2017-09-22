
import numpy as np
import matplotlib.pyplot as plt
from cmath import *
import meshpy as mp
import scipy.spatial as scsp




def generate_wing(a,c,d,n):
    r = abs(complex(c - a, d))
    theta = np.linspace(0,2*np.pi,n)
    z=np.zeros(n)*complex(0,1)
    x = np.zeros(n)
    y = np.zeros(n)
    wing = np.zeros((n,2))
    for k in range(n):
        z[k] = 0.5 * ((complex(c, d) + r * exp(complex(0, theta[k]))) + (a * a) / (complex(c, d) + r * exp(complex(0, theta[k]))))
        x[k] = (z[k].real)
        y[k] = (z[k].imag)
        wing[k] = [x[k], y[k]]
    return(z,x,y,wing)



def generate_mesh(wing_complex,nbr_of_meshes):
    n = len(wing_complex)
    points = np.zeros((nbr_of_meshes*n,2))
    for h in range(nbr_of_meshes):
        rapport = np.exp(h*h/(nbr_of_meshes*nbr_of_meshes)) - 1
        for k in range(n-1):
            zprime = wing_complex[k+1] - wing_complex[k]
            argu = phase(zprime)
            zc = exp(complex(0,np.pi/2 + argu))
            zf = (wing_complex[k+1] + wing_complex[k]) / 2
            z0 = zf + zc*rapport
            points[h*n+k,0] =  z0.real
            points[h * n + k, 1] = z0.imag
        zprime = wing_complex[0] - wing_complex[n-1]
        argu = phase(zprime)
        zc = exp(complex(0, np.pi / 2 + argu))
        zf = (wing_complex[0] + wing_complex[n-1]) / 2
        z0 = zf + zc * rapport
        points[(h + 1) * n - 1,0] = z0.real
        points[(h + 1) * n - 1, 1] = z0.imag
    return(points)

a, b, c, p0 = generate_wing(5, 0.5, 0.4, 100)
p = generate_mesh(a,20)
tri = scsp.Delaunay(p)


plt.triplot(p[:,0], p[:,1], tri.simplices.copy())
plt.plot(p[:,0], p[:,1], 'o')

plt.show()



