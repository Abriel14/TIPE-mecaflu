
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
    return(x,y,wing)



def generate_mesh(wing):
    points = np.zeros((10*n,2))
    for h in range(0,10):
        rapport = np.exp(h/10) - 1
        for k in range(n):
            argu = phase(z[k])
            zc = exp(complex(0,argu))
            z0 = complex(wing[k,0],wing[k,1]) + zc*rapport
            points[h*n+k] =  [z0.real,z0.imag]
    return(points)

a, b, p = generate_wing(5, 0.5, 0.4, 50)
tri = scsp.Delaunay(p)


plt.triplot(p[:,0], p[:,1], tri.simplices.copy())
plt.plot(p[:,0], p[:,1], 'o')

plt.show()



