
import numpy as np
import matplotlib.pyplot as plt
from cmath import *
import meshpy as mp



def generate_wing(a,c,d,n):
    r = abs(complex(c - a, d))
    theta = np.linspace(0,2*np.pi,n)
    z=np.zeros(n)*complex(0,1)
    x = np.zeros(n)
    y = np.zeros(n)
    for k in range(n):
        z[k] = 0.5 * ((complex(c, d) + r * exp(complex(0,theta[k]))) + (a * a) / (complex(c, d) + r * exp(complex(0,theta[k]))))
        x[k] = z[k].real
        y[k] = z[k].imag
    return(x,y)


a,b = generate_wing(5,0.5,0.4,1000)



plt.plot(a,b)

plt.show()



