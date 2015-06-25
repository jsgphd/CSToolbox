# coding: utf-8
u"""
   OMP (Orthogonal Matching Pursuit) 
   on noise experiment
"""
import numpy as np
from greedy.OMP import OMP 
from greedy.IHT import IHT

import matplotlib.pyplot as plt 
from random_matrix import bernoulli, gaussian
from sparse import sparse
         
m  = 50
n  = 100
s  = 2
A       =  gaussian(m, n)
x       = np.zeros(n)
x[n/5]    = -np.sqrt(5)
x[n/2]   = np.pi


# add noise
e = np.linalg.norm(x)

x = x + np.random.randn(n) / e**2

y       = np.dot(A,x)

OMP = OMP(A, y)
IHT = IHT(A, y, s)

for z in OMP:
    plt.scatter(np.arange(n), x, c='k') 
    plt.stem(z)
    plt.show()

