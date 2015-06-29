# coding: utf-8
u"""
   OMP (Orthogonal Matching Pursuit) 
   on noise experiment
"""
import numpy as np
from greedy.OMP import OMP 
from greedy.IHT import IHT
from greedy.CoSaMP import CoSaMP

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
e = 1.0 / np.linalg.norm(x)**2
x = x + np.random.randn(n) * e 

y       = np.dot(A,x)

iterator = OMP(A, y)
iterator = IHT(A, y, s)
iterator = CoSaMP(A, y, s)


iterator.set_epsilon( np.sqrt(e) )



for z in iterator:
    plt.scatter(np.arange(n), x, c='k') 
    plt.stem(z)
    plt.show()

