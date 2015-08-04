# coding: utf-8
u"""
   OMP (Orthogonal Matching Pursuit) 
   on noise experiment
"""
import numpy as np
import matplotlib.pyplot as plt 

from generator.random_matrix import bernoulli, gaussian
from generator.sparse import sparse

from greedy.OMP import OMP 
from greedy.IHT import IHT
from greedy.CoSaMP import CoSaMP

import sys



# make raw signal & measurement matrix 
m  = 50
n  = 200
s  = 2
A       =  gaussian(m, n)
x       = np.zeros(n)
x[n/5]    = -np.sqrt(5)
x[n/2]   = np.pi

# add noise,
# the size of which is square of norm of raw signal  
e = 1.0 / np.linalg.norm(x)**2
x = x + np.random.randn(n) * e 

# linear measurement
y       = np.dot(A,x)


if len(sys.argv) == 1:
    iterator = OMP(A, y)
    iterator = IHT(A, y, s)

else:
    if sys.argv[1] == 'OMP':
        iterator = OMP(A, y)
    
    elif sys.argv[1] == 'IHT':
        iterator = IHT(A, y, s)
    
    elif sys.argv[1] == 'CoSaMP':
        iterator = CoSaMP(A, y, s)
     
    
    iterator.set_epsilon( np.sqrt(e) *0.5 )


for z in iterator:
    plt.scatter(np.arange(n), x, c='k') 
    plt.stem(z)
    plt.show()

