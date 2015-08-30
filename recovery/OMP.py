# coding: utf-8
u"""
   OMP (Orthogonal Matching Pursuit) 
"""
import numpy as np
from greedy_baseclass import Greedy


class OMP(Greedy):
    u"""
    perform OMP
    [args]
        A: measurement matrix (2d ndarray)
        y: measured vector (1d ndarray)
    [return]
        recovered vector (1d ndarray)
    """

    def __init__(self, A, y):
        
        Greedy.__init__(self, A, y)
        self.name   = "OMP"
        self.S      = set([]) # support set (indexes)

    def __iter__(self):

        return self
  
    
    def iterate(self):    

        # project residual vector on measurement matrix,
        # and find the index of the largest entry
        g = np.dot( np.conj(self.A.T), self.r ) 
        j = np.argmax([ np.abs(g[i]) / np.linalg.norm(self.A[:,i]) for i in range(self.n) ])

        # add the index to the supports set S
        self.S.add(j)
        
        # make a matrix which of columns have the index in S
        As  = self.A[:, sorted(self.S)]

        # to minimum solution of || As z - y ||2 = 0,
        # solve least square
        zs  = np.dot( np.linalg.pinv(As), self.y )
        
        # make approximated signal z,
        # the entries of which are the solutions of
        # the previous least square
        z  = np.zeros(self.A.shape[1], dtype=np.complex)
        for j, s in enumerate(sorted(self.S)):
            z[s] = zs[j]

        return z 
    
    

     
if __name__ == '__main__':

 
    import matplotlib.pyplot as plt 
    from generator.sensing_matrix import bernoulli, gaussian
    from CSToolbox.generator import sparse
   
    m  = 10
    n  = 20
    s  = 2
     
    A       =  gaussian(m, n)
    x       = np.zeros(n)
    x[3]    = -np.sqrt(5)
    x[10]   = np.pi
    y       = np.dot(A,x)
   
    iter = OMP(A, y) 
    for z in iter:
        plt.clf()
        plt.scatter(np.arange(n), x, s=60, marker='x', c='r') 
        plt.stem(z.real)
        #plt.show()
        #print iter.get_status()
        
    plt.show()
        
        
        