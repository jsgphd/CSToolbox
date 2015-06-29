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
        self.name = "OMP"

    def __iter__(self):

        return self
  
    
    def iterate(self):    

        # B1
        p = np.dot( np.conj(self.A.T), self.r ) 
        j = np.argmax( np.abs(p) )

        # B2
        self.S.add(j)
        
        # B3
        As  = self.A[:, sorted(self.S)]             # pick up columns which have the index in S
        xs  = np.dot( np.linalg.pinv(As), self.y )  # solve least square
        self.x   = np.zeros(self.A.shape[1], dtype=np.complex)
        for j, s in enumerate(sorted(self.S)):
            self.x[s] = xs[j]
        return self.x 
    
    

     
if __name__ == '__main__':

 
    import matplotlib.pyplot as plt 
    from random_matrix import bernoulli, gaussian
    from sparse import sparse
   
    m  = 10
    n  = 20
    s  = 2
     
    A       =  gaussian(m, n)
    x       = np.zeros(n)
    x[3]    = -np.sqrt(5)
    x[10]   = np.pi
    y       = np.dot(A,x)
   
    
    for z in OMP(A, y):
        plt.scatter(np.arange(n), x) 
        plt.stem(z)
        plt.show()
        
    plt.show()
        
        
        