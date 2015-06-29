# coding: utf-8
u"""
   CoSaMP (Compressive sampling matching pursuit) 
"""
import numpy as np
from greedy_baseclass import Greedy
from greedy.thresholding import indexThresholding, hardThresholding



class CoSaMP(Greedy):
    u"""
    perform CoSaMP
    [args]
        A: measurement matrix (2d ndarray)
        y: measured vector (1d ndarray)
        k: sparsity
    [return]
        recovered vector (1d ndarray)
    """

    def __init__(self, A, y, k):

        Greedy.__init__(self, A, y)
        self.name = "CoSaMP"
        self.k = k


    def __iter__(self):

        return self
  
    
    def iterate(self):    

        # update support sets 
        self.r   = self.y - np.dot(self.A, self.x)
        z        = np.dot( np.conj(self.A.T), self.r ) 
        s        = indexThresholding(z, 2*self.k)
        self.S  |= set(s)
        
        # 
        As  = self.A[:, sorted(self.S)]  # pick up columns which have the index in S
        us  = np.dot( np.linalg.pinv(As), self.y )  # solve least square
        u   = np.zeros(self.A.shape[1], dtype=np.complex)
        for j, s in enumerate(sorted(self.S)):
            u[s] = us[j]
        self.x   = hardThresholding(u, self.k)
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
    
    for z in CoSaMP(A, y, s):
        plt.scatter(np.arange(n), x) 
        plt.stem(z)
        plt.show()
        
    plt.show()
        
        
 