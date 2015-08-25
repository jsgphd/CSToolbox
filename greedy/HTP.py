# coding: utf-8
u"""
   HTP (Hard Thresholding Pursuit) 
"""
import numpy as np
from greedy_baseclass import Greedy
from thresholding import indexThresholding



class HTP(Greedy):
    u"""
    perform HTP 
    [args]
        A: measurement matrix (2d ndarray)
        y: measured vector    (1d ndarray)
        k: sparsity
    [return]
        recovered vector      (1d ndarray)
    """

    def __init__(self, A, y, k):

        Greedy.__init__(self, A, y)
        self.name   = "HTP"
        self.S      = set([]) # support set (indexes)
        self.k      = k

             
    def __iter__(self):

        return self

    
    def iterate(self):    
       
        # update support sets 
        p        = self.z + np.dot( np.conj(self.A.T), self.r ) 
        self.S   = set( indexThresholding(p, self.k) )
         
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
    from CSToolbox.generator.random_matrix import bernoulli, gaussian
    from CSToolbox.generator.sparse import sparse
   
    n  = 20
    m  = 10
    s  = 2
     
    A       =  gaussian(m, n)
    x       =  np.zeros(n)
    x[3]    = -np.sqrt(5)
    x[10]   =  np.pi
    y       =  np.dot(A,x)
    
    iter = HTP(A, y, s) 
    for z in iter:
        
        #plt.clf()
        plt.scatter(np.arange(n), x, s=60, marker='x', c='r') 
        plt.stem(z.real, c='k')
        plt.show()
        print iter.get_status()
        
    plt.show()
    
 