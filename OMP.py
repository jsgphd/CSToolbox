# coding: utf-8
u"""
   OMP (Orthogonal Matching Pursuit) 
"""
import numpy as np


u"""
    perform OMP
    [I/O]
        A: measurement matrix
        y: measured vector
        return: recovered vector
    [flow]
        A. init values
        B. iteration
            B1. find most correlated column
            B2. append index to the support set
            B3. calc estimated singal
            B4. calc residual
"""
class OMP:

    def __init__(self, A, y):
        # Initialize params
        self.A = A 
        self.y = y
        self.S = set([]) # support set
        self.r = y
        # Constants 
        self.EPS         = 10**-4   # acceptable residual
        self.ITER_MAX    = 10**3    # max of loops 
        
        
    def next(self):
        "return n-step estimated signal" 
        x       = self.iter_omp()
        self.r  = self.y - np.dot(self.A, x)
        if self.EPS > np.abs( np.linalg.norm(self.r) / np.linalg.norm(self.y) ):
            print "estimated"
        return x
    
    def iter_omp(self):    

        # B1
        p = np.dot( np.conj(self.A.T), self.r ) 
        j = np.argmax( np.abs(p) )

        # B2
        self.S.add(j)
        
        # B3
        As  = np.compress( [i in self.S for i in np.arange(13)], self.A, axis=1) 
        #As  = self.A[:,list(self.S)]  # pick up columns which have the index in S
        xs  = np.dot( np.linalg.pinv(As), self.y )  # solve least square
        x   = np.zeros(n, dtype=np.complex)
        for j, s in enumerate(sorted(self.S)):
            x[s] = xs[j]
        return x 
     

    
if __name__ == '__main__':
 
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt 
    from random_matrix import bernoulli, gaussian
    from sparse import sparse
   
    
    m = 7
    n = 13
    s = 2
     
    A       =  gaussian(m, n)
    x       = np.zeros(n)
    x[3]    = -np.sqrt(5)
    x[10]   = np.pi
    y       = np.dot(A,x)
    
     
    omp = OMP(A, y)
    for i in range(5):
        
        z = omp.next()
        align = "51" + str(i+1)
        plt.subplot(align)
        plt.scatter(np.arange(n), x) 
        #plt.bar(np.arange(n), z,0)
        plt.stem(z)

    plt.show()
        
        
        