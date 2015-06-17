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
        self.A  = A 
        self.y  = y
        self.S  = set([]) # support set
        self.r  = y
        self.e  = 0.0
        self.x  = np.zeros(n, dtype=np.complex)
        # iterator var
        self.step = 0 
        # Constants 
        self.EPS         = 10**-5   # acceptable residual
        self.ITER_MAX    = 10**1    # max of loops 
    
     
    def __iter__(self):
        return self
   
     
    def next(self):
        
        # check number of loops
        if self.step == self.ITER_MAX:
            print "Reach to MAX Iterations"
            raise StopIteration
        
        # check condition of convergence 
        self.r  = self.y - np.dot(self.A, self.x)
        self.e = np.abs( np.linalg.norm(self.r) / np.linalg.norm(self.y) )
        self.e = np.linalg.norm(self.r)        
        if self.e < self.EPS:
            print "Converged"
            self._printdebug()
            raise StopIteration

        # return n-step estimated signal 
        self.step += 1 
        self.x = self.iter_omp()
        return self.x
   
    
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
     
    def _printdebug(self):
        
        #print "residual signal: ", self.r
        print "residual norm e: ", self.e
        print "steps: %d" % self.step
   
    
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
    for z in omp:
        plt.scatter(np.arange(n), x) 
        plt.stem(z)
    
    plt.show()
        
        
        