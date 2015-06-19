# coding: utf-8
u"""
   IHT (Iterative Hard Thresholding) 
"""
import numpy as np


class IHT:
    u"""
    perform IHT 
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

    def __init__(self, A, y, k):
        # Initialize params
        self.A  = A 
        self.y  = y
        self.k  = k
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
            self._print_status()
            raise StopIteration

        # return n-step estimated signal 
        self.step += 1 
        self.x = self.iter_iht()
        return self.x
   
    
    def iter_iht(self):    

        # B1
        p = self.x + np.dot( np.conj(self.A.T),  self.y - np.dot(self.A, self.x) )
        x = hard_thres(p, self.k)
        return x 

 
    def _print_status(self):
        
        #print "residual signal: ", self.r
        print "residual norm e: ", self.e
        print "steps: %d" % self.step



def hard_thres(x, k):
    
    n = len(x) 
    desc_idxes  = np.argsort(np.abs(x))[::-1]   # sort indexes in descending order
    S           = desc_idxes[:k] 
    x_          = np.zeros(n)
    for s in S:
        x_[s] = x[s]
    return x_ 


 
if __name__ == '__main__':

 
    import matplotlib
    matplotlib.use("TkAgg")
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
    
    iht = IHT(A, y, s)
    for z in iht:
        plt.scatter(np.arange(n), x) 
        plt.stem(z)
        plt.show()
        
    plt.show()
    
    
    