# coding: utf-8
u"""
   OMP (Orthogonal Matching Pursuit) 
"""
import numpy as np
from random_matrix import bernoulli, gaussian
from sparse import sparse


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
        self.A = A 
        self.y = y
        self.S = set([]) # support set
        self.r = y
        
    def next(self):
        "return n-step estimated signal" 
        return self.iter_omp()
    
    def iter_omp(self):    

        # B1
        p = np.dot( np.conj(self.A.T), self.r ) 
        j = np.argmax( np.abs(p) )

        # B2
        self.S.add(j)
        #print "self.S", self.S
        
        # B3
        As  = self.A[:,list(self.S)]  # pick up columns which have the index in S
        xs  = np.dot( np.linalg.pinv(As), self.y )  # solve least square
        x   = np.zeros(n, dtype=np.complex)
        for s, j in enumerate(sorted(self.S)):
            x[j] = xs[s]
            
        self.r = self.y - np.dot(self.A, x)
        return x 
     

    
def omp(A, y):
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

    m, n    = A.shape
    M       = np.arange(m)
    N       = np.arange(n)
    norm_y  = np.linalg.norm(y) 

    def iter_omp(x):

        # OMP1 ~ find a most correlated column
        p = np.dot( np.conj(A.T), y-np.dot(A, x) ) 
        j = np.argmax( np.abs(p) )
        if not j in S:
            S.append(j) 
            #print "S:", S

        # OMP2 
        As  = np.compress( [i in S for i in N], A, axis=1)  # pick up columns which have the index in S
        xs  = np.dot( np.linalg.pinv(As), y )      # solve least square
        x   = np.zeros(n, dtype=np.complex)
        for s, j in enumerate(sorted(S)):
            x[j] = xs[s]
        return x 
     
    # 初期状態
    x   = np.zeros(n, dtype=np.complex)
    S   = [] # 順序付きセット
 
    EPS         = 10**-4   # acceptable residual
    ITER_MAX    = 10**3    # max of loops 
 
    for loop in range(ITER_MAX):
        x = iter_omp(x)
        r = np.linalg.norm(y - np.dot(A, x)) / norm_y
        if np.abs(r) < EPS:
            print loop, " [loops]"
            return x
    return "Failed"
  

    
if __name__ == '__main__':
 
    import matplotlib.pyplot as plt 
    

    m = 7
    n = 13
    s = 2
     
    A =  bernoulli(m, n)
    A =  gaussian(m, n)
    x = sparse(n, s)
    y = np.dot(A,x)
    
    print "x ", x
    #print "A ", A
     
    omp = OMP(A, y)
    
    for i in range(5):
        
        z = omp.next()
        plt.scatter(np.arange(n), x) 
        plt.bar(np.arange(n), z, 0)
        plt.show()