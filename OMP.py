# coding: utf-8
u"""
   OMP (Orthogonal Matching Pursuit) 
"""
import numpy as np
from random_matrix import bernoulli, gaussian
from vecgen import sparse



def OMP(A, y):
    u"""
        perform OMP
        A: measurement matrix
        y: measurement signal
        return: sparse vector
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
    
    m = 50 
    n = 200
    s = 4

    #A   = bernoulli(m, n)
    A   = gaussian(m, n)
    x   = sparse(n, s)
    nr  = 0.03  # noise ratio
    x   = x + nr * np.random.rand(n) # add noise 
    y   = np.dot(A, x)
   
    #print "meas matrix A[%d x %d]:" % (m, n), "\n", A, "\n"
    #print "sparse signal x:\n", x, "\n"
    #print "meas signal y:\n", y, "\n"
    z   = OMP(A,y)
    
    if z == "Failed":
        print z
    else:
        z = z.real
        #print "recovered signal z:\n", z
        ax1 = plt.subplot(211) 
        ax1.set_xlim(0,n)
        ax1.set_ylim(-2,2)
        ax1.axhline(y=0, c='k')
        ax1.bar(range(n),x, 0)
        ax1.scatter(range(n), x)
        ax2 = plt.subplot(212) 
        ax2.axhline(y=0, c='k')
        ax2.set_xlim(0,n)
        ax2.set_ylim(-2,2)
        ax2.bar(range(n),z, 0)
        ax2.scatter(range(n), z)
    
    plt.show()
    
    