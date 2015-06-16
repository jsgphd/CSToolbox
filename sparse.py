# coding: utf-8
import numpy as np



def sparse(n, k):
    u"""
        [return]  k-sparse vector
        n:        size of vector
        k:        number of nonzero entries
    """
    z   = np.zeros(n)
    for i in np.random.choice( np.arange(n), k, replace=None ):   # supports of nonzero entries
        z[i] = np.random.randn()
    return z




def compressible(n, k, e):
    u"""
        [return]  k-sparse vector
        n:        size of vector
        k:        number of nonzero entries
        e:        noise level
    """
    z    = np.zeros(n)
    N    = np.arange(n)
    K    = np.random.choice( N, k, replace=None )
    K_c  = [ i for i in N if i not in K]

    for k in K:   # supports of nonzero entries
        z[k] = np.random.randn() + 0.2
    for i in K_c:
        z[i] = e * np.random.randn()
        
    return z





if __name__ == '__main__':

    s = 2
    print "%s-sparse vector:" % s
    print  sparse(10, s)
    print  compressible(10, s, 0.1)
