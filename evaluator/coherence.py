# coding: utf-8
u"""
    Util funcs of coherence
"""
import numpy as np
import itertools


def coherence(A):
    u"""
        Calculate coherence
        A: measurement matrix
        return: coherence mu 
    """
    n       = A.shape[1]
    cols    = [ A[:,j] for j in range(n) ]
    
    N = range(n) 
    P = []
    for i, j in itertools.combinations(N, 2):
        p = np.abs( np.dot(A[:,i], A[:,j]) )
        P.append(p)
    return np.max(P)





if __name__ == '__main__':
    
    from generator.random_matrix import gaussian, bernoulli
    m = 3
    n = 5
    A = np.arange(m*n).reshape(m,n)
    B = gaussian(m,n)
    C = bernoulli(m,n) 

    print C 

    print "coherence of A:",            coherence(A)
    print "coherence of Gaussian:",     coherence(B)
    print "coherence of Bernoulli:",    coherence(C)