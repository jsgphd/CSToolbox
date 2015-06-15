# coding: utf-8
import numpy as np


u"""
    Vector generators
"""


def sparse(n, s): 
    u"""
        return sparse vector
        which have normal distributed sparse entries
    """
    
    N   = np.arange(n)                          # supports of all entries   
    S   = np.random.choice( np.arange(n), s )   # supports of nonzero entries
    _S  = [ i for i in N if i not in S ]        # complementary set of S
    z   = np.zeros(n)
    
    for s in S:
        z[s] = np.random.randn()
    return z



if __name__ == '__main__':

    s = 2
    print "%s-sparse vector:" % s
    print  sparse(10, s)
