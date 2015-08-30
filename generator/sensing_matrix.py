# coding: utf-8
import numpy as np
u"""
    Matrix generators
"""


 
def bernoulli(m, n):
    u"""
    return a matrix, 
    which have bernoulli distribution elements
    columns are l2 normalized
    """
    return np.random.choice( (0,1), (m, n) )


def gaussian(m, n):
    u"""
    return a matrix, 
    which have gaussian distribution elements
    columns are l2 normalized
    """
    A = np.random.randn(m, n)
    A = column_normalize(A)
    return A



def normalize(v):
    norm = np.linalg.norm(v)
    if norm==0:
        return v 
    else:
        return v/norm
    

def column_normalize(A): 
    n       = A.shape[1]
    cols    = np.hsplit(A, n)
    return  np.hstack([ normalize(col) for col in cols ])
 



if __name__ == '__main__':
    
    print "bernoulli"
    print bernoulli(2, 5)

    print "gaussian"
    print gaussian(2, 5)