# coding: utf-8
u"""
    Thresholding funcs 
"""
import numpy as np




def indexThresholding(z, k):
    u"""
    return k-largetst indexes of vector x,
    which are sorted descending
    z: vector (real or complex)
    k: thresholding bound
    """
    desc_idxes  = np.argsort(np.abs(z))[::-1]   # sort indexes in descending order
    return desc_idxes[:k] 


def hardThresholding(X, k):

    dim = len(X.shape)

    if dim == 1:
        return _hardThres1D(X, k)

    elif dim == 2:
        return _hardThres2D(X, k)

    else:
        print "support only vector(1d) or matrix(2d)"
        return None


 
def _hardThres1D(z, k):

    x_  = np.zeros(len(z), dtype=np.complex)
    for s in indexThresholding(z, k):
        x_[s] = z[s]
    return x_ 


def _hardThres2D(X, k):
    
    # flatten
    m, n    = X.shape
    x       = X.flatten()
    # thresholding
    x_      = _hardThres1D(x, k)
    # reshape back
    X_      = x_.reshape(m,n)

    return X_




if __name__ == '__main__':
    
    m, n = 3, 5
    k = 3
    a = np.arange(m*n)
    A = a.reshape(m,n)

    print indexThresholding(a, k)
    print hardThresholding(A, k)
