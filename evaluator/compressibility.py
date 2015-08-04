# coding: utf-8
u"""
    Util funcs for compressiblity
"""
import numpy as np


def compressibility(k, x, p):
    
    K     = np.argsort(x)[:k]
    n     = x.shape[0]
    for k in K:
        x[k] = 0.0
    return np.linalg.norm(x,p)



if __name__ == '__main__':
    
    from CSToolbox.generator.sparse import compressible

    N, k  = 10, 2
    n     = np.arange(N)
    x     = compressible(N,k, 0.05)
    print "[compressibility]:  %f" % compressibility(k, x, p=2)