# coding: utf-8
import numpy as np



def sparse(n, k):
    u"""
    perform IHT 
    [args]
        n: size of vector
        k: number of nonzero entries
    [return]
        k-sparse vector
    """
    z = np.zeros(n)
    for i in np.random.choice( np.arange(n), k, replace=None ):   # supports of nonzero entries
        z[i] = np.random.randn()
    return z




def compressible(n, k, e=0.1):
    u"""
    perform IHT 
    [args]
        n: size of vector
        k: number of nonzero entries
        e: noise factor (x e)
    [return]
        k-compressible vector
    """
    z = sparse(n, k) + e * np.random.randn(n)
    return z





if __name__ == '__main__':


    s = 2
    print "%s-sparse vector:" % s
    print  sparse(10, s)
    print  compressible(10, s, 0.1)


