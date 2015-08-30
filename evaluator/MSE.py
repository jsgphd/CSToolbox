# coding: utf-8
u"""
    MSE calculator for vector/matrix
"""
import numpy as np



def MSE(actual, standard):
    
    dim = len(standard.shape)

    if dim == 1:
        return MSE1D(actual, standard)

    elif dim == 2:
        return MSE2D(actual, standard)

    else:
        print "support only vector(1d) or matrix(2d)"
        return None


def MSE1D(act_x, std_x):
    u"""
    act_x: actual
    std_x: standard 
    """
    N       = len(std_x)
    sum     = np.sum([ (act_x[i] - std_x[i])**2 for i in range(N) ])
    return sum/N


def MSE2D(act_X, std_X):
    u"""
    act_X: actual
    std_X: standard
    """
    act_x = act_X.flatten()
    std_x = std_X.flatten()
    return MSE1D(act_x, std_x)
    
    
if __name__ == '__main__':
    
    z       = np.zeros(10)
    x       = z.copy()
    x[2]    = 2.0
    print "MSE should be 0.4 ->", MSE1D(x, z)
   
    Z       = np.zeros([2,5])
    X       = Z.copy()
    X[1,1]  = 2.0
    print "MSE should be 0.4 ->", MSE2D(X, Z)
    
    print "----"
    print "MSE should be 0.4 ->", MSE(x, z)
    print "MSE should be 0.4 ->", MSE(X, Z)
 