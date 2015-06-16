# coding: utf-8
import numpy as np
from itertools import combinations



def spark(A):

    n = A.shape[1]
    for k in range(n):
        k = k + 1    # [0..n-1] -> [1..n]
        for column_set in combinations(range(n), k):
            rank = np.linalg.matrix_rank( A[:,column_set] )
            if rank < k:
                return k


if __name__ == '__main__':
    
    A1 = np.array([ [1, 0, 1, 1],
                   [0, 1, 1,-1]  ])
    print "spark(A1) = %d" % spark(A1) 
    
    
    