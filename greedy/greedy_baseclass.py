# coding: utf-8
u"""
    Base class of Greedy algorithms
"""
import numpy as np




class Greedy:
    u"""
    base class of Greedy algorithms
    [args]
        A: measurement matrix (2d ndarray)
        y: measured vector    (1d ndarray)
    [return]
        recovered vector      (1d ndarray)
    """
        
    def __init__(self, A, y):

        # Constants about convergence
        self.EPS         = 10**-5        # acceptable residual
        self.ITER_MAX    = 1*10**1  # max of iterations 
        
        self.name        = "Unknown"

        # Initialization 
        self.A  = A 
        self.y  = y
        self.S  = set([]) # support (indexes)
        self.r  = y
        self.e  = float("inf") 
        self.x  = np.zeros(A.shape[1], dtype=np.complex)
        # iterator var
        self.iterations = 0 

    def __iter__(self):
        return self
   
     
    def next(self):
        
        # check number of loops
        if self.iterations == self.ITER_MAX:
            print "Reach to MAX Iterations"
            print self.get_result()
            raise StopIteration
        
        # check convergence by previous iteration
        if self.e < self.EPS:
            print "Converged"
            print self.get_result()
            raise StopIteration

        # return signal estimated by n-iterations 
        self.iterations += 1 
        self.x  = self.iterate()
        self.r  = self.y - np.dot(self.A, self.x)
        self.e  = np.abs( np.linalg.norm(self.r) / np.linalg.norm(self.y) )
        print self.get_status()
        return self.x
   
 
    def set_epsilon(self, e):
        
        self.EPS = e
        
  
    def get_status(self):
        
        status =  "" 
        status += "iterations:        %d\n"    % self.iterations
        status += "residual norm (e): %.2e\n"  % self.e
        return status
    
    def get_result(self):
        
        result  = "------- summary ----\n"
        result += "[ %s ]\n"                    % self.name
        result += "number of iterations: %d\n"  % self.iterations 
        result += "specified error:   %.2e\n"   % self.EPS
        result += "residual norm (e): %.2e\n"   % self.e
        return result

