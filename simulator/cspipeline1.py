# coding: utf-8
import numpy as np
from CSToolbox.generator.sparse import sparse
from CSToolbox.recovery import HTP, OMP, CoSaMP
from CSToolbox.evaluator import MSE
from numpy import meshgrid



class CSPipeline:
    
    def __init__(self):
     
        self.default_params =  {
            # [initial params]
                # recovery algorithm
                  'method':         'OMP',
                # test signal
                  'signal_length':  100,
                # number of repeats in one condition (alpha, rho)
                  'repeat_num':     5,

            # [dynamic params]
                # alpha: (m/n) 
                  'a_start':        0.01,
                  'a_end':          0.50,
                  'a_num':          5,
                # rho: signal density (k/n)
                  'r_start':        0.01,
                  'r_end':          0.20,
                  'r_num':          5,
        }
        self.load_matrix()


    def load_matrix(self, filename='../data/gaussian64x64.pkl'):
        
        self.matrix = pd.read_pickle(filename)  
 

    def init_params(self):
        
        self.set_simulation_params(self.default_params)
        self.alphas = np.linspace(self.a_start, self.a_end, self.a_num)
        self.rhos   = np.linspace(self.r_start, self.r_end, self.r_num)

           
    def set_simulation_params(self, params): 
        
        self.method         = params['method']
        self.n              = params['signal_length']
        self.repeat_num     = params['repeat_num']
        self.a_start        = params['a_start']
        self.a_end          = params['a_end']
        self.a_num          = params['a_num']
        self.r_start        = params['r_start']
        self.r_end          = params['r_end']
        self.r_num          = params['r_num']


    def run(self):
        
        self.init_params()

        recovery_rate = []
        for i, rho in enumerate(self.rhos):
            row = []
            for j, alpha in enumerate(self.alphas):
                row.append(100 * np.sum([ self.oneflow(alpha, rho) for k in range(self.repeat_num)]) / float(self.repeat_num))
                self.status( "%d [%%]" % (100.0*(i*self.r_num + j) / (self.a_num*self.r_num)) )

            recovery_rate.append(row)
        self.recovery_rate = np.array(recovery_rate)

        self.save_pickle()
        x, y = meshgrid(self.alphas, self.rhos)
        return (x, y, self.recovery_rate)
        
        
    def save_pickle(self, filename='recovery_rate.pkl'):
        pd.to_pickle(self, filename) 
    

    def status(self, string):
        print string
        

    def oneflow(self, alpha, rho):
        u"""
            n:     signal length
            alpha: m/n, compressibility
            rho:   k/n, signal density
        """
        # prepare parameters
        m = self.n * alpha
        k = self.n * rho 

        # random generation
        x0 = sparse(self.n, k)
        
        # generate A0
        # prepare static sensing matrix from pkl file.
        A0 = self.matrix[:m,:self.n] 
    
        # linear measurement (Ax = y)
        y0 = np.dot(A0, x0)
        
        # recovery 
        if self.method == "OMP":
            iterator = OMP(A0, y0) 
        elif self.method == 'HTP':
            iterator = HTP(A0, y0, k) 
        elif self.method == 'CoSaMP':
            iterator = CoSaMP(A0, y0, k)
        else:
            print "Not supported"
            return
        
        iterator.deactivate_logging()
        x1 = iterator.get_last()
        
        # evaluation [ RMSE / (norm of x0) ]
        RMSE = np.sqrt(np.abs( MSE(x0, x1) ))
        if (RMSE / np.linalg.norm(x0)) < 10**-13:
            return 1
        else:
            return 0
    
         
    
if __name__ == '__main__':
    

    import matplotlib.pyplot as plt
    import pandas as pd
     
    pl  = CSPipeline() 
    x, y, recovery_rate = pl.run()
  
    plt.xlabel('alpha (m/n)')
    plt.ylabel('rho (k/n)')
    plt.pcolor(x,y, recovery_rate, cmap='gray')
    plt.colorbar()
   
    plt.show()
     
        