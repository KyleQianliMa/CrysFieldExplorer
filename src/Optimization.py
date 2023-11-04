# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 02:50:17 2023

@author: qmc
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pdb
from alive_progress import alive_bar
from time import sleep
import sympy as sym
from sympy import symbols, solve,cos,sin,Matrix, Inverse
from scipy import optimize
from scipy.special import wofz
from scipy import integrate
import scipy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from Operators import *
import CrysFieldExplorer as crs

class Optimization(crs.CrysFieldExplorer):
    def __init__(self, Magnetic_ion, Stevens_idx, alpha, beta, gamma, Parameter, temperature, field):
        super().__init__(Magnetic_ion, Stevens_idx, alpha, beta, gamma, Parameter, temperature, field)
    
    def test(self):
        print(self.J)
        
    @staticmethod
    def cma_loss_single(Parameters):

        global true_eigenvalue1, true_intensity1, bound, print_flag, true_T, true_X
        
        x_norm = Parameters
        
        sol = crs.CrysFieldExplorer(Magnetic_ion, Stevens_idx, alpha, beta, gamma, Parameter, temperature, field).Hamiltonian()
        
        Jx1=sol1[3]
        Jy1=sol1[4]
        Jz1=sol1[5]
        
        eigenvalues1 = sol1[1]
        Eigenvectors1=sol1[2]
        HMatrix1 = sol1[6]
        
        intensity1 = sol1[0][0:7].real
        
        sol25 = kp.solver(x_norm[0],x_norm[1],x_norm[2],x_norm[3],x_norm[4],x_norm[5],x_norm[6],x_norm[7],x_norm[8],x_norm[9],x_norm[10],x_norm[11],x_norm[12],x_norm[13],x_norm[14],25)
        intensity25 = sol25[0].real
        
        
        # ----------------------------------------------------------------
        
        loss1 = 0.0
        loss1 = loss1 + (np.linalg.det((true_eigenvalue1[0] + eigenvalues1[0]) * np.eye(int(2*J+1)) - HMatrix1))**2 
        loss1 = loss1 + (np.linalg.det((true_eigenvalue1[1] + eigenvalues1[0]) * np.eye(int(2*J+1)) - HMatrix1))**2 
        loss1 = loss1 + (np.linalg.det((true_eigenvalue1[2] + eigenvalues1[0]) * np.eye(int(2*J+1)) - HMatrix1))**2 
        loss1 = loss1 + (np.linalg.det((true_eigenvalue1[3] + eigenvalues1[0]) * np.eye(int(2*J+1)) - HMatrix1))**2 
        loss1 = loss1 + (np.linalg.det((true_eigenvalue1[4] + eigenvalues1[0]) * np.eye(int(2*J+1)) - HMatrix1))**2 
        loss1 = loss1 + (np.linalg.det((true_eigenvalue1[5] + eigenvalues1[0]) * np.eye(int(2*J+1)) - HMatrix1))**2 
        loss1 = loss1 + (np.linalg.det((true_eigenvalue1[6] + eigenvalues1[0]) * np.eye(int(2*J+1)) - HMatrix1))**2 
        loss1 = np.log10(np.absolute(loss1))
        
        loss2=0.0
        loss2 = np.sqrt(np.sum((true_intensity1 - intensity1)**2))/ np.sqrt(np.sum((true_intensity1)**2))
        
        loss3 = np.sqrt(np.sum((true_intensity25 - intensity25)**2))/ np.sqrt(np.sum((true_intensity25)**2))
        
        # T, X = calc_x(Eigenvectors1,Jx1, Jy1, Jz1, (sol1[1]-sol1[1][0])/0.086173303)
        # lossX = np.sqrt(np.mean(((1./X - 1./true_X))**2.0))/np.sqrt(np.mean(((1./true_X))**2.0))
        
        
        t=100.0
        k=100.0
        X=100
        loss = np.maximum(-20,loss1)+np.maximum(0.001,loss2*t) +loss3*k# + X*lossX
        
        # print(loss1, loss2*t, loss3*k,X*lossX)
        
        total_loss = loss
        
        return total_loss
#%%

#%%#############################
# 	  The main algorithm
################################
if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #--------------------------------------
    # Process the spectrum data
    
    alpha= 4/(45*35)
    beta=2/(11*15*273)
    gamma=8/(13**2*11**2*3**3*7)
    
    par_dim = 15
    bound = np.zeros((2,par_dim))
    
    bound[:,0]  = [-1, 1]
    bound[:,1]  = [-1, 1]
    bound[:,2]  = [-1, 1]
    bound[:,3]  = [-1, 1]
    bound[:,4]  = [-1, 1]
    bound[:,5]  = [-1, 1]
    bound[:,6]  = [-1, 1]
    bound[:,7]  = [-1, 1]
    bound[:,8]  = [-1, 1]
    bound[:,9]  = [-1, 1]
    bound[:,10]  = [-1, 1]
    bound[:,11]  = [-1, 1]
    bound[:,12]  = [-1, 1]
    bound[:,13]  = [-1, 1]
    bound[:,14]  = [-1, 1]
    bound = bound * 100.0
    
    
    true_eigenvalue1 = np.array([1.77, 5.25, 7.17,  13.72, 22.58, 27.81, 49.24]) #type in experiment measured values
    true_intensity1=np.array([   1.00,0.365,0.000,0.167,0.074,0.027,0.010])
    
    
    true_intensity25=np.array([1   ,       0.343, 0.00, 0.123, 0.070, 0.0218, 0.0162, 0.207, 0.0228, 0.03944])
    #                          1.75,       5.29 , 7.1 , 13.73, 22.63, 27.72 , 49.32 , 3.44 , 11.81 , 8.29
    
    ntry = 30
    
    true_X=np.array([0.059169, 0.054744, 0.050943, 0.047583, 0.044683, 0.041926, 0.039777])
    
    final_result = np.zeros((ntry, par_dim+2))
    comm.Barrier()
    
    for iter_num  in range(ntry):

        print_flag = False
        #-----------------------------------------------------------------------------
        #                    First-round optimization using PSO
        #-----------------------------------------------------------------------------    
        x_init = np.random.rand(par_dim) * 2.0 - 1.0
        x_init = x_init * (bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5
    
    
        max_bound = np.ones(par_dim)
        min_bound = - max_bound
        bnds = (min_bound, max_bound)
        bnds = (bound[0,:], bound[1,:])
        res = cma.fmin(loss_func_single, x_init, 1e-7, options={'maxfevals': 10000000, 'tolfacupx': 1e9}, args=(), gradf=None, \
    		    restarts=1, restart_from_best=True, incpopsize=1, eval_initial_x=True, \
    		     parallel_objective=None, noise_handler=None, noise_change_sigma_exponent=1, \
    		     noise_kappa_exponent=0, bipop=True, callback=None)
            
        print('======================')
        print(res[0])
        print(res[1])
    
        final_result[iter_num, 0:par_dim] = res[0]
        final_result[iter_num, par_dim+0] = res[1] # total loss
    
    comm.Barrier()
    
    if rank == 0:
        result=final_result
        for source in range(1, size):
            message=comm.recv(source=source)
            result=np.append(result,message,axis=0)
            np.savetxt('Eradam_MPI_Newfit_09252023'+'.csv', result,fmt='%2.20e', delimiter=', ')
    else:
        comm.send(final_result, dest=0)
    comm.Barrier()
    
    exit()
    
    
