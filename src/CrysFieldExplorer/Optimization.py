# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 02:50:17 2023

@author: qmc
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from alive_progress import alive_bar
from time import sleep
import sympy as sym
from sympy import symbols, solve,cos,sin,Matrix, Inverse
from scipy import optimize
from scipy.special import wofz
from scipy import integrate
import scipy.linalg as LA
from mpi4py import MPI
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
import CrysFieldExplorer.Operators as op
import CrysFieldExplorer.CrysFieldExplorer as crs
import cma

class Optimization(crs.CrysFieldExplorer):
    def __init__(self, Magnetic_ion, Stevens_idx, alpha, beta, gamma, Parameter, temperature, field, true_eigenvalue, true_intensity):
        self.Stevens_idx=Stevens_idx
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.Parameter=Parameter
        self.T=temperature
        self.field=field
        self.true_eigenvalue=true_eigenvalue
        self.true_intensity=true_intensity
        # self.true_X=true_X
        super().__init__(Magnetic_ion, Stevens_idx, alpha, beta, gamma, Parameter, temperature, field)

        
    def test(self):
        print(self.Parameter)
         
    
    def cma_loss_single(self):
        # global true_eigenvalue1, true_intensity1, bound, print_flag, true_T, true_X
        
        #calculate Hamiltonian and ev, ef, H
        sol = super().Hamiltonian()
        
        Jx   =super().Jx()
        Jy   =super().Jy()
        Jz   =super().Jz
        dim  =len(Jx)
        J    =self.J
        
        eigenvalues  = sol[0]
        Eigenvectors = sol[1]
        HMatrix      = sol[2]
        
        #take in experimentally observed energy levels
        true_eigenvalue=self.true_eigenvalue
        true_intensity=self.true_intensity
        
        calint=super().Neutron_Intensity(2,0,True)
        Intensity=np.zeros(len(calint))
        j=0
        for i in calint:
            Intensity[j]=calint[i].real
            j+=1
        Intensity=Intensity[1:len(calint)] # excluding E=0 ground state intensity
        # ----------------------------------------------------------------
        loss1=0
        for i in range(len(true_eigenvalue)):
            loss1+=(np.linalg.det((true_eigenvalue[i] + eigenvalues[0]) * np.eye(int(2*J+1)) - HMatrix))**2  
        loss1 = np.log10(np.absolute(loss1)) 
        
        # loss2=0.0
        # loss2 = np.sqrt(np.sum((true_intensity - Intensity)**2))/ np.sqrt(np.sum((true_intensity)**2))
        
        # print((np.linalg.det((true_eigenvalue[0] + eigenvalues[0]) * np.eye(int(2*J+1)) - HMatrix))**2  )

        # loss3 = np.sqrt(np.sum((true_intensity25 - intensity25)**2))/ np.sqrt(np.sum((true_intensity25)**2))
        
        # T, X = calc_x(Eigenvectors1,Jx1, Jy1, Jz1, (sol1[1]-sol1[1][0])/0.086173303)
        # lossX = np.sqrt(np.mean(((1./X - 1./true_X))**2.0))/np.sqrt(np.mean(((1./true_X))**2.0))
        
        # t=100.0
        # k=100.0
        # X=100
        # loss = np.maximum(-20,loss1)+np.maximum(0.001,loss2*t) +loss3*k# + X*lossX
        loss=loss1
        # print(loss1)
        total_loss = loss
        
        return total_loss
    
    def cma_loss_single_fast(self):
        # global true_eigenvalue1, true_intensity1, bound, print_flag, true_T, true_X
        
        #calculate Hamiltonian and ev, ef, H
        sol = super().Hamiltonian()
        
        Jx   =super().Jx()
        Jy   =super().Jy()
        Jz   =super().Jz
        dim  =len(Jx)
        J    =self.J
        
        eigenvalues  = sol[0]
        Eigenvectors = sol[1]
        HMatrix      = sol[2]
        
        #take in experimentally observed energy levels
        true_eigenvalue=self.true_eigenvalue
        true_intensity=self.true_intensity
        
        calint=super().Neutron_Intensity_fast(1,0)
        Intensity=calint[1:len(true_intensity)+1]
        # print(Intensity)
        # print(true_intensity)
        # ----------------------------------------------------------------
        loss1=0
        for i in range(len(true_eigenvalue)):
            loss1+=(np.linalg.det((true_eigenvalue[i] + eigenvalues[0]) * np.eye(int(2*J+1)) - HMatrix))**2  
        loss1 = np.log10(np.absolute(loss1)) 
        
        loss2=0.0
        loss2 = np.sqrt(np.sum((true_intensity - Intensity)**2))/ np.sqrt(np.sum((true_intensity)**2))
        
        # print((np.linalg.det((true_eigenvalue[0] + eigenvalues[0]) * np.eye(int(2*J+1)) - HMatrix))**2  )

        # loss3 = np.sqrt(np.sum((true_intensity25 - intensity25)**2))/ np.sqrt(np.sum((true_intensity25)**2))
        
        # T, X = calc_x(Eigenvectors1,Jx1, Jy1, Jz1, (sol1[1]-sol1[1][0])/0.086173303)
        # lossX = np.sqrt(np.mean(((1./X - 1./true_X))**2.0))/np.sqrt(np.mean(((1./true_X))**2.0))
        
        # t=100.0
        # k=100.0
        # X=100
        # loss = np.maximum(-20,loss1)+np.maximum(0.001,loss2*t) +loss3*k# + X*lossX
        loss=loss1+10*loss2
        # print(loss1,loss2)
        total_loss = loss
        
        return total_loss
    
    def cma_loss_single_fast_mag(self):
        # global true_eigenvalue1, true_intensity1, bound, print_flag, true_T, true_X
        
        #calculate Hamiltonian and ev, ef, H
        sol = super().magsolver()
        
        Jx   =super().Jx()
        Jy   =super().Jy()
        Jz   =super().Jz
        dim  =len(Jx)
        J    =self.J
        
        eigenvalues  = sol[0]
        Eigenvectors = sol[1]
        HMatrix      = sol[2]
        
        #take in experimentally observed energy levels
        true_eigenvalue=self.true_eigenvalue
        true_intensity=self.true_intensity
        
        # calint=super().Neutron_Intensity_fast(1,0)
        # Intensity=calint[1:len(true_intensity)+1]
        # print(Intensity)
        # print(true_intensity)
        # ----------------------------------------------------------------
        loss1=0
        for i in range(len(true_eigenvalue)):
            loss1+=(np.linalg.det((true_eigenvalue[i] + eigenvalues[0]) * np.eye(int(2*J+1)) - HMatrix))**2  
        loss1 = np.log10(np.absolute(loss1)) 
        
        # loss2=0.0
        # loss2 = np.sqrt(np.sum((true_intensity - Intensity)**2))/ np.sqrt(np.sum((true_intensity)**2))
        
        # print((np.linalg.det((true_eigenvalue[0] + eigenvalues[0]) * np.eye(int(2*J+1)) - HMatrix))**2  )

        # loss3 = np.sqrt(np.sum((true_intensity25 - intensity25)**2))/ np.sqrt(np.sum((true_intensity25)**2))
        
        # T, X = calc_x(Eigenvectors1,Jx1, Jy1, Jz1, (sol1[1]-sol1[1][0])/0.086173303)
        # lossX = np.sqrt(np.mean(((1./X - 1./true_X))**2.0))/np.sqrt(np.mean(((1./true_X))**2.0))
        
        # t=100.0
        # k=100.0
        # X=100
        # loss = np.maximum(-20,loss1)+np.maximum(0.001,loss2*t) +loss3*k# + X*lossX
        loss=loss1
        # print(loss1, loss2)
        total_loss = loss
        
        return total_loss
    
#%%
if __name__=="__main__":
    alpha=0.01*10.0*4/(45*35)
    beta=0.01*100.0*2/(11*15*273)
    gamma=0.01*10.0*8/(13**2*11**2*3**3*7)
    Stevens_idx=[[2,0],[2,1],[2,2],[4,0],[4,1],[4,2],[4,3],[4,4],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]
    test=pd.read_csv(f'C:/Users/qmc/OneDrive/ONRL/Data/CEF/Python/Eradam/Eradam_MPI_Newfit_goodsolution.csv',header=None)
    Parameter=dict()
    temp=5
    field=0
    
    Para=np.zeros(15)
    for i in range(15):
        Para[i]=test[i][0]
        
    Para[2]=10*Para[2] #22 -2
    Para[4]=0.1*Para[4] #41 - 4
    Para[6]=10*Para[6]  #43 -6
    Para[9]=0.1*Para[9] #61 -9
    Para[11]=10*Para[11] #63 -11
    Para[13]=10*Para[13] #65 -13
    Para[14]=10*Para[14] #66 -14
    
    true_eigenvalue = np.array([1.77, 5.25, 7.17,  13.72, 22.58, 27.81, 49.24]) #type in experiment measured values
    true_intensity  = np.array([   1.00,0.365,0.000,0.167,0.074,0.027,0.010])
    # true_X=np.array([0.059169, 0.054744, 0.050943, 0.047583, 0.044683, 0.041926, 0.039777])
    obj=Optimization('Er3+', Stevens_idx, alpha, beta, gamma, Para, temp, field,true_eigenvalue,true_intensity)
    # ev,_,_=obj.Hamiltonian()
    # obj.test(Parameter)
    # print(np.round(ev-ev[0],3))

    def opt(Para):
        '''The optimization requires defining a function to have Parameter(Para) as sole input'''
        return Optimization('Er3+', Stevens_idx, alpha, beta, gamma, Para, temp, field,true_eigenvalue,true_intensity).cma_loss_single()

#%%##############################################
# 	  The main algorithm with parallel capability
#     To parallel this on multi-core cpu:
#        1. Open a python console such as Anaconda Prompt
#        2. Use cd "Path" to navigate to the code folder
#        3. Type mpiexec -n numprocs python -m mpi4py pyfile
#################################################

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #--------------------------------------
    # Process the spectrum data
    
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
    
    ntry = 1
    
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
        res = cma.fmin(opt, x_init, 1e-7, options={'maxfevals': 10000000, 'tolfacupx': 1e9}, args=(), gradf=None, \
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
    
    
