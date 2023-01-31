import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import scipy.linalg as LA
import Operators_Yb2Ti2O7_normalized as kp
from scipy.optimize import minimize
import time
import pyswarms as ps
# import pandas as pd
# from mpi4py import MPI
import os
import time
import pdb

x_true = np.asarray([1.135, -0.0615, 0.315, 0.0011, 0.037, 0.005])
B20  = x_true[0]
B40  = x_true[1]
B43  = x_true[2]
B60  = x_true[3]
B63  = x_true[4]
B66  = x_true[5]

true_sol = kp.solver(B20,B40,B43,B60,B63,B66)
Jx=true_sol[3]
Jy=true_sol[4]
Jz=true_sol[5]
E=(true_sol[1] - true_sol[1][0])/0.0862

true_eigenvalue0 = true_sol[1]
true_eigenvalue1 = true_eigenvalue0 - true_eigenvalue0[0]
true_eigenvalue1 = true_eigenvalue1[[2,4,6]]
true_intensity0 = true_sol[0].round(3).real
true_intensity1 = true_intensity0


stepnum=100
#vector = [1,1,1,1,1,1]
lossZhang=[]
lossMSR=[]
PlotB=[]
vec=np.random.randn(6)
vec=vec/np.linalg.norm(vec)

for i in np.arange(0, 2.1,0.003):
    
    D20=D40=D43=D60=D63=D66=0
    # D20=B20+i*vec[0]*np.abs(x_true[0])
    # D40=B40+i*vec[1]*np.abs(x_true[1])
    # D43=B43+i*vec[2]*np.abs(x_true[2])
    # D60=B60+i*vec[3]*np.abs(x_true[3])
    # D63=B63+i*vec[4]*np.abs(x_true[4])
    # D66=B66+i*vec[5]*np.abs(x_true[5])
    
    # D20=-1+i*vec[0]
    # D40=-1+i*vec[1]
    # D43=-1+i*vec[2]
    # D60=-1+i*vec[3]
    # D63=-1+i*vec[4]
    # D66=-1+i*vec[5]
    D20=-1+i
    D40=1-i
    D43=-1+i
    D60=-1+i
    D63=-1+i
    D66=-1+i


    sol1 = kp.solver(D20,D40,D43,D60,D63,D66)

    Jx1=sol1[3]
    Jy1=sol1[4]
    Jz1=sol1[5]
    E1=(sol1[1] - sol1[1][0])/0.0862

    eigenvalues1 = sol1[1]

    eigenvectors1 = np.transpose(sol1[2])

    HMatrix1 = sol1[6]

    intensity1 = sol1[0].real
    
    CalcEnergy=eigenvalues1-eigenvalues1[0]
    CalcEnergy=CalcEnergy[[2,4,6]]

    loss1 = 0.0
    loss1 = loss1 + (np.linalg.det((true_eigenvalue1[0] + eigenvalues1[0]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[0] * np.eye(8)))**2
    loss1 = loss1 + (np.linalg.det((true_eigenvalue1[1] + eigenvalues1[0]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[1] * np.eye(8)))**2
    loss1 = loss1 + (np.linalg.det((true_eigenvalue1[2] + eigenvalues1[0]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[2] * np.eye(8)))**2
    loss1=np.log10(np.absolute(loss1))
    # loss1=loss1+np.sqrt(np.mean((true_intensity1 - intensity1)**2.0))/np.sqrt(np.mean(intensity1**2.0))
    
    loss2=0
    loss2=loss2+np.sqrt(np.mean((true_eigenvalue1 - CalcEnergy)**2.0))/np.sqrt(np.mean(true_eigenvalue1**2.0))
    # loss2=np.log10(np.absolute(loss2))
    # loss2=loss2+np.sqrt(np.mean((true_intensity1 - intensity1)**2.0))/np.sqrt(np.mean(intensity1**2.0))
    
    lossZhang.append(loss1)
    lossMSR.append(loss2)
    PlotB.append(D20)

fig,ax=plt.subplots(2)
ax[0].plot(PlotB,lossZhang)
ax[1].plot(PlotB,lossMSR)