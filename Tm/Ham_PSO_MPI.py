import numpy as np
# import matplotlib.pyplot as plt
import numpy.matlib
import scipy.linalg as LA
import Operators_Tm as kp
from scipy.optimize import minimize
import time
import pyswarms as ps
import pandas as pd
from mpi4py import MPI
import os
import time
# import true_data as td
S=1;L=5;J=6;
def L(x,A,a1,dE):
    pi=3.14159265358
    Y=(A/pi)*(a1/2.0)/((x-dE)**2+(a1/2.0)**2)    
    return Y


def cal_x(Eigenvectors, E, Jx, Jy, Jz):

    gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
    Na=6.0221409e23
    muB=9.274009994e-21
    kb=1.38064852e-16
    C=1*(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
    Z=0
    T=np.linspace(1, 320,320)
    for n in range(0,6):
        Z=Z+np.exp(-E[n]/T)
    X=0
    for n in range(0,6):
        X=X+(np.absolute(Eigenvectors[n,:]*Jx*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
        for m in range(0,6):
            if np.abs(E[m]-E[n])<1e-5: continue
            else: X = X+ (np.absolute(Eigenvectors[m,:]*Jx*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
    for n in range(0,6):
        X=X+(np.absolute(Eigenvectors[n,:]*Jy*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
        for m in range(0,6):
            if np.abs(E[m]-E[n])<1e-5: continue
            else: X = X+ (np.absolute(Eigenvectors[m,:]*Jy*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
    for n in range(0,6):
        X=X+(np.absolute(Eigenvectors[n,:]*Jz*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
        for m in range(0,6):
            if np.abs(E[m]-E[n])<1e-5: continue
            else: X = X+ ((np.absolute(Eigenvectors[m,:]*Jz*Eigenvectors[n,:].H).item())**2)*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
    X=C*X/(3*Z)
    return T,X



# This loss is exact the same as loss_func_one_sample. This loss is used for PSO and the loss_func_one_sample is used for the scipy optimizer
def loss_func_ensemble(x):

    global true_eigenvalue1, true_intensity1, bound, print_flag, true_E, true_I, true_T, true_X

    ns = x.shape[0]

    total_loss = np.zeros(ns)

    for i in range(ns):

        x_norm = x[i,:]
        sol1 = kp.solver(x_norm[0],x_norm[1],x_norm[2],x_norm[3],x_norm[4],x_norm[5])

        Jx1=sol1[3]
        Jy1=sol1[4]
        Jz1=sol1[5]
        E1=(sol1[1] - sol1[1][0])/0.0862

        eigenvalues1 = sol1[1]

        eigenvectors1 = np.transpose(sol1[2])

        HMatrix1 = sol1[6]

        intensity1 = sol1[0].real

        CalcEnergy1 = [sol1[1][1],sol1[1][2],sol1[1][3],sol1[1][4],sol1[1][5],sol1[1][6],sol1[1][7],sol1[1][8],sol1[1][9],sol1[1][10],sol1[1][11],sol1[1][12]] - sol1[1][0] 
        calcscattering1=np.array([sol1[0][0],sol1[0][1],sol1[0][2]]).real
        #calcscattering1=np.array(sol1[0]).round(3).real.squeeze()

        a1=1000000000000 #!!!!!!Change Peak width to observed dataI_cal=0
        I_cal=0
        for i in range(0,13):
            I_cal =I_cal+ L(true_E, calcscattering1[i], a1, CalcEnergy1[i])
        I_cal = I_cal / np.sum(I_cal)

        # ----------------------------------------------------------------

        loss1 = 0.0
        for i in range(0,13):
            loss1 = loss1 + (np.linalg.det((true_eigenvalue1[i] + eigenvalues1[i]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[i] * np.eye(8)))**2  

        loss2 = np.sqrt(np.mean((true_I - I_cal)**2.0))/np.sqrt(np.mean(true_I**2.0)) # Spectrum Fitting

        loss4 = np.mean((true_intensity1 - intensity1)**2) # Discrete CEF transition level fitting

        T, X = cal_x(eigenvectors1, E1, Jx1, Jy1, Jz1)
        loss3 = np.sqrt(np.mean(((1./X - 1./true_X))**2.0))/np.sqrt(np.mean(((1./true_X))**2.0))

        loss = np.absolute(loss1) * 1e10 + loss2 * 0.0 + loss3 * 1.0 + loss4 * 20.0 # Spectrum fitting turned off at the start

        total_loss[i] = loss

    return total_loss



# This loss is exact the same as loss_func_one_sample. This loss is used for PSO and the loss_func_one_sample is used for the scipy optimizer
def print_loss(x):

    global true_eigenvalue1, true_intensity1, bound, print_flag, true_E, true_I, true_T, true_X

    ns = x.shape[0]

    total_loss = np.zeros(ns)

    for i in range(ns):

        x_norm = x[i,:]
        sol1 = kp.solver(x_norm[0],x_norm[1],x_norm[2],x_norm[3],x_norm[4],x_norm[5])

        Jx1=sol1[3]
        Jy1=sol1[4]
        Jz1=sol1[5]
        E1=(sol1[1] - sol1[1][0])/0.0862

        eigenvalues1 = sol1[1]

        eigenvectors1 = np.transpose(sol1[2])

        HMatrix1 = sol1[6]

        intensity1 = sol1[0].real

        CalcEnergy1 = [sol1[1][1],sol1[1][2],sol1[1][3],sol1[1][4],sol1[1][5],sol1[1][6],sol1[1][7],sol1[1][8],sol1[1][9],sol1[1][10],sol1[1][11],sol1[1][12]] - sol1[1][0] 
        calcscattering1=np.array([sol1[0][0],sol1[0][1],sol1[0][2]]).real
        #calcscattering1=np.array(sol1[0]).round(3).real.squeeze()

        a1=1000000000000 #!!!!!!Change Peak width to observed dataI_cal=0
        I_cal=0
        for i in range(0,13):
            I_cal =I_cal+ L(true_E, calcscattering1[i], a1, CalcEnergy1[i])
        I_cal = I_cal / np.sum(I_cal)

        # ----------------------------------------------------------------

        loss1 = 0.0
        for i in range(0,13):
            loss1 = loss1 + (np.linalg.det((true_eigenvalue1[i] + eigenvalues1[i]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[i] * np.eye(8)))**2  

        loss2 = np.sqrt(np.mean((true_I - I_cal)**2.0))/np.sqrt(np.mean(true_I**2.0)) # Spectrum Fitting

        loss4 = np.mean((true_intensity1 - intensity1)**2) # Discrete CEF transition level fitting

        T, X = cal_x(eigenvectors1, E1, Jx1, Jy1, Jz1)
        loss3 = np.sqrt(np.mean(((1./X - 1./true_X))**2.0))/np.sqrt(np.mean(((1./true_X))**2.0))

        loss = np.absolute(loss1) * 1e10 + loss2 * 0.0 + loss3 * 1.0 + loss4 * 20.0 # Spectrum fitting turned off at the start

        total_loss[i] = loss

    return total_loss, loss1, loss2, loss3, eigenvalues1, intensity1




##############################
#       The main algorithm
##############################

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()


#--------------------------------------
# Process the spectrum data

true_E=np.linspace(0, 150, 300)

a1=1000000000 #!!!Same as before, set this to experimentally observed peak width
true_I = L(true_E,0.284, a1, 76.8)+L(true_E,1, a1, 82.12)+L(true_E, 0.19, a1, 116.24)#Sum up all the intensities
true_I = true_I / np.sum(true_I)


par_dim = 6
bound = np.zeros((2,par_dim))
# bound[:,0]  = [-2.0, 2.0]
# bound[:,1]  = [-0.1, 0.1]
# bound[:,2]  = [-1.0, 1.0]
# bound[:,3]  = [-0.01, 0.01]
# bound[:,4]  = [-0.1, 0.1]
# bound[:,5]  = [-0.01, 0.01]

bound[:,0]  = [-10.0, 10.0]
bound[:,1]  = [-10.0, 10.0]
bound[:,2]  = [-10.0, 10.0]
bound[:,3]  = [-10.0, 10.0]
bound[:,4]  = [-10.0, 10.0]
bound[:,5]  = [-10.0, 10.0]



# true_sol = kp.solver(B20,B40,B43,B60,B63,B66)
# Jx=true_sol[3]
# Jy=true_sol[4]
# Jz=true_sol[5]
# E=(true_sol[1] - true_sol[1][0])/0.0862

true_eigenvalue1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
true_intensity1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

# print(true_eigenvalue1)
# print(np.array(true_sol[0]).round(3).real.squeeze())
# exit()

true_T,true_X=np.array[1,2]#!!!!need to import data here

ntry = 90*1

case_num=1 #labeling for different scenarios

final_result_buff = np.zeros((ntry, par_dim+17))
final_result = np.zeros((ntry, par_dim+17))

# comm.Barrier()

for iter_num  in range(ntry):

    # if rank == iter_num%size:

        # print_flag = False

        #-----------------------------------------------------------------------------
        #                    First-round optimization using PSO
        #-----------------------------------------------------------------------------
        n_particles = 400

        x_init = np.random.rand(n_particles,par_dim) * 2.0 - 1.0
        for i in range(n_particles):
            x_init[i,:] = x_init[i,:] * (bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5

        # x_init = np.matlib.repmat((np.random.rand(1,par_dim) * 2.0 - 1.0) * (bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5, n_particles, 1)

        # # x_init = 2.0*(x_true - bound[0,:]) / (bound[1,:]-bound[0,:]) - 1.0
        # # x_init = x_init.reshape((1,par_dim))
        # f_init = loss_func_ensemble(x_init)
        # print(f_init)
        # exit()

        # print_flag = True
        # f_true = loss_func_ensemble(true_par)
        # # # print((true_par/bound[1,:]).shape)
        # print(f_true)
        # exit()


        # x1 = 2.0*(np.asarray([0.84333546, 0.29882854, 2.86984116]) - bound[0,:]) / (bound[1,:]-bound[0,:]) - 1.0 
        # x1 = x1.reshape((1,par_dim))
        # f_init1 = loss_func_ensemble(x1)

        # print(f_init1)
        # exit()


        max_bound = np.ones(par_dim)
        min_bound = - max_bound
        bnds = (min_bound, max_bound)
        bnds = (bound[0,:], bound[1,:])

        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=par_dim, options=options, bounds=bnds, init_pos=x_init)
        # optimizer = ps.single.GlobalBestPSO(n_particles=400, dimensions=par_dim, options=options, bounds=bnds, init_pos=np.random.rand(400,par_dim) * 2.0 - 1.0)

        cost, pos = optimizer.optimize(loss_func_ensemble, iters=1000)

        # print_flag = True

        f_final = print_loss(pos.reshape((1, par_dim)))

        final_result_buff[iter_num, 0:par_dim] = pos
        final_result_buff[iter_num, par_dim+0] = f_final[0] # total loss
        final_result_buff[iter_num, par_dim+1] = np.absolute(f_final[1]) # loss 1
        final_result_buff[iter_num, par_dim+2] = f_final[2] # loss 2
        final_result_buff[iter_num, par_dim+3] = f_final[3] # loss 3
        final_result_buff[iter_num, par_dim+4:par_dim+8] = f_final[4][[0,2,4,6]]        # eigenvalues
        final_result_buff[iter_num, par_dim+8:par_dim+11] = true_eigenvalue1 # eigenvalues
        final_result_buff[iter_num, par_dim+11:par_dim+14] = f_final[5]        # intensity
        final_result_buff[iter_num, par_dim+14:par_dim+17] = true_intensity1 # intensity
        # final_result_buff[iter_num, par_dim+17:par_dim+21] = f_final[6][[0,2,4,6]]        # eigenvalues
        # final_result_buff[iter_num, par_dim+21:par_dim+24] = true_eigenvalue2 # eigenvalues
        # final_result_buff[iter_num, par_dim+24:par_dim+27] = f_final[7]        # intensity
        # final_result_buff[iter_num, par_dim+27:par_dim+30] = true_intensity2 # intensity



# comm.Barrier()
# comm.Reduce([final_result_buff, ntry*(par_dim+17), MPI.DOUBLE], [final_result, ntry*(par_dim+17), MPI.DOUBLE], op=MPI.SUM, root=0)

# if rank == 0:

        np.savetxt('Yb2Ti2O7_result_case'+str(case_num)+'.csv', final_result, fmt='%2.3e', delimiter=', ')

# comm.Barrier()


# exit()


# total_loss, loss1, loss2, loss3, eigenvalues1, intensity1, eigenvalues2, intensity2

# np.savetxt(file1, pos.reshape((1, par_dim)), fmt='%2.3e', delimiter=', ')


# total_loss, loss1, loss2, loss3, eigenvalues2, intensity2

    # fig, axes = plt.subplots(3,5, figsize=(15,10))

    # for i in range(3):
    #     for j in range(5):

    #         ind = i*5 + j

    #         lb = pos[ind] * 0.9
    #         ub = pos[ind] * 1.1

    #         ns = 100
    #         sample = np.matlib.repmat(pos, ns, 1)
    #         sample[:,ind] = np.linspace(lb, ub, ns).T
    #         f_sample = loss_func_ensemble(sample)

    #         axes[i][j].plot(np.linspace(lb, ub, ns), f_sample, '-')
    #         axes[i][j].set_xlabel(str(ind))

    # plt.show()




    # #-----------------------------------------------------------------------------
    # #                    Second-round optimization using scipy
    # #-----------------------------------------------------------------------------

    # # iter_count = 1
    # bnds = ((bound[0,0],bound[1,0]),\
    #         (bound[0,1],bound[1,1]),\
    #         (bound[0,2],bound[1,2]),\
    #         (bound[0,3],bound[1,3]),\
    #         (bound[0,4],bound[1,4]),\
    #         (bound[0,5],bound[1,5]),\
    #         (bound[0,6],bound[1,6]),\
    #         (bound[0,7],bound[1,7]),\
    #         (bound[0,8],bound[1,8]),\
    #         (bound[0,9],bound[1,9]),\
    #         (bound[0,10],bound[1,10]),\
    #         (bound[0,11],bound[1,11]),\
    #         (bound[0,12],bound[1,12]),\
    #         (bound[0,13],bound[1,13]),\
    #         (bound[0,14],bound[1,14]),\
    #     )
    # res = minimize(loss_func_one_sample, pos, method='Powell', tol=1e-8, bounds = bnds, options={'gtol': 1e-20,'eps': 1e-8,'maxiter':40000,'disp': False,'return_all': True})

    # print(res)



    # #-----------------------------------------------------------------------------
    # #                    Print results
    # #-----------------------------------------------------------------------------

    # eigenvalues_app1, eigenvectors_app1, HMatrix_app1, Intensity_app1 = SimpleCalculationFunction(pos*(bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5, gamma, scale, hw1, temp)
    # eigenvalues_app2, eigenvectors_app2, HMatrix_app2, Intensity_app2 = SimpleCalculationFunction(res.x*(bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5, gamma, scale, hw1, temp)

    # print('-------------------')
    # print('App_PSO    eigenvalues = ', eigenvalues_app1)
    # print('App_Scipy  eigenvalues = ', eigenvalues_app2)
    # print('True eigenvalues = ', true_eigenvalue0)
    # print('-------------------')
    # print('App_PSO   parameters = ', pos * (bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5)
    # print('App_PSO   parameters = ', pos.T)
    # print('App_Scipy parameters = ', res.x * (bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5)
    # print('True parameters = ', x_true)
    # print('-------------------')



    
    # print(f_final)
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print(iter_num)
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')




# figure, (ax1) = plt.subplots(1, 1)
# ax1.plot(hw1, true_intensity0,label='true')
# ax1.plot(hw1, Intensity_app1,label='PSO')
# ax1.plot(hw1, Intensity_app2,label='Scipy')
# # plt.title('eigenvalue index: '+str(2)+','+str(4)+','+str(12)+' par:'+str(pos * bound[1,:]))
# ax1.legend()

# app_intensity = Intensity / np.linalg.norm(Intensity)

# scale_factor = np.linalg.norm(np.reshape(Intensity, (50*150))[notnan_ind])/np.linalg.norm(true_intensity2)

# print('loss2 = ', np.sqrt(np.mean((true_intensity - app_intensity)**2))) #/ np.linalg.norm(true_intensity2))
# print('scale factor = ', scale_factor)

# plt.show()


