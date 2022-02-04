import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import scipy.linalg as LA
import Yb2O3 as kp
from scipy.optimize import minimize
import time
import pyswarms as ps
import pandas as pd
import true_data as td





def cal_x(Eigenvectors, E, Jx, Jy, Jz, T):
     S=1/2;L=3;J=7/2;
     gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
     Na=6.0221409e23
     muB=9.274009994e-21
     kb=1.38064852e-16
     C=0.92*(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
     Z=0
     # T=np.linspace(1, 320,320)
     for n in range(0,8):
         Z=Z+np.exp(-E[n]/T)
     X=0
     for n in range(0,8):
         X=X+(np.absolute(Eigenvectors[n,:]*Jx*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
         for m in range(0,8):
             if np.abs(E[m]-E[n])<1e-5: continue
             else: X = X+ (np.absolute(Eigenvectors[m,:]*Jx*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
     for n in range(0,8):
         X=X+(np.absolute(Eigenvectors[n,:]*Jy*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
         for m in range(0,8):
             if np.abs(E[m]-E[n])<1e-5: continue
             else: X = X+ (np.absolute(Eigenvectors[m,:]*Jy*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
     for n in range(0,8):
         X=X+(np.absolute(Eigenvectors[n,:]*Jz*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
         for m in range(0,8):
             if np.abs(E[m]-E[n])<1e-5: continue
             else: X = X+ ((np.absolute(Eigenvectors[m,:]*Jz*Eigenvectors[n,:].H).item())**2)*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
     X=C*X/(3*Z)
     return T,X







# This loss is exact the same as loss_func_one_sample. This loss is used for PSO and the loss_func_one_sample is used for the scipy optimizer
def loss_func_ensemble(x):

	global true_eigenvalue1, true_intensity1, true_eigenvalue2, true_intensity2, true_T, true_X, bound, print_flag

	ns = x.shape[0]

	total_loss = np.zeros(ns)

	for i in range(ns):

		x_norm = x[i,:]

		sol1 = kp.solveryb1(x_norm[0],x_norm[1],x_norm[2],x_norm[3],x_norm[4],x_norm[5],x_norm[6],x_norm[7],x_norm[8])

		Jx1=sol1[3]
		Jy1=sol1[4]
		Jz1=sol1[5]
		E1=(sol1[1] - sol1[1][0])/0.0862

		eigenvalues1 = sol1[1]

		eigenvectors1 = np.transpose(sol1[2])

		HMatrix1 = sol1[6]

		intensity1 = sol1[0].real

		T1,X1=cal_x(eigenvectors1, E1, Jx1, Jy1, Jz1, true_T.copy())

		# ----------------------------------------------------------------

		sol2 = kp.solveryb2(x_norm[9],x_norm[10],x_norm[11],x_norm[12], x_norm[13],x_norm[14],x_norm[15],x_norm[16],x_norm[17],x_norm[18],x_norm[19],x_norm[20],x_norm[21],x_norm[22],x_norm[23])

		Jx2=sol2[3]
		Jy2=sol2[4]
		Jz2=sol2[5]
		E2=(sol2[1] - sol2[1][0])/0.0862

		eigenvalues2 = sol2[1]

		eigenvectors2 = np.transpose(sol2[2])

		HMatrix2 = sol2[6]

		intensity2 = sol2[0].real

		T2,X2=cal_x(eigenvectors2, E2, Jx2, Jy2, Jz2, true_T.copy())

		X = 0.25 * X1 + 0.75 * X2


		# ----------------------------------------------------------------
		#
		loss1 = 0.0
		loss1 = loss1 + (np.linalg.det((true_eigenvalue1[0] + eigenvalues1[0]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[0] * np.eye(8) - HMatrix1))**2 
		loss1 = loss1 + (np.linalg.det((true_eigenvalue1[1] + eigenvalues1[0]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[1] * np.eye(8) - HMatrix1))**2 
		loss1 = loss1 + (np.linalg.det((true_eigenvalue1[2] + eigenvalues1[0]) * np.eye(8) - HMatrix1))**2 / (np.linalg.det(true_eigenvalue1[2] * np.eye(8) - HMatrix1))**2 

		loss1 = loss1 + (np.linalg.det((true_eigenvalue2[0] + eigenvalues2[0]) * np.eye(8) - HMatrix2))**2 / (np.linalg.det(true_eigenvalue2[0] * np.eye(8) - HMatrix2))**2
		loss1 = loss1 + (np.linalg.det((true_eigenvalue2[1] + eigenvalues2[0]) * np.eye(8) - HMatrix2))**2 / (np.linalg.det(true_eigenvalue2[1] * np.eye(8) - HMatrix2))**2
		loss1 = loss1 + (np.linalg.det((true_eigenvalue2[2] + eigenvalues2[0]) * np.eye(8) - HMatrix2))**2 / (np.linalg.det(true_eigenvalue2[2] * np.eye(8) - HMatrix2))**2

		loss2  = np.sqrt(np.mean(((true_intensity1 - intensity1)/true_intensity1)**2)) + np.sqrt(np.mean(((true_intensity2 - intensity2)/true_intensity2)**2))

		# loss3 = np.sqrt(np.mean(((1./X - 1./true_X)/(1./true_X))**2.0))
		loss3 = np.sqrt(np.mean(((X - true_X)/(true_X))**2.0))

		loss = np.absolute(loss1) * 0.0 + loss2 * 0.0 + loss3 * 100.0

		total_loss[i] = loss

		if print_flag: 
			print(loss1, loss2, loss3)
			print(intensity1, true_intensity1)
			print(intensity2, true_intensity2)
			plt.plot(T1,1./X)
			plt.plot(T1,1./true_X)
			plt.show()

	return total_loss


##############################
# 	  The main algorithm
##############################

#--------------------------------------
# Process the T-X data
# T = pd.read_excel('T_X.xlsx')
# T = np.array(T)
# np.savetxt('TX.txt',T)
TX = np.loadtxt('TX.txt')
T_min = np.amin(TX[:,0])
T_max = np.amax(TX[:,0])
true_T = np.linspace(T_min, T_max, 300)
true_X = np.interp(true_T,TX[:,0],TX[:,1])




par_dim = 24
bound = np.zeros((2,par_dim))
bound[:,0]  = [-20.0, 20.0]
bound[:,1]  = [-1.0, 1.0]
bound[:,2]  = [-10.0, 10.0]
bound[:,3]  = [-5, 5]
bound[:,4]  = [-0.01, 0.01]
bound[:,5]  = [-0.01, 0.01]
bound[:,6]  = [-0.1, 0.1]
bound[:,7]  = [-0.1, 0.1]
bound[:,8]  = [-0.1, 0.1]

bound[:,9]  = [-10, 10]
bound[:,10] = [-10, 10]
bound[:,11] = [-20, 20]
bound[:,12] = [-1, 1]
bound[:,13] = [-0.1, 0.1]
bound[:,14] = [-10, 10]
bound[:,15] = [-10, 10]
bound[:,16] = [-1, 1]
bound[:,17] = [-0.01, 0.01]
bound[:,18] = [-1e-4, 1e-4]
bound[:,19] = [-0.01, 0.01]
bound[:,20] = [-0.01, 0.01]
bound[:,21] = [-1e-3, 1e-3]
bound[:,22] = [-0.01, 0.01]
bound[:,23] = [-1e-4, 1e-4]

print_flag = False


#-----------------------------------------------------------------------------
#                    First-round optimization using PSO
#-----------------------------------------------------------------------------
# x_init = np.random.rand(1,par_dim) * 2.0 - 1.0
# # x_init = 2.0*(x_true - bound[0,:]) / (bound[1,:]-bound[0,:]) - 1.0
# # x_init = x_init.reshape((1,par_dim))
# f_init = loss_func_ensemble(x_init)
# # print(f_init)
# # exit()

case_num = 3

true_eigenvalue1, true_eigenvalue2, true_intensity1, true_intensity2 = td.get_data(case_num)

n_particles = 200

x_init = np.random.rand(n_particles,par_dim) * 2.0 - 1.0
for i in range(n_particles):
	x_init[i,:] = x_init[i,:] * (bound[1,:]-bound[0,:])*0.5 + (bound[1,:]+bound[0,:])*0.5


bnds = (bound[0,:], bound[1,:])

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=par_dim, options=options, bounds=bnds, init_pos=x_init)
# optimizer = ps.single.GlobalBestPSO(n_particles=400, dimensions=par_dim, options=options, bounds=bnds, init_pos=np.matlib.repmat(x_init, 400,1))
# optimizer = ps.single.GlobalBestPSO(n_particles=400, dimensions=par_dim, options=options, bounds=bnds, init_pos=np.random.rand(400,par_dim) * 2.0 - 1.0)

cost, pos = optimizer.optimize(loss_func_ensemble, iters=500)


print('App_PSO   parameters = ', pos)


print_flag = True
f_final = loss_func_ensemble(pos.reshape((1, par_dim)))
print(f_final)


filename = './Result/Case_'+str(case_num)+'.txt'
file1 = open(filename, 'a')  # append mode
np.savetxt(file1, pos.reshape((1, par_dim)), fmt='%2.3e', delimiter=', ')
file1.write("\n")
file1.close()



