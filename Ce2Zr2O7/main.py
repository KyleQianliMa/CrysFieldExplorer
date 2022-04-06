import Operators_Ce2Zr2O7 as kp
import numpy as np
import matplotlib.pyplot as plt
# B20  = 1.135
# B40  = -0.0615
# B43  = 0.315
# B60  = 0.0011
# B63  = 0.037
# B66 = 0.005

B20  = 0.835
B40  = 0.299
B43  = 2.875


sol=kp.solver(B20,B40,B43)
CalcEnergy=[sol[1][2].round(2),sol[1][4].round(2)]
calcscattering=np.array(sol[0]).round(3).real.squeeze() #percentage of relative intensity
Eigenvectors=np.transpose(sol[2]).round(3)

E=np.linspace(0, 150,300)

def L(x,A,a1,dE):
    pi=3.14159265358
    Y=(A/pi)*(a1/2)/((x-dE)**2+(a1/2)**2)    
    return Y
a1=3.7
I_paper=L(E,1.2, a1, 55.9)+L(E,1, a1, 110.5)

I_fitted=L(E,calcscattering[0], a1, CalcEnergy[0])+L(E,calcscattering[1], a1, CalcEnergy[1])

plt.plot(E,I_paper,'.')
plt.plot(E,I_fitted,'-')


# Jx=sol[3]
# Jy=sol[4]
# Jz=sol[5]
# E=sol[1]/0.0862
# S=1/2;L=3;J=5/2;
# gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))

# def x(Eigenvectors, Jx, Jy, Jz, E):
#     S=1/2;L=3;J=7/2;
#     gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
#     Na=6.0221409e23
#     muB=9.274009994e-21
#     kb=1.38064852e-16
#     C=0.92*(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
#     Z=0
#     T=np.linspace(1, 320,320)
#     for n in range(0,6):
#         Z=Z+np.exp(-E[n]/T)
#     X=0
#     for n in range(0,6):
#         X=X+(np.absolute(Eigenvectors[n,:]*Jx*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
#         for m in range(0,6):
#             if np.abs(E[m]-E[n])<1e-5: continue
#             else: X = X+ (np.absolute(Eigenvectors[m,:]*Jx*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
#     for n in range(0,6):
#         X=X+(np.absolute(Eigenvectors[n,:]*Jy*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
#         for m in range(0,6):
#             if np.abs(E[m]-E[n])<1e-5: continue
#             else: X = X+ (np.absolute(Eigenvectors[m,:]*Jy*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
#     for n in range(0,6):
#         X=X+(np.absolute(Eigenvectors[n,:]*Jz*Eigenvectors[n,:].H).item())**2*(np.exp(-E[n]/T))/T
#         for m in range(0,6):
#             if np.abs(E[m]-E[n])<1e-5: continue
#             else: X = X+ ((np.absolute(Eigenvectors[m,:]*Jz*Eigenvectors[n,:].H).item())**2)*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
#     X=C*X/(3*Z)
#     return T,X

# T,X=x(Eigenvectors,sol[3], sol[4], sol[5], sol[1]/0.0862)
# plt.plot(T,1/X,'-')
# plt.xlim(0,300)
# plt.ylim(0, 600)
# test=1/X