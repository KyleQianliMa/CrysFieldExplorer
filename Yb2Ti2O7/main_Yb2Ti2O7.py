import Operators_Yb2Ti2O7 as kp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
Par=pd.read_csv('Yb2Ti2O7_result_case6.csv',header=None)
# B20  = 1.135
# B40  = -0.0615
# B43  = 0.315
# B60  = 0.0011
# B63  = 0.037
# B66 = 0.005
def L(x,A,a1,dE):
    pi=3.14159265358
    Y=(A/pi)*(a1/2)/((x-dE)**2+(a1/2)**2)    
    return Y

def x(Eigenvectors, Jx, Jy, Jz, E):
    S=1/2;L=3;J=7/2;
    gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
    Na=6.0221409e23
    muB=9.274009994e-21
    kb=1.38064852e-16
    C=0.92*(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
    Z=0
    T=np.linspace(1, 320,320)
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



i=0
B20  =Par[0][i]#1.135
B40  =Par[1][i]#-0.0615
B43  = Par[2][i]#0.315
B60  = Par[3][i]#0.0011
B63  = Par[4][i]#0.037
B66  = Par[5][i]#0.005

sol=kp.solver(B20,B40,B43,B60,B63,B66)
CalcEnergy=[sol[1][2].round(2),sol[1][4], sol[1][6]]
calcscattering=np.array(sol[0]).round(3).real.squeeze() #percentage of relative intensity
Eigenvectors=(sol[2]).round(3)
E_x=np.linspace(0, 150,300)
a1=3.7
I_paper=L(E_x,0.284, a1, 76.8)+L(E_x,1, a1, 82.12)+L(E_x,0.19, a1, 116.24)
  
I_fitted=L(E_x,calcscattering[0], a1, CalcEnergy[0])+L(E_x,calcscattering[1], a1, CalcEnergy[1])\
 +L(E_x,calcscattering[2], a1, CalcEnergy[2])

Jx=sol[3]
Jy=sol[4]
Jz=sol[5]
Jplus=sol[6]
E=sol[1]/0.0862
S=1/2;L=3;J=7/2;
gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))

gx=np.absolute(-2*gj*np.absolute(Eigenvectors[:,0].H*Jx*Eigenvectors[:,1]))
gy=np.absolute(-2*gj*np.absolute(Eigenvectors[:,0].H*Jy*Eigenvectors[:,1]))
gz=np.absolute(2*gj*np.absolute(Eigenvectors[:,0].H*Jz*Eigenvectors[:,0]))

#--------------------Magnetization-------------------------------
B=[]
Magnetization=[]
for k in range(1000,60000,1000):
    # i=70000
    Bx=k/10000   
    By=k/10000
    Bz=k/10000
    S=1/2;L=3;J=7/2;
    gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
    muBT = 5.7883818012e-2
    k_B = 8.6173303e-2
    T=10 #temperature where magnetization is measured
    M=0
    for m in range(0,1000):
        X=np.random.normal(0,1,3)[0]
        Y=np.random.normal(0,1,3)[1]
        Z=np.random.normal(0,1,3)[2]
        norm=np.sqrt(X**2+Y**2+Z**2)
        X=X/norm
        Y=Y/norm
        Z=Z/norm        
        mag1=kp.magsovler1(B20,B40,B43,B60,B63,B66,Bx*X,By*Y,Bz*Z)
        E1=mag1[0] 
        magvec1=mag1[1]
        jx=mag1[2]
        jy=mag1[3]
        jz=mag1[4]
        M1x=0
        M1y=0
        M1z=0
        Z1=0
        gmubJ=gj*(jx+jy+jz)
        
        for n in range(0,8):
            Z1=Z1+np.exp(-E1[n]/(k_B*T))  
    
        for n in range(0,8):
            M1x=M1x+((magvec1[:,n].H*(gj*jx)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1y=M1y+((magvec1[:,n].H*(gj*jy)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1z=M1z+((magvec1[:,n].H*(gj*jz)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1x=M1x[0,0].real
            M1y=M1y[0,0].real
            M1z=M1z[0,0].real
        M=M+(M1x*X+M1y*Y+M1z*Z)
    # for m in range(0,150):
    #     X=np.random.rand(1,3)[0,0]
    #     Y=np.random.rand(1,3)[0,1]
    #     Z=np.random.rand(1,3)[0,2]
    #     norm=np.sqrt(X**2+Y**2+Z**2)
    #     X=X/norm
    #     Y=Y/norm
    #     Z=Z/norm
    #     M=M+np.sqrt((M1x*X)**2+(M1y*Y)**2+(M1z*Z)**2)
    Magnetization.append(M/10000)
    B.append(k/10000)
    print(k)
#--------------------Plots---------------------------------------
T,X=x(Eigenvectors,sol[3], sol[4], sol[5], sol[1]/0.0862)
fig,ax=plt.subplots(2,2)
ax[0,0].plot(T,1/X,'-')
ax[0,0].set_xlim(0,300)
ax[0,0].set_ylim(0, 600)

ax[0,1].plot(E_x,I_paper,'.')
ax[0,1].plot(E_x,I_fitted,'-')

ax[1,0].plot(B,Magnetization)

ax[1,1].plot(i,gx,'.')
ax[1,1].plot(i,gz,'*')