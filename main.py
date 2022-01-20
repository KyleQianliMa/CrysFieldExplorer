import Yb2O3 as kp
import numpy as np
import matplotlib.pyplot as plt
# B20  = 1.135
# B40  = -0.0615
# B43  = 0.315
# B60  = 0.0011
# B63  = 0.037
# B66 = 0.005
#Yb1
B20 = 1.135
B40 = -0.0615
B43 = 0.315
B4n3=0
B60 = 0.0011
B63 = 0.037
B6n3=0
B66 = 0.005
B6n6 = 0.005

#Yb2
BB20 = 1.135
BB22 = 0
BB2n2 = 0
BB40 = -0.0615
BB42 = 0
BB4n2 = 0
BB44 = 0
BB4n4 = 0
BB60 = 0.0011
BB62 = 0
BB6n2 = 0
BB64 = 0
BB6n4 =0
BB66 = 0.005
BB6n6 = 0

solyb1=kp.solveryb1(B20,B40,B43,B4n3,B60,B63,B6n3,B66,B6n6)
CalcEnergy1=[solyb1[1][2].round(2),solyb1[1][4].round(2), solyb1[1][6]]
calcscattering1=np.array([solyb1[0][0][0,0],solyb1[0][1][0,0],solyb1[0][2][0,0]]).real.round(3) #percentage of relative intensity
Eigenvectors1=np.transpose(solyb1[2]).round(3)

solyb2=kp.solveryb2(BB20,BB22,BB2n2,BB40,BB42,BB4n2,BB44,BB4n4,BB60,BB62,BB6n2,BB64,BB6n4,BB66,BB6n6)
CalcEnergy2=[solyb2[1][2].round(2),solyb2[1][4].round(2), solyb2[1][6]]
calcscattering2=np.array([solyb2[0][0][0,0],solyb2[0][1][0,0],solyb2[0][2][0,0]]).real.round(3) #percentage of relative intensity
Eigenvectors2=np.transpose(solyb2[2]).round(3)

totalscatt=np.concatenate((calcscattering1,calcscattering2))
totalenergy=np.concatenate((CalcEnergy1,CalcEnergy2))
combined=np.vstack((totalenergy,totalscatt)).T
combined=combined[combined[:,0].argsort()]


def x(Eigenvectors, Jx, Jy, Jz, E):
     S=1/2;L=3;J=7/2;
     gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
     Na=6.0221409e23
     muB=9.274009994e-21
     kb=1.38064852e-16
     C=0.92*(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
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

T,X1=x(Eigenvectors1,solyb1[3], solyb1[4], solyb1[5], solyb1[1]/0.0862)
T,X2=x(Eigenvectors2,solyb2[3], solyb2[4], solyb2[5], solyb2[1]/0.0862)
X=X1+X2
plt.plot(T,1/X,'-')
plt.xlim(0,300)
plt.ylim(0, 400)