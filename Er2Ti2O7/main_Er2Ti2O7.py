import Operators_Er2Ti2O7 as kp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alive_progress import alive_bar
from time import sleep
from matplotlib.ticker import FormatStrFormatter
import sympy as sym
from sympy import symbols, solve,cos,sin,Matrix, Inverse
from scipy import optimize
from scipy.special import wofz
from scipy import integrate
from scipy.stats import chisquare
import scipy.linalg as LA
from scipy.interpolate import BSpline

np.load.__defaults__=(None, True, True, 'ASCII')
# Par=pd.read_csv('Yb2Ti2O7_result_case6.csv',header=None)
# B20  = 1.135
# B40  = -0.0615
# B43  = 0.315
# B60  = 0.0011
# B63  = 0.037
# B66 = 0.005
def lorentzian(x,A,a1,dE):
    pi=3.14159265358
    Y=(A/pi)*(a1/2)/((x-dE)**2+(a1/2)**2)
    return Y

def gtensor(Jx,Jy,Jz,minus,plus):
    Sx=[[(plus.H*Jx*plus).item(), (minus.H*Jx*plus).item()],
        [(plus.H*Jx*minus).item(),(minus.H*Jx*minus).item()],
        ]
    # Sx=np.true_divide(Sx,2)

    Sy=[[(plus.H*Jy*plus).item(), (minus.H*Jy*plus).item()],
        [(plus.H*Jy*minus).item(),(minus.H*Jy*minus).item()],
        ]
    # Sy=np.true_divide(Sy,2)

    Sz=[[(plus.H*Jz*plus).item(), (minus.H*Jz*plus).item()],
        [(plus.H*Jz*minus).item(),(minus.H*Jz*minus).item()],
        ]
    # Sz=np.true_divide(Sz,2)

    g=[[Sx[1][0].real, Sx[1][0].imag, Sx[0][0].real],
       [Sy[1][0].real, Sy[1][0].imag, Sy[0][0].real],
       [Sz[1][0].real, Sz[1][0].imag, Sz[0][0].real],
        ]
    g=np.dot((2*gj),g)
    return g

#-----To Guannan: Copy this section for cal_x function--------
S=3/2;L=6;J=15/2;
dim=int(2*J+1)
gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
def x(Eigenvectors, Jx, Jy, Jz, E):
     Na=6.0221409e23
     muB=9.274009994e-21
     kb=1.38064852e-16
     C=(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
     Z=0
     T=np.linspace(1, 400,200)
     for n in range(0,dim):
         Z=Z+np.exp(-E[n]/T)
     X=0
     for n in range(0,dim):
         for m in range(0,dim):
             if np.abs(E[m]-E[n])<1e-5: X=X+(np.absolute(Eigenvectors[:,n].H*Jx*Eigenvectors[:,m]).item())**2*(np.exp(-E[n]/T))/T
             else: X = X+ 2*(np.absolute(Eigenvectors[:,m].H*Jx*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T))/(E[m]-E[n])
     for n in range(0,dim):
         for m in range(0,dim):
             if  np.abs(E[m]-E[n])<1e-5: X=X+(np.absolute(Eigenvectors[:,n].H*Jy*Eigenvectors[:,m]).item())**2*(np.exp(-E[n]/T))/T
             else: X = X+ 2*(np.absolute(Eigenvectors[:,m].H*Jy*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T))/(E[m]-E[n])
     for n in range(0,dim):
         for m in range(0,dim):
             if  np.abs(E[m]-E[n])<1e-5: X=X+(np.absolute(Eigenvectors[:,n].H*Jz*Eigenvectors[:,m]).item())**2*(np.exp(-E[n]/T))/T
             else: X = X+ 2*((np.absolute(Eigenvectors[:,m].H*Jz*Eigenvectors[:,n]).item())**2)*(np.exp(-E[n]/T))/(E[m]-E[n])
     X=C*X/(3*Z)
     return T,X
#----------------------------------------------------------------------
def magnetization(B20,B40,B43,B60,B63,B66,T):
    B=[]
    Magnetization=[]
    with alive_bar(21,bar='bubbles') as bar:
        for k in range(1000,210000,10000):
            # i=70000
            Bx=k/10000
            By=k/10000
            Bz=k/10000
            # S=1/2;L=3;J=7/2;
            gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
            muBT = 5.7883818012e-2
            k_B = 8.6173303e-2
            #T=10 #temperature where magnetization is measured
            M=0
            for m in range(0,500):
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

                for n in range(0,dim):
                    Z1=Z1+np.exp(-E1[n]/(k_B*T))

                for n in range(0,dim):
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
            Magnetization.append(M/500)
            B.append(k/10000)
            # print('Field=',k)
            sleep(0.03)
            bar()
            bar.title('Magnetization')
    return B, Magnetization

def dmdh(B20,B40,B43,B60,B63,B66,T):
    k=100
    Bx=k/10000
    By=k/10000
    Bz=k/10000
    # S=1/2;L=3;J=7/2;
    gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
    muBT = 5.7883818012e-2
    k_B = 8.6173303e-2
    Na=6.0221409e23
    muB=9.274009994e-21
    #T=10 #temperature where magnetization is measured
    M=0
    for m in range(0,500):
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

        for n in range(0,dim):
            Z1=Z1+np.exp(-E1[n]/(k_B*T))

        for n in range(0,dim):
            M1x=M1x+((magvec1[:,n].H*(gj*jx)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1y=M1y+((magvec1[:,n].H*(gj*jy)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1z=M1z+((magvec1[:,n].H*(gj*jz)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1x=M1x[0,0].real
            M1y=M1y[0,0].real
            M1z=M1z[0,0].real
        M=M+(M1x*X+M1y*Y+M1z*Z)
        MB=Na*muB*M/Bx/500/10000
    return MB

def calchi(O,E):
    summation=0
    for i in range(len(O)):
        summation+=(O[i]-E[i])**2/E[i]
    return summation

def chi2(CalcEnergy, calcscattering,Mag,chi, RealE, RealI, Realmag, Realchi):
    chisqr=0
    chisqr=calchi(CalcEnergy,RealE)+calchi(calcscattering,RealI)+calchi(Mag, Realmag)+calchi(chi,Realchi)
    return chisqr

S=3/2;L=6;J=15/2;
dim=int(2*J+1)
gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))

alpha=4/(45*35)
beta=2/(11*15*273)
gamma=8/(13*13*11*11*3*3*3*7)
theta_6=-1/891891
r2=0.6980
r4=1.218
r6=4.502
i=0
B20  = 37.5*alpha*r2
B40  = 33.5*beta*r4
B43  = 282*beta*r4
B60  = 1.25*gamma*r6
B63  = -17.15*gamma*r6
B66  = 21.6*gamma*r6

sol=kp.solver(B20,B40,B43,B60,B63,B66)
CalcEnergy=[sol[1][2].round(2),sol[1][4], sol[1][6], sol[1][8],sol[1][10],sol[1][12],sol[1][14]]-sol[1][0]
calcscattering=np.array(sol[0]).round(3).real.squeeze() #percentage of relative intensity
Eigenvectors=(sol[2]).round(3)
E_x=np.linspace(0, 150,150)
a1=3.0
I_paper=lorentzian(E_x,1, a1, 6.3)+lorentzian(E_x,0.75, a1, 7.3)+lorentzian(E_x,0.20, a1, 15.7)\
    +lorentzian(E_x,0.04, a1, 60.2)+lorentzian(E_x,0.04, a1, 62.3)+lorentzian(E_x,0.09, a1, 66.3)+lorentzian(E_x,0.01, a1, 87.2)

for i in range(0,8):
    I_fitted=lorentzian(E_x,calcscattering[0], a1, CalcEnergy[0])+lorentzian(E_x,calcscattering[1], a1, CalcEnergy[1])+lorentzian(E_x,calcscattering[2], a1, CalcEnergy[2])+\
        lorentzian(E_x,calcscattering[3], a1, CalcEnergy[3])+lorentzian(E_x,calcscattering[4], a1, CalcEnergy[4])+lorentzian(E_x,calcscattering[5], a1, CalcEnergy[5])+lorentzian(E_x,calcscattering[6], a1, CalcEnergy[6])

Jx=sol[3]
Jy=sol[4]
Jz=sol[5]
Jplus=sol[6]
E=sol[1]/0.0862
gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))

gx=np.absolute(-2*gj*np.absolute(Eigenvectors[:,0].H*Jx*Eigenvectors[:,1]))
gy=np.absolute(-2*gj*np.absolute(Eigenvectors[:,0].H*Jy*Eigenvectors[:,1]))
gz=np.absolute(2*gj*np.absolute(Eigenvectors[:,0].H*Jz*Eigenvectors[:,0]))

g=gtensor(Jx,Jy,Jz,Eigenvectors[:,0],Eigenvectors[:,1])

theta=symbols('theta')
A=Matrix([[cos(theta),0,sin(theta)],
          [0,1,0],
          [-sin(theta),0,cos(theta)]])

gprime=g*A.inv()
theta=solve(gprime[0,2]-gprime[2,0])[0]
A=Matrix([[cos(theta),0,sin(theta)],
          [0,1,0],
          [-sin(theta),0,cos(theta)]])

gprime=g*A.inv()
gprime=sym.Matrix(gprime)
print('Original Matrix=\n', gprime)
gmatrix=gprime.diagonalize()[1]
print('gtensor=\n',gmatrix)
gxx=gmatrix[0,0]
gyy=gmatrix[1,1]
gzz=gmatrix[2,2]

#-------------------------susceptibility by dM/dH----------------
temp=[]
chi=[]
with alive_bar(31) as bar:
    for T in range(1,310,10):
        X=dmdh(B20,B40,B43,B60,B63,B66, T)
        temp.append(T)
        chi.append(X)
        # print('Temperature=', T)
        sleep(0.03)
        bar.title('Calculating dM/dH')
        bar()

temp=np.array(temp)
chi=np.array(chi)
#--------------------Magnetization-------------------------------

B,Magnetization=magnetization(B20, B40, B43, B60, B63, B66,10)
#--------------------Plots---------------------------------------
T,X=x(Eigenvectors,sol[3], sol[4], sol[5], sol[1]/0.0862)
fig,ax=plt.subplots(2,2)
ax[0,0].plot(T,1/X,'-',temp, 1/chi,'.')
ax[0,0].set_xlim(0,300)
ax[0,0].set_ylim(0, 50)

ax[0,1].plot(E_x,I_paper,'.')
ax[0,1].plot(E_x,I_fitted,'-')

ax[1,0].plot(B,Magnetization)

ax[1,1].plot(i,gx,'.')
ax[1,1].plot(i,gz,'*')