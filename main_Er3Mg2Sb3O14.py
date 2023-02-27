import Er3Mg2Sb3O14 as kp
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
np.load.__defaults__=(None, True, True, 'ASCII')

TX=np.loadtxt("TX.txt")

test=pd.read_csv('x_solu.csv',header=None)
MH=Exp_MH_5K = np.loadtxt("Exp_Er-tripod_MH.dat", skiprows=1, max_rows=99,  usecols=[1,2])     # MH data
H=MH[:,0]
M=MH[:,1]
Exp_1 = np.loadtxt("Exp_Er-tripod_30meV_5K.dat", skiprows=1, usecols=[0,1,2])     #INS data
Exp_4 = np.loadtxt("Exp_Er-tripod_120meV_5K.dat", skiprows=1, usecols=[0,1,2])     #INS data
RealE=[0, 0, 6.49402, 6.49402, 10.3456, 10.3456, 21.386, 21.386, 49.8102, 49.8102, 58.4949, 58.4949, 64.6031,64.6031, 68.0658, 68.0658]
RealI=[1,0.325306,0.276555,0.418738,0.0357799,0.0962073,0.0618048,]
Realmag=np.loadtxt("Magnetization.txt")

def lorentzian(x,A,a1,dE):
    pi=3.14159265358
    Y=(A/pi)*(a1/2)/((x-dE)**2+(a1/2)**2)
    return Y

def calchi(O,E):
    summation=0
    for i in range(len(O)):
        if E[i]<2e-5:continue
        else:
            summation+=(O[i]-E[i])**2/E[i]
    return summation

def chi2(CalcEnergy, calcscattering,Mag,chi, RealE, RealI, Realmag, Realchi):
    chisqr=0
    chisqr=calchi(CalcEnergy,RealE)+calchi(calcscattering,RealI)+calchi(Mag, Realmag)+calchi(chi,Realchi)
    print(calchi(CalcEnergy,RealE),calchi(calcscattering,RealI),calchi(Mag, Realmag),calchi(chi,Realchi))
    return chisqr

S=3/2;L=6;J=15/2
gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
def x(Eigenvectors, Jx, Jy, Jz, E):
     # S=3/2;L=6;J=15/2
     # gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
     Na=6.0221409e23
     muB=9.274009994e-21
     kb=1.38064852e-16
     C=(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
     Z=0
     # T=np.linspace(1, 400,400)
     T=TX[:,0]
     for n in range(0,16):
         Z=Z+np.exp(-E[n]/T)
     X=0
     for n in range(0,16):
         X=X+(np.absolute(Eigenvectors[:,n].H*Jx*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T))/T
         for m in range(0,16):
             if E[m]==E[n]: continue
             else: X = X+ (np.absolute(Eigenvectors[:,m].H*Jx*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
     for n in range(0,16):
         X=X+(np.absolute(Eigenvectors[:,n].H*Jy*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T))/T
         for m in range(0,16):
             if  E[m]==E[n]: continue
             else: X = X+ (np.absolute(Eigenvectors[:,m].H*Jy*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
     for n in range(0,16):
         X=X+(np.absolute(Eigenvectors[:,n].H*Jz*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T))/T
         for m in range(0,16):
             if  E[m]==E[n]: continue
             else: X = X+ ((np.absolute(Eigenvectors[:,m].H*Jz*Eigenvectors[:,n]).item())**2)*(np.exp(-E[n]/T)-np.exp(-E[m]/T))/(E[m]-E[n])
     X=C*X/(3*Z)
     return T,X

def gtensor(Jx,Jy,Jz,minus,plus):
    Sx=[[(plus.H*Jx*plus).item(), (minus.H*Jx*plus).item()],
        [(plus.H*Jx*minus).item(),(minus.H*Jx*minus).item()],
        ]
    # Sx=np.transpose(Sx)
    # Sx=np.true_divide(Sx,2)

    Sy=[[(plus.H*Jy*plus).item(), (minus.H*Jy*plus).item()],
        [(plus.H*Jy*minus).item(),(minus.H*Jy*minus).item()],
        ]
    # Sy=np.transpose(Sy)
    # Sy=np.true_divide(Sy,2)

    Sz=[[(plus.H*Jz*plus).item(), (minus.H*Jz*plus).item()],
        [(plus.H*Jz*minus).item(),(minus.H*Jz*minus).item()],
        ]
    # Sy=np.transpose(Sy)
    # Sz=np.true_divide(Sz,2)

    g=[[Sx[1][0].real, Sx[1][0].imag, Sx[0][0].real],
       [Sy[1][0].real, Sy[1][0].imag, Sy[0][0].real],
       [Sz[1][0].real, Sz[1][0].imag, Sz[0][0].real],
        ]
    g=np.dot((2*gj),g)
    return g

def magnetization(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,T):
    B=[]
    Magnetization=[]
    with alive_bar(15,bar='bubbles') as bar:
        for k in range(1000,150000,10000):
            # i=70000
            Bx=k/10000
            By=k/10000
            Bz=k/10000
            S=3/2;L=6;J=15/2
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
                mag1=kp.magsovler1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,Bx*X,By*Y,Bz*Z)
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

                for n in range(0,16):
                    Z1=Z1+np.exp(-E1[n]/(k_B*T))

                for n in range(0,16):
                    M1x=M1x+((magvec1[:,n].H*(gj*jx)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
                    M1y=M1y+((magvec1[:,n].H*(gj*jy)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
                    M1z=M1z+((magvec1[:,n].H*(gj*jz)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
                    M1x=M1x[0,0].real
                    M1y=M1y[0,0].real
                    M1z=M1z[0,0].real
                M=M+(M1x*X+M1y*Y+M1z*Z)
            Magnetization.append(M/500)
            B.append(k/10000)
            # print(k)
            sleep(0.03)
            bar()
            bar.title('Magnetization')
    return B, Magnetization

def sus(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,T):
    m=100
    Bx=m/10000
    By=m/10000
    Bz=m/10000
    S=3/2;L=6;J=15/2
    gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
    muBT = 5.7883818012e-2
    Na=6.0221409e23
    muB=9.274009994e-21
    k_B = 8.6173303e-2
    # M=0
    # for m in range(0,500):
    # X=np.random.normal(0,1,3)[0]
    # Y=np.random.normal(0,1,3)[1]
    # Z=np.random.normal(0,1,3)[2]
    # norm=np.sqrt(X**2+Y**2+Z**2)
    # X=X/norm
    # Y=Y/norm
    # Z=Z/norm
    mag1x=kp.magsovler1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,Bx,0,0)
    E1x=mag1x[0]
    magvec1x=mag1x[1]

    mag1y=kp.magsovler1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,0,By,0)
    E1y=mag1y[0]
    magvec1y=mag1y[1]

    mag1z=kp.magsovler1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,0,0,Bz)
    E1z=mag1z[0]
    magvec1z=mag1z[1]

    jx=mag1x[2]
    jy=mag1x[3]
    jz=mag1x[4]
    M1x=0
    M1y=0
    M1z=0
    Z1x=0
    Z1y=0
    Z1z=0
    muBT=5.7883818012e-2
    gmubJ=gj*(jx+jy+jz)

    for n in range(0,16):
        Z1x=Z1x+np.exp(-E1x[n]/(k_B*T))

    for n in range(0,16):
        M1x=M1x+((magvec1x[:,n].H*(gj*jx)*magvec1x[:,n])/Z1x)*np.exp(-E1x[n]/((k_B*T)))
        M1x=M1x[0,0].real

    for n in range(0,16):
        Z1y=Z1y+np.exp(-E1y[n]/(k_B*T))

    for n in range(0,16):
        M1y=M1y+((magvec1y[:,n].H*(gj*jx)*magvec1y[:,n])/Z1y)*np.exp(-E1y[n]/((k_B*T)))
        M1y=M1y[0,0].real

    for n in range(0,16):
        Z1z=Z1z+np.exp(-E1z[n]/(k_B*T))

    for n in range(0,16):
        M1z=M1z+((magvec1z[:,n].H*(gj*jx)*magvec1z[:,n])/Z1z)*np.exp(-E1z[n]/((k_B*T)))
        M1z=M1z[0,0].real
    X=Na*muB*(M1x/Bx+M1y/By+M1z/Bz)/3/10000
    return X

def dmdh(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,T):
    k=100
    Bx=k/10000
    By=k/10000
    Bz=k/10000
    S=3/2;L=6;J=15/2
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
        mag1=kp.magsovler1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,Bx*X,By*Y,Bz*Z)
        E1=mag1[0]
        magvec1=mag1[1]
        jx=mag1[2]
        jy=mag1[3]
        jz=mag1[4]
        M1x=0
        M1y=0
        M1z=0
        Z1=0

        for n in range(0,16):
            Z1=Z1+np.exp(-E1[n]/(k_B*T))

        for n in range(0,16):
            M1x=M1x+((magvec1[:,n].H*(gj*jx)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1y=M1y+((magvec1[:,n].H*(gj*jy)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1z=M1z+((magvec1[:,n].H*(gj*jz)*magvec1[:,n])/Z1)*np.exp(-E1[n]/((k_B*T)))
            M1x=M1x[0,0].real
            M1y=M1y[0,0].real
            M1z=M1z[0,0].real
        M=M+(M1x*X+M1y*Y+M1z*Z)
        MB=Na*muB*M/Bx/500/10000
    return MB
alpha= 4/(45*35)
beta=2/(11*15*273)
gamma=8/(13**2*11**2*3**3*7)

# B20 = -4.81*alpha
# B21 = 6.51*alpha
# B22 = -57.37*alpha
# B40 = 27.39*beta
# B41 = -1.91*beta
# B42 = 17.19*beta
# B43 = 290.52*beta
# B44= -22.24*beta
# B60 = 2.21*gamma
# B61 = -0.66*gamma
# B62 = -8.94*gamma
# B63 = -83.42*gamma
# B64 = 9.17*gamma
# B65 = 12.34*gamma
# B66 = 90.74*gamma


# test=np.array([-0.00428620,0.03220003,-0.13602942,0.00119561,-0.00009416,0.00084995,0.01305353,-0.00134215,\
#                 0.00000500,-0.00000401,-0.00002230,-0.00017266,0.00002577,0.00004230,0.00018612])
# B20 = test[0]
# B21 = test[1]
# B22 = test[2]
# B40 = test[3]
# B41 = test[4]
# B42 = test[5]
# B43 = test[6]
# B44= test[7]
# B60 = test[8]
# B61 = test[9]
# B62 = test[10]
# B63 = test[11]
# B64 =test[12]
# B65 = test[13]
# B66 = test[14]

Bpar=[]
I_plot30=[]
I_plot120=[]
X_plot=[]
mag_plot=[]
g_plot=[]
num=146

for i in range(0,num):
# i=0
    B20 = test[0][i]
    B21 = test[1][i]
    B22 = test[2][i]
    B40 = test[3][i]
    B41 = test[4][i]
    B42 = test[5][i]
    B43 = test[6][i]
    B44=  test[7][i]
    B60 = test[8][i]
    B61 = test[9][i]
    B62 = test[10][i]
    B63 = test[11][i]
    B64 = test[12][i]
    B65 = test[13][i]
    B66 = test[14][i]

    sol=kp.solver(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66)
    CalcEnergy1=[sol[1]-sol[1][0]]
    CalcEnergy1=CalcEnergy1[0].squeeze()
    calcscattering1=np.array(sol[0]).real.squeeze() #percentage of relative intensity
    Eigenvectors1=sol[2]
    Jx=sol[3]
    Jy=sol[4]
    Jz=sol[5]
    gx=np.absolute(-2*gj*np.absolute(Eigenvectors1[:,0].H*Jx*Eigenvectors1[:,1]))
    gy=np.absolute(-2*gj*np.absolute(Eigenvectors1[:,0].H*Jy*Eigenvectors1[:,1]))
    gz=np.absolute(2*gj*np.absolute(Eigenvectors1[:,0].H*Jz*Eigenvectors1[:,0]))
    #remeber to change n for different elements too!
    a1_30=1.5
    area_30=a1_30*3.1415926/2
    E_x_30=np.linspace(0,30,90)
    I_paper_30=lorentzian(E_x_30,1*area_30,a1_30,6.4)+lorentzian(E_x_30,0.5*area_30,a1_30,10.5)+lorentzian(E_x_30,0.2*area_30,a1_30,21.6)
    I_fitted_30=lorentzian(E_x_30,calcscattering1[0]*area_30,a1_30,CalcEnergy1[2])+\
        lorentzian(E_x_30,calcscattering1[1]*area_30,a1_30,CalcEnergy1[4])+lorentzian(E_x_30,calcscattering1[2]*area_30,a1_30,CalcEnergy1[6])

    a1_120=4
    area_120=a1_120*3.1415926/2
    E_x_120=np.linspace(26,90,180)
    I_paper_120=lorentzian(E_x_120,0.2*area_120,a1_120,50)+lorentzian(E_x_120,0.04*area_120,a1_120,65)+lorentzian(E_x_120,0.04*area_120,a1_120,67.5)
    I_fitted_120=lorentzian(E_x_120,calcscattering1[3]*area_120,a1_120,CalcEnergy1[8])+lorentzian(E_x_120,calcscattering1[4]*area_120,a1_120,CalcEnergy1[10])\
        +lorentzian(E_x_120,calcscattering1[5]*area_120,a1_120,CalcEnergy1[12])+lorentzian(E_x_120,calcscattering1[6]*area_120,a1_120,CalcEnergy1[14])
    I_fitted_120=I_fitted_120/2


    g=gtensor(Jx,Jy,Jz,Eigenvectors1[:,1],Eigenvectors1[:,0])
    # def Ry(theta):
    #     return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    # def Pesduospinrotation(theta):
    #     A=g.dot(np.linalg.inv(Ry(theta)))
    #     B=A-A.T
    #     return B.item((2, 0))
    # x=optimize.fsolve(np.vectorize(Pesduospinrotation), 0, xtol=1e-6)
    # g2=np.dot(g,np.linalg.inv(Ry(x[0])))
    # print('gxx,gzz,gyy are', np.linalg.eigvals(g2))

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
    gprime=np.array(gprime,dtype=float)
    print(gprime)
    print('gxx=',np.linalg.eigvals(gprime)[0],'gyy=',np.linalg.eigvals(gprime)[2],'gzz=',np.linalg.eigvals(gprime)[1])
    gxx=np.absolute(np.linalg.eigvals(gprime)[0])
    gyy=np.absolute(np.linalg.eigvals(gprime)[2])
    gzz=np.absolute(np.linalg.eigvals(gprime)[1])

    #-----------plot results------------------------------------------------------
    T1,X1=x(Eigenvectors1,sol[3], sol[4], sol[5], (sol[1]-sol[1][0])/0.086173303)
    B,Mag5=magnetization(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,5)
    temp=[]
    chi=[]
    with alive_bar(25) as bar:
        for T in TX[:,0]:
            X=dmdh(B20, B21, B22, B40, B41, B42, B43, B44, B60, B61, B62, B63, B64, B65, B66, T)
            temp.append(T)
            chi.append(X)
            bar.title('Calculating dM/dH')
            bar()

    temp=np.array(temp)
    chi=np.array(chi)

    # fig,ax=plt.subplots(3,2)
    # fig.suptitle('B20={} gtensor ={},{},{}'.format(B20,gxx.round(3),gyy.round(3),gzz.round(3)))
    # ax[0,0].plot(temp,1/chi,T1,1/X1, TX[:,0],TX[:,1],'.')

    # ax[0,1].plot(B,Mag5,H,M,'.')
    # ax[0,1].legend(loc="lower right")
    # ax[0,1].set_ylim(0,6)
    # gfactor=['gx','gy','gz']
    # gvalue=[np.absolute(gxx),np.absolute(gyy),np.absolute(gzz)]
    # ax[1,1].bar(gfactor,gvalue)

    # ax[1,0].plot(E_x_30,I_paper_30,E_x_30,I_fitted_30,'.')
    # ax[2,0].plot(E_x_120,I_paper_120,E_x_120,I_fitted_120,'.')

    # plt.show()

    Bpar.append([B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,chi2(CalcEnergy1, calcscattering1, Mag5, 1/chi, RealE, RealI, Realmag, TX[:,1])]) #TX[:,1] is already 1/X
    I_plot30.append(I_paper_30)
    I_plot120.append(I_paper_120)
    X_plot.append(chi)
    mag_plot.append(Mag5)
    g_plot.append([gxx,gyy,gzz])
    print(i)

#-----------------plot-----------------------------
Bpar=np.array(Bpar)
Bsort=Bpar[np.argsort(Bpar[:,-1])]
order=np.argsort(Bpar[:,-1])
color=['yellow','green','pink']
xlim=-0.25
ylim=0.25
casenum=10
#--------------panel1-------------------------------
fig,ax=plt.subplots(3,3,figsize=(16,16),dpi=4002)
mpl.rcParams['lines.markersize'] = 10
# fig.tight_layout()
ax[0,0].plot(Bsort[0:casenum,0],Bsort[0:casenum,15],'o',alpha=0.5)
ax[0,0].plot(-0.0122,0,'.',color='red',markersize=20,alpha=0.5)
ax[0,0].set_xlabel('B20')
ax[0,0].set_ylabel('Chi^2')
ax[0,0].set_ylim(xlim,ylim)

ax[0,1].plot(Bsort[0:casenum,1],Bsort[0:casenum,15],'o',alpha=0.5)
ax[0,1].plot(0.01653,0,'.',color='red',markersize=20,alpha=0.5)
ax[0,1].set_xlabel('B21')
# ax[0,1].set_ylabel('Chi^2')
ax[0,1].set_ylim(xlim,ylim)

ax[0,2].plot(Bsort[0:casenum,2],Bsort[0:casenum,15],'o',alpha=0.5)
ax[0,2].plot(-0.1457,0,'.',color='red',markersize=20,alpha=0.5)
ax[0,2].set_xlabel('B22')
# ax[0,2].set_ylabel('Chi^2')
ax[0,2].set_ylim(xlim,ylim)

ax[1,0].plot(Bsort[0:casenum,3],Bsort[0:casenum,15],'o',alpha=0.5)
ax[1,0].plot(0.001216,0,'.',color='red',markersize=20,alpha=0.5)
ax[1,0].set_xlabel('B40')
ax[1,0].set_ylabel('Chi^2')
ax[1,0].set_ylim(xlim,ylim)

ax[1,1].plot(Bsort[0:casenum,4],Bsort[0:casenum,15],'o',alpha=0.5)
ax[1,1].plot(-8e-5,0,'.',color='red',markersize=20,alpha=0.5)
ax[1,1].set_xlabel('B41')
# ax[1,1].set_ylabel('Chi^2')
ax[1,1].set_ylim(xlim,ylim)

ax[1,2].plot(Bsort[0:casenum,5],Bsort[0:casenum,15],'o',alpha=0.5)
ax[1,2].plot(0.000763,0,'.',color='red',markersize=20,alpha=0.5)
ax[1,2].set_xlabel('B42')
# ax[1,2].set_ylabel('Chi^2')
ax[1,2].set_ylim(xlim,ylim)
ax[1,2].ticklabel_format(style='sci',scilimits=(0,0))

ax[2,0].plot(Bsort[0:casenum,6],Bsort[0:casenum,15],'o',alpha=0.5)
ax[2,0].plot(0.012899100899100898,0,'.',color='red',markersize=20,alpha=0.5)
ax[2,0].set_xlabel('B43')
ax[2,0].set_ylabel('Chi^2')
ax[2,0].set_ylim(xlim,ylim)

ax[2,1].plot(Bsort[0:casenum,7],Bsort[0:casenum,15],'o',alpha=0.5)
ax[2,1].plot(-0.0009874569874569873,0,'.',color='red',markersize=20,alpha=0.5)
ax[2,1].set_xlabel('B44')
# ax[2,1].set_ylabel('Chi^2')
ax[2,1].set_ylim(xlim,ylim)

ax[2,2].plot(Bsort[0:casenum,8],Bsort[0:casenum,15],'o',alpha=0.5)
ax[2,2].plot(4.574550029095484e-06,0,'.',color='red',markersize=20,alpha=0.5)
ax[2,2].set_xlabel('B60')
# ax[2,2].set_ylabel('Chi^2')
ax[2,2].set_ylim(xlim,ylim)
# for i, c in zip((41,42,97),color):
#     ax[0,0].plot(Bpar[i,0],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15) #representative data
#     ax[0,1].plot(Bpar[i,1],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)
#     ax[0,2].plot(Bpar[i,2],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)
#     ax[1,0].plot(Bpar[i,3],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)
#     ax[1,1].plot(Bpar[i,4],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)
#     ax[1,2].plot(Bpar[i,5],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)
#     ax[2,0].plot(Bpar[i,6],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)
#     ax[2,1].plot(Bpar[i,7],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)
#     ax[2,2].plot(Bpar[i,8],Bpar[i,15],'o',alpha=0.5,color=c,markersize=15)


# fig.savefig('D:\CEF\Python\Er3Mg2Sb3O14\Bparameters1', format='png',dpi=600)
#--------------panel2-------------------------------
fig,ax=plt.subplots(3,3)
mpl.rcParams['lines.markersize'] = 10
# fig.tight_layout()
ax[0,0].plot(Bsort[0:casenum,9],Bsort[0:casenum,15],'o',alpha=0.5)
ax[0,0].plot(-1.3661552123090585e-06,0,'.',color='red',markersize=20,alpha=0.5)
ax[0,0].set_xlabel('B61')
ax[0,0].set_ylabel('Chi^2')
ax[0,0].set_ylim(xlim,ylim)

ax[0,1].plot(Bsort[0:casenum,10],Bsort[0:casenum,15],'o',alpha=0.5)
ax[0,1].plot(-1.8505193330368153e-05,0,'.',color='red',markersize=20,alpha=0.5)
ax[0,1].set_xlabel('B62')
# ax[0,1].set_ylabel('Chi^2')
ax[0,1].set_ylim(xlim,ylim)
ax[0,1].ticklabel_format(style='sci',scilimits=(0,0))

ax[0,2].plot(Bsort[0:casenum,11],Bsort[0:casenum,15],'o',alpha=0.5)
ax[0,2].plot(-0.00017267373910730553,0,'.',color='red',markersize=20,alpha=0.5)
ax[0,2].set_xlabel('B63')
# ax[0,2].set_ylabel('Chi^2')
ax[0,2].set_ylim(xlim,ylim)
ax[0,2].ticklabel_format(style='sci',scilimits=(0,0))

ax[1,0].plot(Bsort[0:casenum,12],Bsort[0:casenum,15],'o',alpha=0.5)
ax[1,0].plot(1.8981277722536463e-05,0,'.',color='red',markersize=20,alpha=0.5)
ax[1,0].set_xlabel('B64')
ax[1,0].set_ylabel('Chi^2')
ax[1,0].set_ylim(xlim,ylim)
ax[1,0].ticklabel_format(style='sci',scilimits=(0,0))

ax[1,1].plot(Bsort[0:casenum,13],Bsort[0:casenum,15],'o',alpha=0.5)
ax[1,1].plot(2.5542962605899667e-05,0,'.',color='red',markersize=20,alpha=0.5)
ax[1,1].set_xlabel('B65')
# ax[1,1].set_ylabel('Chi^2')
ax[1,1].set_ylim(xlim,ylim)
ax[1,1].ticklabel_format(style='sci',scilimits=(0,0))

ax[1,2].plot(Bsort[0:casenum,14],Bsort[0:casenum,15],'o',alpha=0.5)
ax[1,2].plot(0.0001878256423710969,0,'.',color='red',markersize=20,alpha=0.5)
ax[1,2].set_xlabel('B66')
# ax[1,2].set_ylabel('Chi^2')
ax[1,2].set_ylim(xlim,ylim)

for i in range(0,num):
    ax[2,0].plot(E_x_30,I_plot30[i],alpha=0.5,color='blue',linewidth=5)
ax[2,0].set_xlabel('E')
ax[2,0].set_ylabel('Intensity (arb.unit)')
ax[2,0].plot(E_x_30,I_paper_30,color='red',alpha=1,linewidth=3)

for i in range(0,num):
    ax[2,1].plot(E_x_120,I_plot120[i],alpha=0.5,color='blue',linewidth=5)
ax[2,1].set_xlabel('E')
ax[2,1].plot(E_x_120,I_paper_120,color='red',alpha=1,linewidth=3)

for i in range(0,num):
    ax[2,2].plot(temp,1/X_plot[i],alpha=0.5,color='blue',linewidth=5)
ax[2,2].set_xlabel('Susceptibility')
ax[2,2].set_ylabel('Normalized Unit')
ax[2,2].plot(temp,TX[:,1],color='red',alpha=1,linewidth=3)

# for i, c in zip((41,42,97),color):
#     ax[0,0].plot(Bpar[i,9],Bpar[i,6],'o',alpha=0.5,color=c,markersize=15) #representative data
#     ax[0,1].plot(Bpar[i,10],Bpar[i,6],'o',alpha=0.5,color=c,markersize=15)
#     ax[0,2].plot(Bpar[i,11],Bpar[i,6],'o',alpha=0.5,color=c,markersize=15)
#     ax[1,0].plot(Bpar[i,12],Bpar[i,6],'o',alpha=0.5,color=c,markersize=15)
#     ax[1,1].plot(Bpar[i,13],Bpar[i,6],'o',alpha=0.5,color=c,markersize=15)
#     ax[1,2].plot(Bpar[i,14],Bpar[i,6],'o',alpha=0.5,color=c,markersize=15)
#     ax[2,0].plot(E_x_30,I_plot30[i],alpha=0.5,color=c,linewidth=3)
#     ax[2,1].plot(E_x_120,I_plot120[i],alpha=0.5,color=c,linewidth=3)
#     ax[2,2].plot(temp,1/X_plot[i],alpha=0.5,color=c,linewidth=3)

fig.savefig('D:\CEF\Python\Er3Mg2Sb3O14\Bparamters2.png', format='png',dpi=600)


#--------------panel3-------------------------------
fig,ax=plt.subplots(2,2)
# fig.tight_layout()
for i in order[0:casenum]:
    ax[0,0].plot(B,mag_plot[i],alpha=0.5,color='blue',linewidth=4)
ax[0,0].set_xlabel('H(T)')
ax[0,0].set_ylabel('Magnetization')
ax[0,0].plot(B,Realmag,color='red',alpha=1,linewidth=4)

for i in order[0:casenum]:
    ax[0,1].plot(g_plot[i][0],g_plot[i][2]/g_plot[i][0],'.',alpha=0.5,color='blue',markersize=25)
ax[0,1].plot(14.0505,-0.03580,'.',color='red',markersize=25,alpha=0.8)
# ax[0,1].set_ylim(-1,50)
ax[0,1].set_xlabel('gx')
ax[0,1].set_ylabel('gz/gx')

for i in order[0:casenum]:
    ax[1,0].plot(g_plot[i][1],g_plot[i][2]/g_plot[i][1],'.',alpha=0.5,color='blue',markersize=25)
ax[1,0].plot(0.18,2.778,'.',color='red',markersize=25,alpha=0.8)
# ax[1,0].set_ylim(-1,5)
ax[1,0].set_xlabel('gy')
ax[1,0].set_ylabel('gz/gy')

for i in order[0:casenum]:
    ax[1,1].plot(g_plot[i][0],g_plot[i][1]/g_plot[i][0],'.',alpha=0.5,color='blue',markersize=25)
ax[1,1].plot(14.0505,0.0128,'.',color='red',markersize=25,alpha=0.8)
# ax[1,1].set_ylim(-0.1,0.1)
ax[1,1].set_xlabel('gx')
ax[1,1].set_ylabel('gy/gx')



# for i, c in zip((41,42,97),color):
#     ax[0,0].plot(B,mag_plot[i],alpha=0.8,linewidth=2,color=c)

#     ax[0,1].plot(g_plot[i][0],g_plot[i][2]/g_plot[i][0],'o',alpha=1,color=c)

#     ax[1,0].plot(g_plot[i][1],g_plot[i][2]/g_plot[i][1],'o',alpha=1,color=c)

#     ax[1,1].plot(g_plot[i][0],g_plot[i][1]/g_plot[i][0],'o',alpha=1,color=c)

fig.savefig('D:\CEF\Python\Er3Mg2Sb3O14\Properties.png', format='png',dpi=600)

g_plot=np.array(g_plot)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

for i in order[0:casenum]:
    if g_plot[i,0]>7:
               ax.scatter(g_plot[i,0],g_plot[i,1],g_plot[i,2],color='red',s=100,marker='o',alpha=0.5)
    if g_plot[i,1]>7:
               ax.scatter(g_plot[i,0],g_plot[i,1],g_plot[i,2],color='blue',s=100,marker='o',alpha=0.5)
    if g_plot[i,2]>7:
               ax.scatter(g_plot[i,0],g_plot[i,1],g_plot[i,2],color='green',s=100,marker='o',alpha=0.5)
ax.scatter(14.05,0.18,0.50, color='yellow', s=200,marker='o',alpha=1)
ax.set_xlabel('gx')
ax.set_ylabel('gy')
ax.set_zlabel('gz')
