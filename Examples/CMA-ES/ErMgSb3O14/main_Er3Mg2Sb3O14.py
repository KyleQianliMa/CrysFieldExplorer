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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
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
        for k in range(1000,150000,2000):
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
            sampling=1000
            for m in range(0,sampling):
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
            Magnetization.append(M/sampling)
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

def Rotate(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66):
    theta=(np.arctan(B21/(3*B20-B22)))/2
    ST=np.sin(theta)
    CT=np.cos(theta)
    RB22=1/2 * (3*B20*np.sin(theta)**2 - 1/2 * B21 * np.sin(2*theta) + B22*(1+np.cos(theta)**2))

    RB21=2*ST*CT*B22 - 6*ST*CT*B20 + 2*CT**2*B21-B21

    RB20=(ST**2*B22+ST*CT*B21 + 3*CT**2*B20-B20)/2

    RB44=(35*ST**4*B40-7*ST**3*CT*B41+7*ST**2*CT**2*B42 + 7*ST**2*B42-\
          ST*CT**3*B43-3*ST*CT*B43+CT**4*B44 + 6*CT**2*B44+B44)/8

    RB43=-(140*ST**3*CT*B40 - 28*ST**2*CT**2*B41 +7*ST**2*B41 - 4*ST*CT**3*B44+ 28*ST*CT**3*B42\
            -12*ST*CT*B44 - 4*CT**4*B43 - 3*CT**2*B43 + 3*B43)/4

    RB42=(ST**2*CT**2*B44 + 35*ST**2*CT**2*B40 + ST**2*B44 - 5*ST**2*B40 + \
          ST*CT**3*B43 - 7*ST*CT**3*B41 + 4*ST*CT*B41 + 7*CT**4*B42 - \
              6*CT**2*B42 +B42)/2

    RB41=(4*ST**3*CT*B44 + 4*ST**2*CT**2*B43 - ST**2*B43 + 28*ST*CT**3*B42-\
          140*ST*CT**3*B40 - 16*ST*CT*B42 + 60*ST*CT*B40 + 28*CT**4*B41-
          27*CT**2*B41 + 3*B41)/4

    RB40=(ST**4*B44 + ST**3*CT*B43 + 7*ST**2*CT**2*B42 - ST**2*B42 + 7*ST*CT**3*B41 -\
          3*ST*CT*B41 + 35*CT**4*B40 - 30*CT**2*B40 + 3*B40)/8

    RB66=(B66*CT**6 + 15*B66*CT**4 + 15*B66*CT**2 + B66 - B65*ST*CT**5 -\
          10*B65*ST*CT**3 - 5*B65*ST*CT + 11*B64*ST**2*CT**4 + 66*B64*ST**2*CT**2+\
          11*B64*ST**2 - 11*B63*ST**3*CT**3 - 33*B63*ST**3*CT + 33*B62*ST**4*CT**2+\
          33*B62*ST**4 - 33*B61*ST**5*CT + 231*B60*ST**6)/32

    RB65=(6*B66*ST*CT**5+60*B66*ST*CT**3 + 30*B66*ST*CT+6*B65*CT**6+\
          35*B65*CT**4- 20*B65*CT**2 - 5*B65 - 66*B64*ST*CT**5 - 220*B64*ST*\
              CT**3 + 110*B64*ST*CT + 66*B63*ST**2*CT**4 +99*B63*ST**2*CT**2-\
                  33*B63*ST**2 - 198*B62*ST**3*CT**3 - 66*B62*ST**3*CT+198*B61*ST**4*CT**2-\
                      33*B61*ST**4-1386*B60*ST**5*CT)/16

    RB64=(3*B66*ST**2*CT**4 + 18*B66*ST**2*CT**2 + 3*B66*ST**2 + 3*B65*ST*CT**5 + 10 *B65*ST*CT**3-\
          5*B65*ST*CT + 33*B64*CT**6 + 35*B64*CT**4- 65*B64*CT**2 + 13*B64- 33*B63*ST*CT**5 -\
          6*B63*ST*CT**3 + 15*B63*ST*CT + 99*B62*ST**2*CT**4 -30*B62*ST**2*CT**2 + 3*B62*ST**2\
              -99*B61*ST**3*CT**3 +39*B61*ST**3*CT + 693*B60*ST**4*CT**2 - 63*B60*ST**4)/16

    RB63=(10*B66*ST**3*CT**3 + 30*B66*ST**3*CT + 10*B65*ST**2*CT**4 + \
          15*B65*ST**2*CT**2 - 5*B65*ST**2 +110*B64*ST*CT**5 + 20*B64*ST*CT**3 -\
          50*B64*ST*CT + 110*B63*CT**6 - 105*B63*CT**4 + 12*B63*CT**2 - B63-
          330*B62*ST*CT**5 + 300*B62*ST*CT**3 - 66*B62*ST*CT + 330*B61*ST**2*CT**4 -\
          225*B61*ST**2*CT**2 + 15*B61*ST**2 - 2310*B60*ST**3*CT**3+ 630*B60*ST**3*CT)/16

    RB62=(15*B66*ST**4*CT**2 + 15*B66*ST**4 + 15*B65*ST**3*CT**3 + 5*B65*ST**3*CT+\
          165*B64*ST**2*CT**4 - 50*B64*ST**2*CT**2 + 5*B64*ST**2 + 165*B63*ST*CT**5 - \
          150*B63*ST*CT**3 + 33*B63*ST*CT + 495*B62*CT**6 - 735*B62*CT**4 +\
          289*B62*CT**2 - 17*B62 - 495*B61*ST*CT**5 + 510*B61*ST*CT**3 - 95*B61*ST*CT+\
          3465*B60*ST**2*CT**4 - 1890*B60*ST**2*CT**2 + 105*B60*ST**2)/32

    RB61=(6*B66*ST**5*CT + 6*B65*ST**4*CT**2 - B65*ST**4 + 66*B64*ST**3*CT**3 -\
          26*B64*ST**3*CT + 66*B63*ST**2*CT**4 - 45*B63*ST**2*CT**2 + 3*B63*ST**2+
          198*B62*ST*CT**5 - 204*B62*ST*CT**3 + 38*B62*ST*CT + 198*B61*CT**6 -\
          285*B61*CT**4 + 100*B61*CT**2 - 5*B61 - 1386*B60*ST*CT**5 +\
          1260*B60*ST*CT**3 - 210*B60*ST*CT)/8

    RB60=(B66*ST**6 + B65*ST**5*CT + 11*B64*ST**4*CT**2 - B64*ST**4 +\
          11*B63*ST**3*CT**3 - 3*B63*ST**3*CT + 33*B62*ST**2*CT**4 - 18*B62*ST**2*CT**2 +\
          B62*ST**2 + 33*B61*ST*CT**5 - 30*B61*ST*CT**3 + 5*B61*ST*CT +\
          231*B60*CT**6 - 315*B60*CT**4 + 105*B60*CT**2 - 5*B60)/16
    # print(theta/(2*np.pi)*360)
    # print('',B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,'\n',RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66)
    return RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66,theta
alpha= 4/(45*35)
beta=2/(11*15*273)
gamma=8/(13**2*11**2*3**3*7)
#%%---------------code----------------------
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

#%%------------------------------testing
# num=146
# g_plot_1=[]
# for i in range(0,num):
#     B20 = test[0][i]
#     B21 = test[1][i]
#     B22 = test[2][i]
#     B40 = test[3][i]
#     B41 = test[4][i]
#     B42 = test[5][i]
#     B43 = test[6][i]
#     B44=  test[7][i]
#     B60 = test[8][i]
#     B61 = test[9][i]
#     B62 = test[10][i]
#     B63 = test[11][i]
#     B64 = test[12][i]
#     B65 = test[13][i]
#     B66 = test[14][i]
#     sol=kp.solver(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66)
#     CalcEnergy1=[sol[1]-sol[1][0]]
#     CalcEnergy1=CalcEnergy1[0].squeeze()
#     calcscattering1=np.array(sol[0]).real.squeeze() #percentage of relative intensity
#     Eigenvectors1=sol[2]
#     Jx=sol[3]
#     Jy=sol[4]
#     Jz=sol[5]
#     g=gtensor(Jx,Jy,Jz,Eigenvectors1[:,15],Eigenvectors1[:,14])
#     # def Ry(theta):
#     #     return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
#     # def Pesduospinrotation(theta):
#     #     A=g.dot(np.linalg.inv(Ry(theta)))
#     #     B=A-A.T
#     #     return B.item((2, 0))
#     # x=optimize.fsolve(np.vectorize(Pesduospinrotation), 0, xtol=1e-6)
#     # g2=np.dot(g,np.linalg.inv(Ry(x[0])))
#     # print('gxx,gzz,gyy are', np.linalg.eigvals(g2))

#     theta=symbols('theta')
#     A=Matrix([[cos(theta),0,sin(theta)],
#               [0,1,0],
#               [-sin(theta),0,cos(theta)]])

#     gprime=g*A.inv()
#     theta=solve(gprime[0,2]-gprime[2,0])[0]
#     A=Matrix([[cos(theta),0,sin(theta)],
#               [0,1,0],
#               [-sin(theta),0,cos(theta)]])

#     gprime=g*A.inv()
#     gprime=np.array(gprime,dtype=float)
#     print(gprime)
#     print('gxx=',np.linalg.eigvals(gprime)[0],'gyy=',np.linalg.eigvals(gprime)[2],'gzz=',np.linalg.eigvals(gprime)[1])
#     gxx=np.absolute(np.linalg.eigvals(gprime)[0])
#     gyy=np.absolute(np.linalg.eigvals(gprime)[2])
#     gzz=np.absolute(np.linalg.eigvals(gprime)[1])
#     g_plot_1.append([gxx,gyy,gzz])
    
# g_plot_1=np.array(g_plot_1)
# g_plot_1=g_plot_1[np.argsort(Bpar[:,-1])]
#%%
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
    RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66,theta=Rotate(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66)


    sol=kp.solver(RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66)
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
    B,Mag5=magnetization(RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66,5)
    temp=[]
    chi=[]
    # with alive_bar(25) as bar:
    #     for T in TX[:,0]:
    #         X=dmdh(RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66, T)
    #         temp.append(T)
    #         chi.append(X)
    #         bar.title('Calculating dM/dH')
    #         bar()

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

    # Bpar.append([RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66,chi2(CalcEnergy1, calcscattering1, Mag5, 1/chi, RealE, RealI, Realmag, TX[:,1])]) #TX[:,1] is already 1/X
    Bpar.append([RB20,RB21,RB22,RB40,RB41,RB42,RB43,RB44,RB60,RB61,RB62,RB63,RB64,RB65,RB66]) #TX[:,1] is already 1/X
   
    I_plot30.append(I_paper_30)
    I_plot120.append(I_paper_120)
    X_plot.append(chi)
    mag_plot.append(Mag5)
    g_plot.append([gxx,gyy,gzz])
    print(i)

#%%-----------------plot-----------------------------
Bpar=np.array(Bpar)
Bsort=Bpar[np.argsort(Bpar[:,-1])]

g_plot=np.array(g_plot)
g_plot=g_plot[np.argsort(Bpar[:,-1])]

I_plot30=np.array(I_plot30)
I_plot30=I_plot30[np.argsort(Bpar[:,-1])]

I_plot120=np.array(I_plot120)
I_plot120=I_plot120[np.argsort(Bpar[:,-1])]

X_plot=np.array(X_plot)
X_plot=X_plot[np.argsort(Bpar[:,-1])]

mag_plot=np.array(mag_plot)
mag_plot=mag_plot[np.argsort(Bpar[:,-1])]

g_plot=np.array(g_plot)
g_plot=g_plot[np.argsort(Bpar[:,-1])]


# B_test=np.zeros([146,17])
# for i in range(len(Bsort)):
#     # pdb.set_trace()
#     B_test[i,0],B_test[i,1],B_test[i,2],B_test[i,3],B_test[i,4],B_test[i,5],B_test[i,6],B_test[i,7],B_test[i,8],B_test[i,9],B_test[i,10],B_test[i,11],B_test[i,12],B_test[i,13],B_test[i,14],B_test[i,16]\
#         =Rotate(Bsort[i,0],Bsort[i,1],Bsort[i,2],Bsort[i,3],Bsort[i,4],Bsort[i,5],Bsort[i,6],Bsort[i,7],Bsort[i,8],Bsort[i,9],Bsort[i,10],Bsort[i,11],Bsort[i,12],Bsort[i,13],Bsort[i,14])
#     B_test[i,15]=Bsort[i,15] #last row is theta
# Bsort=B_test

order=np.argsort(Bpar[:,-1])
xlim=-1
ylim=10
casenum=100
datasize=40
linew=10
scattersize=1000
fs=70
ls=60
#--------------panel1-------------------------------
fig,ax=plt.subplots(5,3,figsize=(60,90),dpi=100)
# mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.major.pad'] = 10
mpl.rcParams['ytick.major.pad'] = 20
# fig.tight_layout()
ax[0,0].plot(Bsort[0:casenum,0],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[0,0].scatter(-0.0119,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[0,0].set_xlabel(r'$B_2^0$',fontsize=fs)
ax[0,0].set_ylabel(r'$\chi^2$',fontsize=fs)
ax[0,0].set_ylim(xlim,ylim)
ax[0,0].tick_params(width=6,direction='in',length=20,right=True,top=True,labelsize=ls)
ax[0,0].locator_params(axis='x', nbins=5)
ax[0,0].annotate("a)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[0,0].spines[axis].set_linewidth(5)

# ax[0,1].plot(Bsort[0:casenum,1],Bsort[0:casenum,15],'o',alpha=0.5)
ax[0,1].plot(B_test[0:casenum,16]/np.pi*180,B_test[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[0,1].scatter(4.31,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[0,1].set_xlabel('Angle (degree)',fontsize=fs)
ax[0,1].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[0,1].locator_params(axis='x', nbins=5)
for axis in ['top','bottom','left','right']:
    ax[0,1].spines[axis].set_linewidth(5)
# ax[0,1].set_ylabel('Chi^2')
ax[0,1].set_ylim(xlim,ylim)
ax[0,1].annotate("b)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)

ax[0,2].plot(Bsort[0:casenum,2],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[0,2].scatter(-0.1460,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[0,2].set_xlabel(r'$B^2_2$',fontsize=fs)
# ax[0,2].set_ylabel('Chi^2')
ax[0,2].set_ylim(xlim,ylim)
ax[0,2].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[0,2].locator_params(axis='x', nbins=5)
ax[0,2].annotate("c)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[0,2].spines[axis].set_linewidth(5)


ax[1,0].plot(Bsort[0:casenum,3],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[1,0].scatter(0.001183,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[1,0].set_xlabel(r'$B_4^0$',fontsize=fs)
ax[1,0].set_ylabel(r'$\chi^2$', fontsize=fs)
ax[1,0].set_ylim(xlim,ylim)
ax[1,0].ticklabel_format(style='sci',scilimits=(-3,3), useMathText=True)
ax[1,0].xaxis.get_offset_text().set_fontsize(50)
ax[1,0].tick_params(width=6,direction='in',length=20,right=True,top=True,labelsize=ls)
ax[1,0].locator_params(axis='x', nbins=5)
ax[1,0].annotate("d)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[1,0].spines[axis].set_linewidth(5)

ax[1,1].plot(Bsort[0:casenum,4],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[1,1].scatter(-0.001663051402711745,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[1,1].set_xlabel(r'$B_4^1$',fontsize=fs)
# ax[1,1].set_ylabel('Chi^2')
ax[1,1].set_ylim(xlim,ylim)
ax[1,1].set_xticks([-0.005,0,0.005])
ax[1,1].ticklabel_format(style='sci',scilimits=(-3,3), useMathText=True)
ax[1,1].xaxis.get_offset_text().set_fontsize(50)
ax[1,1].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[1,1].locator_params(axis='x', nbins=4)
ax[1,1].annotate("e)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[1,1].spines[axis].set_linewidth(5)

ax[1,2].plot(Bsort[0:casenum,5],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[1,2].scatter(0.0013329337244061224,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[1,2].set_xlabel(r'$B_4^2$',fontsize=fs)
# ax[1,2].set_ylabel('Chi^2')
ax[1,2].set_ylim(xlim,ylim)
ax[1,2].ticklabel_format(style='sci',scilimits=(-3,3), useMathText=True)
ax[1,2].xaxis.get_offset_text().set_fontsize(50)
ax[1,2].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[1,2].locator_params(axis='x', nbins=5)
ax[1,2].annotate("f)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[1,2].spines[axis].set_linewidth(5)

ax[2,0].plot(Bsort[0:casenum,6],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[2,0].scatter(0.011984848651199743,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[2,0].set_xlabel(r'$B_4^3$',fontsize=fs)
ax[2,0].set_ylabel(r'$\chi^2$', fontsize=fs)
ax[2,0].set_ylim(xlim,ylim)
ax[2,0].tick_params(width=6,direction='in',length=20,right=True,top=True,labelsize=ls)
ax[2,0].locator_params(axis='x', nbins=5)
ax[2,0].annotate("g)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[2,0].spines[axis].set_linewidth(5)


ax[2,1].plot(Bsort[0:casenum,7],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[2,1].scatter(-0.0014568497755702173,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[2,1].set_xlabel(r'$B_4^4$', fontsize=fs)
# ax[2,1].set_ylabel('Chi^2')
ax[2,1].set_ylim(xlim,ylim)
ax[2,1].ticklabel_format(style='sci',scilimits=(-3,3), useMathText=True)
ax[2,1].xaxis.get_offset_text().set_fontsize(50)
ax[2,1].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[2,1].set_xticks([-0.005,0,0.005])
ax[2,1].annotate("h)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[2,1].spines[axis].set_linewidth(5)

ax[2,2].plot(Bsort[0:casenum,8],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[2,2].scatter(4.118094062189712e-06,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[2,2].set_xlabel(r'$B_6^0$', fontsize=fs)
ax[2,2].ticklabel_format(style='sci',scilimits=(-5,5), useMathText=True)
ax[2,2].xaxis.get_offset_text().set_fontsize(50)
ax[2,2].set_ylim(xlim,ylim)
ax[2,2].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[2,2].locator_params(axis='x', nbins=5)
ax[2,2].annotate("i)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
# ax[2.2].xaxis.get_offset_text().set_fontsize(24)
for axis in ['top','bottom','left','right']:
    ax[2,2].spines[axis].set_linewidth(5)

for i in range(0,10):
    ax[0,0].scatter(Bsort[i,0],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2) #representative data
    ax[0,1].scatter(Bsort[i,16],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[0,2].scatter(Bsort[i,2],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[1,0].scatter(Bsort[i,3],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[1,1].scatter(Bsort[i,4],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[1,2].scatter(Bsort[i,5],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[2,0].scatter(Bsort[i,6],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[2,1].scatter(Bsort[i,7],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[2,2].scatter(Bsort[i,8],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[3,0].scatter(Bsort[i,9],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2) #representative data
    ax[3,1].scatter(Bsort[i,10],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[3,2].scatter(Bsort[i,11],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[4,0].scatter(Bsort[i,12],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[4,1].scatter(Bsort[i,13],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
    ax[4,2].scatter(Bsort[i,14],Bsort[i,15],edgecolors='green',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)

ax[3,0].plot(Bsort[0:casenum,9],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[3,0].scatter(-2.3470390323672103e-05,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[3,0].set_xlabel(r'$B_6^1$',fontsize=fs)
ax[3,0].set_ylabel(r'$\chi^2$',fontsize=ls)
ax[3,0].set_ylim(xlim,ylim)
ax[3,0].ticklabel_format(style='sci',scilimits=(-4,4), useMathText=True)
ax[3,0].xaxis.get_offset_text().set_fontsize(50)
ax[3,0].tick_params(width=6,direction='in',length=20,right=True,top=True,labelsize=ls)
ax[3,0].locator_params(axis='x', nbins=5)
ax[3,0].annotate("j)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[3,0].spines[axis].set_linewidth(5)

ax[3,1].plot(Bsort[0:casenum,10],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[3,1].scatter(-3.45386159169212e-05,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[3,1].set_xlabel(r'$B_6^2$',fontsize=fs)
# ax[0,1].set_ylabel('Chi^2')
ax[3,1].set_ylim(xlim,ylim)
ax[3,1].ticklabel_format(style='sci',scilimits=(0,0), useMathText=True)
ax[3,1].xaxis.get_offset_text().set_fontsize(50)
ax[3,1].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[3,1].locator_params(axis='x', nbins=3)
ax[3,1].annotate("k)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[3,1].spines[axis].set_linewidth(5)

ax[3,2].plot(Bsort[0:casenum,11],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[3,2].scatter(-0.000149446292280511,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[3,2].set_xlabel(r'$B_6^3$',fontsize=fs)
# ax[0,2].set_ylabel('Chi^2')
ax[3,2].set_ylim(xlim,ylim)
ax[3,2].ticklabel_format(style='sci',scilimits=(0,0), useMathText=True)
ax[3,2].xaxis.get_offset_text().set_fontsize(50)
ax[3,2].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[3,2].locator_params(axis='x', nbins=5)
ax[3,2].annotate("l)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[3,2].spines[axis].set_linewidth(5)

ax[4,0].plot(Bsort[0:casenum,12],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[4,0].scatter(3.944589619327707e-05,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[4,0].set_xlabel(r'$B_6^4$',fontsize=fs)
ax[4,0].set_ylabel(r'$\chi^2$', fontsize=fs)
ax[4,0].set_ylim(xlim,ylim)
ax[4,0].xaxis.get_offset_text().set_fontsize(50)
ax[4,0].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
ax[4,0].tick_params(width=6,direction='in',length=20,right=True,top=True,labelsize=ls)
ax[4,0].locator_params(axis='x', nbins=5)
ax[4,0].annotate("m)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[4,0].spines[axis].set_linewidth(5)

ax[4,1].plot(Bsort[0:casenum,13],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[4,1].scatter(8.57241727245872e-05,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[4,1].set_xlabel(r'$B_6^5$',fontsize=fs)
# ax[1,1].set_ylabel('Chi^2')
ax[4,1].set_ylim(xlim,ylim)
ax[4,1].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
ax[4,1].xaxis.get_offset_text().set_fontsize(50)
ax[4,1].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[4,1].locator_params(axis='x', nbins=5)
ax[4,1].annotate("n)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[4,1].spines[axis].set_linewidth(5)

ax[4,2].plot(Bsort[0:casenum,14],Bsort[0:casenum,15],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[4,2].scatter(0.0001856767263214382,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[4,2].set_xlabel('B66',fontsize=fs)
# ax[1,2].set_ylabel('Chi^2')
ax[4,2].set_ylim(xlim,ylim)
ax[4,2].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
ax[4,2].xaxis.get_offset_text().set_fontsize(50)
ax[4,2].tick_params(width=6,direction='in',length=20,right=True,top=True,labelleft=False,labelsize=ls)
ax[4,2].locator_params(axis='x', nbins=5)
ax[4,2].annotate("o)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=100)
for axis in ['top','bottom','left','right']:
    ax[4,2].spines[axis].set_linewidth(5)

#%%-----------
fig,ax=plt.subplots(2,3,figsize=(47,26),dpi=100)
# i=87
# mpl.rcParams['lines.markersize'] = 10
for i in range(120,130):
    ax[0,0].plot(E_x_30,I_plot30[i],alpha=0.5,color='lightcoral',linewidth=7)
# ax[0,0].plot(E_x_30,I_plot30[i],'.',alpha=0.5,color='green',markersize=30,zorder=2)
ax[0,0].set_xlabel('Energy Transfer (meV)',fontsize=40)
ax[0,0].set_ylabel('Intensity (arb.unit)', fontsize=40)
ax[0,0].plot(E_x_30,I_paper_30,color='blue',alpha=1,linewidth=3)
ax[0,0].tick_params(width=4,direction='in',length=10,right=True,top=True,labelsize=30)
ax[0,0].locator_params(axis='x', nbins=5)
for axis in ['top','bottom','left','right']:
    ax[0,0].spines[axis].set_linewidth(5)

for i in range(120,130):
    ax[0,1].plot(E_x_120,I_plot120[i],alpha=0.5,color='lightcoral',linewidth=5)
# ax[0,1].plot(E_x_120,I_plot120[i],'.',alpha=0.5,color='green',markersize=30)
ax[0,1].set_xlabel('Energy Transfer (meV)',fontsize=40)
ax[0,1].plot(E_x_120,I_paper_120,color='blue',alpha=1,linewidth=3)
ax[0,1].tick_params(width=4,direction='in',length=10,right=True,top=True,labelsize=30)
ax[0,1].locator_params(axis='x', nbins=5)
for axis in ['top','bottom','left','right']:
    ax[0,1].spines[axis].set_linewidth(5)

# for i in range(110,130):
#     ax[1,0].plot(temp,1/X_plot[i],alpha=0.5,color='grey',linewidth=5)
for i in range(0,10):
    ax[1,0].plot(temp,1/X_plot[i],alpha=0.5,color='lightcoral',linewidth=5)
ax[1,0].plot(temp,1/X_plot[i],'.',alpha=0.5,color='green',markersize=30)
ax[1,0].set_xlabel('Temperature (K)',fontsize=40)
ax[1,0].set_ylabel(r'$\dfrac{1}{\chi}(Oe/emu \ mol$-$R^{3+})$',fontsize=40)
ax[1,0].plot(temp,TX[:,1],color='blue',alpha=1,linewidth=3)
ax[1,0].tick_params(width=4,direction='in',length=10,right=True,top=True,labelleft=False,labelsize=30)
ax[1,0].locator_params(axis='x', nbins=5)
for axis in ['top','bottom','left','right']:
    ax[1,0].spines[axis].set_linewidth(5)

for i in range(0,10):
    ax[1,1].plot(B,mag_plot[i],color='lightcoral',linewidth=10,alpha=0.3,zorder=2)
for i in range(136,146):
    ax[1,1].plot(B,mag_plot[i],color='grey',linewidth=4,alpha=0.3,zorder=1)
# ax[1,1].plot(B,mag_plot[i],'.',color='green',markersize=30,alpha=0.5,zorder=2)
ax[1,1].set_xlabel(r'$\mu_0H(T)$',fontsize=40)
ax[1,1].set_ylabel(r'$M(\mu_B/R^{3+})$',fontsize=40)
ax[1,1].scatter(B,Realmag,edgecolors='blue',facecolors='white',alpha=1,zorder=2,s=150,linewidth=2)
ax[1,1].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=30)
for axis in ['top','bottom','left','right']:
    ax[1,1].spines[axis].set_linewidth(5)

for i in range(0,10):
    ax[0,2].scatter(Bsort[i,0],Bsort[i,15],edgecolors='green',facecolors='white',s=400,alpha=1,linewidth=5,zorder=2) #representative data
ax[0,2].plot(Bpar[:,0], Bpar[:,15],'o',color='red',zorder=1,alpha=0.7,markersize=20)
ax[0,2].scatter(-0.0119,0.07,edgecolors='blue',facecolors='white',alpha=1,zorder=2,s=400,linewidth=5)
ax[0,2].tick_params(width=4,direction='in',length=10,right=True,top=True,labelsize=30)
ax[0,2].tick_params(which='minor',direction='in',right=True,top=True,width=2,length=5)
ax[0,2].set_ylabel(r'$\chi^2$',fontsize=40)
ax[0,2].set_yscale('log')
ax[0,2].set_xlabel(r'$B_2^0$',fontsize=40)
for axis in ['top','bottom','left','right']:
    ax[0,2].spines[axis].set_linewidth(5)

for i in range(0,10):
    ax[1,2].scatter(B_test[i,16]/np.pi*180,B_test[i,15],edgecolors='green',facecolors='white',s=400,alpha=1,linewidth=5,zorder=2) #representative data
ax[1,2].plot(B_test[:,16]/np.pi*180,B_test[:,15],'o',color='red',markersize=20,zorder=1,alpha=0.7)
ax[1,2].scatter(4.31,0.07,edgecolors='blue',facecolors='white',s=400,alpha=1,linewidth=5,zorder=2)
ax[1,2].set_xlabel('Angle (degree)',fontsize=40)
ax[1,2].tick_params(width=4,direction='in',length=10,right=True,top=True,labelleft=True,labelsize=30)
ax[1,2].locator_params(axis='x', nbins=5)
ax[1,2].set_ylabel(r'$\chi^2$',fontsize=40)
ax[1,2].set_yscale('log')
for axis in ['top','bottom','left','right']:
    ax[1,2].spines[axis].set_linewidth(5)

#%%-----------Susceptibility--------------
for i in range(136,146):
    plt.plot(temp,1/X_plot[i],alpha=0.5,color='grey',zorder=1)
for i in range(0,10):
    plt.plot(temp,1/X_plot[i],alpha=0.5,color='lightcoral',linewidth=1,zorder=2)
# plt.plot(temp,1/X_plot[i],'.',alpha=0.5,color='green',markersize=30)
plt.xlabel('Temperature (K)')
plt.ylabel(r'$\dfrac{1}{\chi}(Oe/emu \ mol$-$R^{3+})$')
plt.plot(temp,TX[:,1],'.',color='blue',alpha=1,zorder=3)
plt.show()

#%%-------------------------g-tensors-------------------------------------------
# plt.plot(Bpar[:,0],g_plot_1[:,0],'.',label=r'$g_x$',markersize=8,color='red',alpha=0.5,zorder=1)
# plt.plot(Bpar[:,0],g_plot_1[:,1],'.',label=r'$g_y$',markersize=8,color='green', alpha=0.5,zorder=2)
# plt.plot(Bpar[:,0],g_plot_1[:,2],'.',label=r'$g_z$',markersize=8,color='blue', alpha=0.4,zorder=1)
for i in range(136,146):
    plt.plot(Bsort[i,0],np.max([g_plot_1[i,0],g_plot_1[i,1],g_plot_1[i,2]])/np.min([g_plot_1[i,0],g_plot_1[i,1],g_plot_1[i,2]]),'o',color='red',zorder=2)
for i in range(11,119):
    plt.plot(Bsort[i,0],np.max([g_plot_1[i,0],g_plot_1[i,1],g_plot_1[i,2]])/np.min([g_plot_1[i,0],g_plot_1[i,1],g_plot_1[i,2]]),'o',color='grey',zorder=1,alpha=0.5)
for i in range(0,11):
    plt.plot(Bsort[i,0],np.max([g_plot_1[i,0],g_plot_1[i,1],g_plot_1[i,2]])/np.min([g_plot_1[i,0],g_plot_1[i,1],g_plot_1[i,2]]),'o',color='blue',zorder=2)
# plt.plot(Bpar[0:10,0],g_plot_1[0:10,0],'x',markersize=7,color='black',alpha=1,zorder=1)
# plt.plot(Bpar[0:10,0],g_plot_1[0:10,1],'x',markersize=7,color='black', alpha=1,zorder=2)
# plt.plot(Bpar[0:10,0],g_plot_1[0:10,2],'x',markersize=7,color='black', alpha=1,zorder=1)

# plt.plot(Bpar[120:130,0],g_plot_1[120:130,0],'x',markersize=7,color='cyan',alpha=1,zorder=1)
# plt.plot(Bpar[120:130,0],g_plot_1[120:130,1],'x',markersize=7,color='cyan', alpha=1,zorder=2)
# plt.plot(Bpar[120:130,0],g_plot_1[120:130,2],'x',markersize=7,color='cyan', alpha=1,zorder=1)

# plt.scatter(-0.0119,14.05,edgecolors='magenta',facecolors='white',alpha=1,zorder=2,s=20,linewidth=2)
# plt.scatter(-0.0119,0.18 ,edgecolors='magenta',facecolors='white',alpha=1,zorder=2,s=20,linewidth=2)
# plt.scatter(-0.0119,0.5  ,edgecolors='magenta',facecolors='white',alpha=1,zorder=2,s=20,linewidth=2)
# plt.plot(-0.0119,14.05,'x',color='magenta',alpha=1,zorder=3)
# plt.plot(-0.0119,0.18 ,'x',color='magenta',alpha=1,zorder=3)
# plt.plot(-0.0119,0.5  ,'x',color='magenta',alpha=1,zorder=3)
# plt.plot(-0.0119,78,'x',color='magenta')
plt.ylim(1,1000)
plt.yscale('log')
plt.xlabel(r'$B_2^0$',fontsize=15)
plt.legend()
plt.tick_params(width=3,direction='in',length=4,right=True,top=True,labelsize=10)
plt.tick_params(which='minor',direction='in',right=True,top=True,width=2,length=2)
plt.title('third excited state g-tensors')
# plt.savefig('goverlay.png',dpi=500)
#%%--------------panel3-------------------------------
fig,ax=plt.subplots(2,2,figsize=(15,15),dpi=100)
# fig.tight_layout()
casenum=10
for i in order[0:casenum]:
    ax[0,0].plot(B,mag_plot[i],color='lightcoral',linewidth=4,alpha=0.3,zorder=2)
# for i in order[120:130]:
#     ax[0,0].plot(B,mag_plot[i],color='green',linewidth=4,alpha=0.3,zorder=1)
ax[0,0].set_xlabel(r'$\mu_0H(T)$',fontsize=20)
ax[0,0].set_ylabel(r'$M(\mu_B/R^{3+})$',fontsize=20)
ax[0,0].plot(B,Realmag,color='blue',linewidth=3,zorder=2)
ax[0,0].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
for axis in ['top','bottom','left','right']:
    ax[0,0].spines[axis].set_linewidth(2)

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

# fig.savefig('D:\CEF\Python\Er3Mg2Sb3O14\Properties.png', format='png',dpi=400)
#%%------------------------------------
g_plot=np.array(g_plot)
fig = plt.figure(figsize=(12, 12),dpi=600)
ax = fig.add_subplot(projection='3d')

for i in order[0:casenum]:
    if g_plot[i,0]>7:
                ax.scatter(g_plot[i,0],g_plot[i,1],g_plot[i,2],color='red',s=100,marker='o',alpha=0.5)
                print(i)
    if g_plot[i,1]>7: continue
                # ax.scatter(g_plot[i,0],g_plot[i,1],g_plot[i,2],color='blue',s=100,marker='o',alpha=0.5)
    if g_plot[i,2]>7:continue
                # ax.scatter(g_plot[i,0],g_plot[i,1],g_plot[i,2],color='green',s=100,marker='o',alpha=0.5)
ax.scatter(14.05,0.18,0.50, color='yellow', s=200,marker='o',alpha=1)
ax.set_xlabel(r'$g_x$',fontsize=20)
ax.set_ylabel(r'$g_y$',fontsize=20)
ax.set_zlabel(r'$g_z$',fontsize=20)

#%%--------------------testing plot ellipsoid---------
casenum=20
fig = plt.figure(figsize=(20,20),dpi=400) # Square figure
mpl.rcParams['xtick.major.pad'] = 10
mpl.rcParams['ytick.major.pad'] = 20
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
rx,ry,rz=(0,0,0)
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))
for i in range(casenum):
# for i in (15,):
    coefs = (g_plot[i,0], g_plot[i,1], g_plot[i,2])  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    # Radii corresponding to the coefficients:
    rx, ry, rz = coefs
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):

    # Plot:
    if rx>10:
        ax.plot_surface(x, y, z,  rstride=10, cstride=10, cmap='autumn',alpha=0.5,antialiased=True)
        print(i)
    elif ry>10:
        ax.plot_surface(x, y, z,  rstride=10, cstride=10, cmap='autumn',alpha=0.5,antialiased=True)
    elif rz>10:
        ax.plot_surface(x, y, z,  rstride=10, cstride=10, cmap='autumn',alpha=0.5,antialiased=True)
    # Adjustment of the axes, so that they all have the same span:

# ax.plot_surface(14.05*np.outer(np.cos(u), np.sin(v)), \
#                 0.18 * np.outer(np.sin(u), np.sin(v)), \
#                 0.50 * np.outer(np.ones_like(u), np.cos(v)),\
#                     rstride=10, cstride=10, color='blue', alpha=1)
max_radius = max(rx, ry, rz)
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
# Remove gray panes and axis grid
ax.xaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.fill = False
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.fill = False
ax.zaxis.pane.set_edgecolor('white')
ax.grid(False)
ax.set_xlabel(r'$g_x$',fontsize=5,labelpad=10)
ax.set_ylabel(r'$g_y$',fontsize=5,labelpad=10)
ax.set_zlabel(r'$g_z$',fontsize=5,labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=5)
plt.savefig('filename.png',  transparent=True)
plt.show()
#%%----------New Plot---------------------------------

plt.plot(Bpar[:,0],g_plot[:,0],'^',label=r'$g_x$',markersize=5)
plt.plot(Bpar[:,0],g_plot[:,1],'1',label=r'$g_y$',markersize=9)
plt.plot(Bpar[:,0],g_plot[:,2],'.',label=r'$g_z$',markersize=7)
plt.tick_params(width=2,direction='in',length=5,right=True,top=True)
plt.xlabel(r'$B_2^0$')
plt.legend()
plt.show()

