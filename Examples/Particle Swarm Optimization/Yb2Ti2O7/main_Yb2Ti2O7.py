import Operators_Yb2Ti2O7 as kp
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

final_result=pd.read_csv('Yb2Ti2O7_result_case6_guannan.csv',header=None)
Realmag=np.loadtxt("Magnetization.txt")
Realchi=np.loadtxt("Chi.txt")
RealX=np.loadtxt("X.txt")
RealE=[76.80033405,82.11933405,116.24306695]
RealI=[0.284,1,0.19]
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

def x(Eigenvectors, Jx, Jy, Jz, E):
    # S=1/2;L=3;J=7/2;
     # gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
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
#--------------------Magnetization-------------------------------
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

S=1/2;L=3;J=7/2;
dim=int(2*J+1)
gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))

Bpar=[]
I_plot=[]
X_plot=[]
mag_plot=[]
g_plot=[]



# for i in range (0,130):
for i in (106,):
# i=105
    B20  = final_result[0][i]#1.135
    B40  = final_result[1][i]#-0.0615
    B43  = final_result[2][i]#0.315
    B60  = final_result[3][i]#0.0011
    B63  = final_result[4][i]#0.037
    B66  = final_result[5][i]#0.005



    # B20  = 1.135
    # B40  = -0.0615
    # B43  = 0.315
    # B60  = 0.0011
    # B63  = 0.037
    # B66  = 0.005

    # alpha=2/64
    # beta=-2/(77*15)
    # gamma=4/(13*33*63)

    # B20_scale  = B20/alpha
    # B40_scale  = B40/beta
    # B43_scale  = B43/beta
    # B60_scale  = B60/gamma
    # B63_scale  = B63/gamma
    # B66_scale  = B66/gamma

    sol=kp.solver(B20,B40,B43,B60,B63,B66)
    CalcEnergy=[sol[1][2].round(2),sol[1][4], sol[1][6]]-sol[1][0]
    calcscattering=np.array(sol[0]).round(3).real.squeeze() #percentage of relative intensity
    Eigenvectors=sol[2]
    E_x=np.linspace(0, 150,300)
    a1=3.7
    I_paper=lorentzian(E_x,0.284, a1, 76.8)+lorentzian(E_x,1, a1, 82.12)+lorentzian(E_x,0.19, a1, 116.24)

    I_fitted=lorentzian(E_x,calcscattering[0], a1, CalcEnergy[0])+lorentzian(E_x,calcscattering[1], a1, CalcEnergy[1])\
      +lorentzian(E_x,calcscattering[2], a1, CalcEnergy[2])

    Jx=sol[3]
    Jy=sol[4]
    Jz=sol[5]
    Jplus=sol[6]
    E=sol[1]/0.0862

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
        # --------------------Plots---------------------------------------
    T,X=x(Eigenvectors,sol[3], sol[4], sol[5], (sol[1]-sol[1][0])/0.0862)
    B,Mag=magnetization(B20, B40, B43, B60, B63, B66,10)

    # fig,ax=plt.subplots(2,2)
    # fig.suptitle('B20={} gtensor ={},{},{}'.format(B20,gxx.round(3),gyy.round(3),gzz.round(3)))
    # ax[0,0].plot(T,1/X,temp,1/chi)
    # ax[0,0].plot(T,1/RealX,color='red')
    # ax[0,0].set_xlim(0,300)
    # ax[0,0].set_ylim(0, 200)

    # ax[0,1].plot(E_x,I_paper,'.',color='red')
    # ax[0,1].plot(E_x,I_fitted,'-')

    # ax[1,0].plot(B,Mag)
    # ax[1,0].plot(B,Realmag,color='red')

    # gfactor=['gx','gy','gz']
    # gvalue=[np.absolute(gxx),np.absolute(gyy),np.absolute(gzz)]
    # ax[1,1].bar(gfactor,gvalue,color='green')
    # ax[1,1].bar(gfactor,[3.753,3.753,1.985],color='red',alpha=1)

    Bpar.append([B20,B40,B43,B60,B63,B66,chi2(CalcEnergy, calcscattering, Mag, 1/X, RealE, RealI, Realmag, 1/RealX)])
    I_plot.append(I_fitted)
    X_plot.append(X)
    mag_plot.append(Mag)
    g_plot.append([gx,gz/gx])

#%%----------------------------------change indent above--------------------------------------------
Bpar=np.array(Bpar)
Bsort=Bpar[np.argsort(Bpar[:,-1])]
order=np.argsort(Bpar[:,-1])

casenum=50
ymax=17
ymin=-3
example=1,4
datasize=10
linew=3
scattersize=100
c='green','black'

fig,ax=plt.subplots(3,2,figsize=(13,15),dpi=400)
plt.subplots_adjust(wspace=0.02)
# fig.tight_layout()
ax[0,0].plot(Bsort[0:casenum,0],Bsort[0:casenum,6],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[0,0].scatter(1.135,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[0,0].set_xlabel('B20',fontsize=15)
ax[0,0].set_ylabel(r'$\chi^2$',fontsize=20)
ax[0,0].set_ylim(ymin,ymax)
ax[0,0].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
ax[0,0].locator_params(axis='x', nbins=5)
ax[0,0].annotate("a)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=30)
for axis in ['top','bottom','left','right']:
    ax[0,0].spines[axis].set_linewidth(2)

ax[0,1].plot(Bsort[0:casenum,1],Bsort[0:casenum,6],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[0,1].scatter(-0.0615,0,edgecolors='blue',facecolors='white',s=scattersize,alpha=1,linewidth=linew,zorder=2)
ax[0,1].set_xlabel('B40',fontsize=15)
ax[0,1].set_ylim(ymin,ymax)
ax[0,1].locator_params(axis='x', nbins=5)
ax[0,1].tick_params(width=4,direction='in',length=5,right=True,top=True,labelleft=False,labelsize=15)
ax[0,1].annotate("b)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=30)
for axis in ['top','bottom','left','right']:
    ax[0,1].spines[axis].set_linewidth(2)


ax[1,0].plot(Bsort[0:casenum,2],Bsort[0:casenum,6],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[1,0].scatter(0.315,0,edgecolors='blue',facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
ax[1,0].set_xlabel('B43',fontsize=15)
ax[1,0].set_ylabel(r'$\chi^2$',fontsize=20)
ax[1,0].set_ylim(ymin,ymax)
ax[1,0].locator_params(axis='x', nbins=6)
ax[1,0].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
ax[1,0].annotate("c)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=30)
for axis in ['top','bottom','left','right']:
    ax[1,0].spines[axis].set_linewidth(2)


ax[1,1].plot(Bsort[0:casenum,3],Bsort[0:casenum,6],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[1,1].scatter(0.0011,0,edgecolors='blue',facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
ax[1,1].set_xlabel('B60',fontsize=15)
# ax[1,1].ticklabel_format(style='sci',scilimits=(0,0))
ax[1,1].set_ylim(ymin,ymax)
ax[1,1].set_xlim(-0.0015,0.0033)
ax[1,1].locator_params(axis='x', nbins=6)
ax[1,1].tick_params(width=4,direction='in',length=5,right=True,top=True,labelleft=False,labelsize=15)
ax[1,1].annotate("d)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=30)
for axis in ['top','bottom','left','right']:
    ax[1,1].spines[axis].set_linewidth(2)



ax[2,0].plot(Bsort[0:casenum,4],Bsort[0:casenum,6],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[2,0].scatter(0.037,0,edgecolors='blue',facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
ax[2,0].set_xlabel('B63',fontsize=15)
ax[2,0].set_ylabel(r'$\chi^2$',fontsize=20)
ax[2,0].set_ylim(ymin,ymax)
ax[2,0].locator_params(axis='x', nbins=6)
ax[2,0].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
ax[2,0].annotate("e)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=30)
for axis in ['top','bottom','left','right']:
    ax[2,0].spines[axis].set_linewidth(2)

ax[2,1].plot(Bsort[0:casenum,5],Bsort[0:casenum,6],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[2,1].scatter(0.005,0,edgecolors='blue',facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
ax[2,1].set_xlabel('B66',fontsize=15)
ax[2,1].set_xlim(-0.045,0.045)
ax[2,1].set_ylim(ymin,ymax)
ax[2,1].locator_params(axis='x', nbins=6)
ax[2,1].tick_params(width=4,direction='in',length=5,right=True,top=True,labelleft=False,labelsize=15)
ax[2,1].annotate("f)", xy=(0.03, 0.85), xycoords="axes fraction",fontsize=30)
for axis in ['top','bottom','left','right']:
    ax[2,1].spines[axis].set_linewidth(2)

for i,color in zip(example,c):
    ax[0,0].scatter(Bsort[i,0],Bsort[i,6],edgecolors=color,facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2) #representative data
    ax[0,1].scatter(Bsort[i,1],Bsort[i,6],edgecolors=color,facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
    ax[1,0].scatter(Bsort[i,2],Bsort[i,6],edgecolors=color,facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
    ax[1,1].scatter(Bsort[i,3],Bsort[i,6],edgecolors=color,facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
    ax[2,0].scatter(Bsort[i,4],Bsort[i,6],edgecolors=color,facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
    ax[2,1].scatter(Bsort[i,5],Bsort[i,6],edgecolors=color,facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2)
#%%-------------Figure 2---------------------------------
fig,ax=plt.subplots(2,2,figsize=(15,10),dpi=400)
casenum=10
# fig.tight_layout()
# for i in order[0:casenum]:
    # ax[0,0].plot(E_x,I_plot[i],color='lightcoral',linewidth=4,alpha=0.3,zorder=1)
ax[0,0].set_xlabel('Energy Transfer (meV)',fontsize=15)
ax[0,0].set_ylabel('Intensity (arb.unit)',fontsize=15)
ax[0,0].plot(E_x,I_paper,color='blue',alpha=1,linewidth=3,zorder=2,label='True Solution')
ax[0,0].plot(E_x,I_plot[23],color='black',linewidth=4,alpha=0.8,zorder=3,label='Calculated Solution a)')
ax[0,0].plot(E_x,I_plot[106],color='green',linewidth=4,alpha=0.8,zorder=3,label='Calculated Solution b)')
ax[0,0].set_xlim(54,140)
ax[0,0].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
ax[0,0].legend()
for axis in ['top','bottom','left','right']:
    ax[0,0].spines[axis].set_linewidth(2)


for i in order[0:casenum]:
    ax[0,1].plot(T,1/X_plot[i],color='lightcoral',linewidth=4,alpha=0.3,zorder=1)
ax[0,1].set_ylabel(r'$\dfrac{1}{\chi}(Oe/emu \ mol$-$R^{3+})$',fontsize=15)
ax[0,1].set_xlabel('Temperature (K)',fontsize=15)
ax[0,1].plot(T,1/RealX,color='blue',linewidth=3,zorder=2)
ax[0,1].plot(T,1/X_plot[23],color='black',linewidth=4,alpha=0.5,zorder=3)
ax[0,1].plot(T,1/X_plot[106],color='green',linewidth=4,alpha=0.8,zorder=3)
ax[0,1].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
for axis in ['top','bottom','left','right']:
    ax[0,1].spines[axis].set_linewidth(2)

for i in order[0:casenum]:
    ax[1,0].plot(B,mag_plot[i],color='lightcoral',linewidth=4,alpha=0.3,zorder=1)
ax[1,0].set_xlabel(r'$\mu_0H(T)$',fontsize=15)
ax[1,0].set_ylabel(r'$M(\mu_B/R^{3+})$',fontsize=15)
ax[1,0].plot(B,Realmag,color='blue',linewidth=3,zorder=2)
# ax[1,0].plot(B,mag_plot[23],color='green',linewidth=4,alpha=0.8,zorder=3)
# ax[1,0].plot(B,mag_plot[106],color='black',linewidth=4,alpha=0.8,zorder=3)
ax[1,0].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
for axis in ['top','bottom','left','right']:
    ax[1,0].spines[axis].set_linewidth(2)


for i in order[0:casenum]:
    ax[1,1].plot(g_plot[i][0],g_plot[i][1],'o',color='lightcoral',markersize=datasize,zorder=1,alpha=0.7)
ax[1,1].scatter(3.753,0.529,edgecolors='blue',facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2,label='True Solution')
ax[1,1].scatter(2.43217531,1.85542285,edgecolors='black',facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=2,label='Calculated Solution a)')
ax[1,1].scatter(3.72969562,0.55484067,edgecolors='green',facecolors='white',alpha=1,s=scattersize,linewidth=linew,zorder=1, label='Calculated Solution a)')
ax[1,1].set_xlabel(r'$g_x$',fontsize=15)
ax[1,1].set_ylabel(r'$\dfrac{g_z}{g_{x(y)}}$',fontsize=15)
ax[1,1].set_ylim(-0.4,2.4)
ax[1,1].legend()
ax[1,1].tick_params(width=4,direction='in',length=5,right=True,top=True,labelsize=15)
for axis in ['top','bottom','left','right']:
    ax[1,1].spines[axis].set_linewidth(2)


#-------------------plot representative data----------------------------------
# for i in (106,):
    # ax[0,0].plot(E_x,I_plot[i],color='lightcoral',linewidth=4,alpha=0.3,zorder=1)

    # ax[0,1].plot(T,1/X_plot[i],color='yellow',linewidth=4,alpha=0.3,zorder=1)

    # ax[1,0].plot(B,mag_plot[i],color='yellow',linewidth=4,alpha=0.3,zorder=1)

    # ax[1,1].plot(g_plot[i][0],g_plot[i][1],'.',alpha=0.8,color='yellow',markersize=20)
