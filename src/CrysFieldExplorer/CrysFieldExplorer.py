import numpy as np
from numpy import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import CrysFieldExplorer.Operators as op
import sympy as sym
from sympy import symbols, solve,cos,sin,Matrix, Inverse
from scipy import optimize
from scipy.special import wofz
from scipy import integrate
import scipy.linalg as LA
from alive_progress import alive_bar
from time import sleep
import CrysFieldExplorer.Visulization as Visulization

class CrysFieldExplorer(op.Stevens_Operator,op.Quantum_Operator):
    '''
    Double inheriting from Stevens_Operator and Quantum_Operator from Operators to have 
    access of all quantum operators.
    
    Input Description:
        
    Magnetic_ion[Str]: Acceptable input format exampel: "Er3+". 
    
    Stevens_idx[List]: Stevens Operator index in the form of [n,m], n=subscript,  m=superscript.  
                Input example [[2,0],[2,1],[2,2],[4,0],[4,1]]
        
    alpha,beta,gamma[float]:radial portion of the CEF wave function.
    
    Parameter[Hash Map]:Crystal field parameters.
    
    Temperature[float]: in unit of Kelvin
    
    Field[list]: Vectorized external magnetic field [Bx,By,Bz].

        '''
    
    def __init__(self,Magnetic_ion,Stevens_idx,alpha,beta,gamma,Parameter,temperature,field):
        self.Stevens_idx=Stevens_idx
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.Parameter=Parameter
        self.T=temperature
        self.field=field
        super().__init__(Magnetic_ion)
    
    def Hamiltonian(self):
        O=super().Stevens_hash(self.Stevens_idx) 
        H=0
        j=0
        for i in O:
            if i[0] == '2':
                H+=self.alpha*self.Parameter[j]*O[i]
            elif i[0] == '4':            
                H+=self.beta*self.Parameter[j]*O[i]
            elif i[0] == '6':            
                H+=self.gamma*self.Parameter[j]*O[i]   
            j+=1
        eigenvalues, eigenvectors=np.linalg.eigh(H)
        eigenvalues=np.sort(eigenvalues)
        eigenvectors=eigenvectors[:,np.argsort(eigenvalues)]
        return eigenvalues, eigenvectors, H
    
    @classmethod
    def Hamiltonian_scale(cls,Magnetic_ion, Stevens_idx, alpha, beta, gamma, Parameter,scale, temperature, field):
        newobj=cls(Magnetic_ion, Stevens_idx, alpha, beta, gamma, Parameter, temperature, field)
        O=newobj.Stevens_hash(Stevens_idx)
        #fitting is an art. Scale O matrix such that parameters are mostly in the same magnitude
        j=0
        for i in O:
            O[i]=O[i]*scale[j]
            j+=1
            print(j,i)
        H=0
        j=0
        for i in O:
            if i[0] == '2':
                H+=alpha*Parameter[j]*O[i]
            elif i[0] == '4':            
                H+=beta*Parameter[j]*O[i]
            elif i[0] == '6':            
                H+=gamma*Parameter[j]*O[i]   
            j+=1
        eigenvalues, eigenvectors=np.linalg.eigh(H)
        eigenvalues=np.sort(eigenvalues)
        eigenvectors=eigenvectors[:,np.argsort(eigenvalues)]
        return eigenvalues, eigenvectors, H
    
    # def magH(jx, jy, jz, Bx,By,Bz):
    def magnetic_Hamiltonian(self,Bx,By,Bz):
         jx=super().Jx()
         jy=super().Jy()
         jz=super().Jz()
         # Bx=self.field[0]
         # By=self.field[1]
         # Bz=self.field[2]
         S=self.S;L=self.L;J=self.J
         gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
         Na=6.0221409e23
         muBT=5.7883818012e-2
         magH = -gj*muBT*(Bx*jx+By*jy+Bz*jz)
         return magH
     
    def magsovler(self,Bx,By,Bz):
         _,_,H=self.Hamiltonian()
         magH=self.magnetic_Hamiltonian(Bx,By,Bz)
         Eigenvalues, Eigenvectors = (np.linalg.eigh(H+magH))
         Energy=Eigenvalues-Eigenvalues[0]
         return Energy, Eigenvectors, magH
    
    def scattering(self,i,j):
        #k=8.6173324*10^(-2)
        jx=super().Jx()
        jy=super().Jy()
        jz=super().Jz()
        x=np.dot(np.dot(i,jx),j)
        y=np.dot(np.dot(i,jy),j)
        z=np.dot(np.dot(i,jz),j)
        S=np.dot(np.conj(x),x)+np.dot(np.conj(y),y)+np.dot(np.conj(z),z)
        return S
    
    def tempfac(self,E,Ei):
        beta = 1/(8.61733e-2*self.T)  # Boltzmann constant is in meV/K
        Z=sum([np.exp(-beta*en) for en in E])
        prefac=np.exp(-beta*Ei)/Z
        return prefac
    
    def Intensity(self,gs): #gs:ground state
        ev,ef,_=self.Hamiltonian()
        ef=ef
        ev=ev-ev[0]
        s=dict()
        for i in range(gs,int(2*self.J+1)):
            s.update({i:self.tempfac(ev, ev[gs])*self.scattering(ef[:,gs].H,ef[:,i])}) #s:transition probability from ground state gs to excited state. This include the probability from gs to gs
        return s
    
    
    def Neutron_Intensity(self, N,gs,Kramer):
        '''N is the baseline of intensity we would like to compare'''
        if Kramer == True:
            s_degen=dict()
            for i in range(0,len(self.Intensity(gs)),2):
                s_degen.update({i:(self.Intensity(gs)[i]+self.Intensity(gs)[i+1])})
            total=s_degen[N]
            S=dict()
            for i in s_degen:
                S.update({i:(s_degen[i]/total).item()})
            return S
        if Kramer == False:
            s_degen=self.Intensity(gs)
            total=s_degen[N]
            S=dict()
            for i in s_degen:
                S.update({i:(s_degen[i]/total).item()})
            return S
        
    def Intensity_fast(self,gs): #gs:ground state
        ev,ef,_=self.Hamiltonian()
        ef=ef
        ev=ev-ev[0]
        s=[]
        for i in range(gs,int(2*self.J+1)):
            s.append(self.tempfac(ev, ev[gs])*self.scattering(ef[:,gs].H,ef[:,i])) #s:transition probability from ground state gs to excited state. This include the probability from gs to gs
        return np.array(s).squeeze()
    
    def Neutron_Intensity_fast(self, N,gs): #only use if it's Kramer's ion in this version
        '''N is the baseline of intensity we would like to compare'''
        s_degen=[]
        for i in range(0,len(self.Intensity_fast(gs)),2):
            s_degen.append((self.Intensity_fast(gs)[i]+self.Intensity_fast(gs)[i+1]).real)
            # print((self.Intensity_fast(gs)[i]+self.Intensity_fast(gs)[i+1]).item())
        total=s_degen[N]
        S=np.zeros(len(s_degen))
        for i in range(len(s_degen)):
            S[i]=s_degen[i]/total
        return S
    
    def Intensity_fast_mag(self,gs): #gs:ground state
        ev,ef,_=self.magsovler()
        ef=ef
        ev=ev-ev[0]
        s=[]
        for i in range(gs,int(2*self.J+1)):
            s.append(self.tempfac(ev, ev[gs])*self.scattering(ef[:,gs].H,ef[:,i])) #s:transition probability from ground state gs to excited state. This include the probability from gs to gs
        return np.array(s).squeeze()
    
    def Neutron_Intensity_fast_mag(self, N,gs): #Be careful here as the energy levels now splits
        '''N is the baseline of intensity we would like to compare'''
        s_degen=[]
        for i in range(0,len(self.Intensity_fast(gs)),2):
            s_degen.append((self.Intensity_fast(gs)[i]+self.Intensity_fast(gs)[i+1]).real)
            # print((self.Intensity_fast(gs)[i]+self.Intensity_fast(gs)[i+1]).item())
        total=s_degen[N]
        S=np.zeros(len(s_degen))
        for i in range(len(s_degen)):
            S[i]=s_degen[i]/total
    
class Utilities(CrysFieldExplorer):
    '''This class contains functions to cacluate loss, construct neutron spectrum, magnetization, susceptibility
       and other common functions needed in CrysFieldExplore in both optimization and visulization module'''
       
    def __init__(self,Magnetic_ion,Stevens_idx,alpha,beta,gamma,Parameter,temperature,field):
        super().__init__(Magnetic_ion,Stevens_idx,alpha,beta,gamma,Parameter,temperature,field)
        
    @staticmethod
    def lorentzian(x,Area,width,pos):
        pi=np.pi
        Y=(Area/pi)*(width/2)/((x-pos)**2+(width/2)**2)
        return Y
    
    @staticmethod
    def chi(Obs,Exp):
        summation=0
        for i in range(len(Obs)):
            if Exp[i]<2e-5:continue
            else:
                summation+=(Obs[i]-Exp[i])**2/Exp[i]
        return summation
    
    
    def test(self,Bx,By,Bz):
        return super().magsovler(Bx, By, Bz)
    
    def dmdh(self,T):
        '''Another way to calculate susceptibility by taking differential of dm/dh. 
           It's good practice to calculate susceptibility this way and compare it with the
           Van-Vleck method'''
        k=100
        Bx=k/10000
        By=k/10000
        Bz=k/10000
        S=self.S;L=self.L;J=self.J
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
            mag1=super().magsovler(Bx*X, By*Y, Bz*Y)
            E1=mag1[0]
            magvec1=mag1[1]
            jx=super().Jx()
            jy=super().Jy()
            jz=super().Jz()
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
            MB=Na*muB*M/Bx/sampling/10000
        return MB
    
    def susceptibility_dmdh(self, T_ini, T_final, step):
        temp=[]
        chi=[]
        with alive_bar((T_final-T_ini)//step) as bar:
            for T in range(T_ini,T_final,step):
                X=self.dmdh(T)
                temp.append(T)
                chi.append(X)
                bar.title('Calculating dM/dH')
                bar()
        temp=np.array(temp)
        chi=np.array(chi)
    
    def magnetization(self,T):
        '''Calculation of powder averaged magnetization from 0.1 to 7.1T with 0.5T step size at temperature T. 
           This range covers most in-house lab measurement capability'''
        
        B=[]
        Magnetization=[]
        with alive_bar(14,bar='bubbles') as bar:
            for k in range(1000,71000,5000):
                # i=70000
                Bx=k/10000
                By=k/10000
                Bz=k/10000
                S=self.S;L=self.L;J=self.J
                gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
                muBT = 5.7883818012e-2
                k_B = 8.6173303e-2
                Na=6.0221409e23
                muB=9.274009994e-21
                dim=int(2*self.J+1)
                #T=10 #temperature where magnetization is measured
                M=0
                sampling=500 #Monte Carlo Sampling size
                for m in range(0,sampling):
                    # print(m)
                    X=np.random.normal(0,1,3)[0]
                    Y=np.random.normal(0,1,3)[1]
                    Z=np.random.normal(0,1,3)[2]
                    norm=np.sqrt(X**2+Y**2+Z**2)
                    X=X/norm
                    Y=Y/norm
                    Z=Z/norm
                    # mag1=kp.magsovler1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,Bx*X,By*Y,Bz*Z)
                    mag1=super().magsovler(Bx*X, By*Y, Bz*Y)
                    E1=mag1[0]
                    # print(E1)
                    magvec1=mag1[1]
                    jx=super().Jx()
                    jy=super().Jy()
                    jz=super().Jz()
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
                Magnetization.append(M/sampling)
                B.append(k/10000)
                # print(k)
                sleep(0.03)
                bar()
                bar.title('Magnetization')
        return B, Magnetization
        
    
    def susceptibility_VanVleck(self):
          ev,ef,H=super().Hamiltonian()  
          Eigenvectors=ef
          E=ev-ev[0]
          S=self.S;L=self.L;J=self.J
          gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
          Jx=super().Jx()
          Jy=super().Jy()
          Jz=super().Jz()
          dim=int(2*self.J+1)
          Na=6.0221409e23
          muB=9.274009994e-21
          kb=1.38064852e-16
          C=(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
          Z=0
          T=np.linspace(1, 300,150)
          # T=TX[0][::50]
         
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
      
    def gtensor(self):
          Jx=super().Jx()
          Jy=super().Jy()
          Jz=super().Jz()
          S=self.S;L=self.L;J=self.J
          gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
          ev,ef,H=super().Hamiltonian()
          minus=ef[:,1]
          plus=ef[:,0]
          Sx=[[(plus.H*Jx*plus).item(), (minus.H*Jx*plus).item()],
              [(plus.H*Jx*minus).item(),(minus.H*Jx*minus).item()],
              ]

          Sy=[[(plus.H*Jy*plus).item(), (minus.H*Jy*plus).item()],
              [(plus.H*Jy*minus).item(),(minus.H*Jy*minus).item()],
              ]
          Sz=[[(plus.H*Jz*plus).item(), (minus.H*Jz*plus).item()],
              [(plus.H*Jz*minus).item(),(minus.H*Jz*minus).item()],
              ]
          g=[[Sx[1][0].real, Sx[1][0].imag, Sx[0][0].real],
             [Sy[1][0].real, Sy[1][0].imag, Sy[0][0].real],
             [Sz[1][0].real, Sz[1][0].imag, Sz[0][0].real],
              ]
          g=np.dot((2*gj),g)
          return g
    def gprime(self,axis):
          g=self.gtensor()
          theta=symbols('theta')
          if axis == 'x':
              A=Matrix([[1, 0, 0],
                        [0,cos(theta),-sin(theta)],
                        [0,sin(theta), cos(theta)]])
          elif axis == 'y':
              A=Matrix([[cos(theta),0,sin(theta)],
                       [0,1,0],
                       [-sin(theta),0,cos(theta)]])
          elif axis == 'z':
              A=Matrix([[cos(theta), -sin(theta), 0],
                       [sin(theta),cos(theta),0],
                       [0, 0, 1]])

          gprime=g*A.inv()
          theta=solve(gprime[0,2]-gprime[2,0])[0]
          A=Matrix([[cos(theta),0,sin(theta)],
                    [0,1,0],
                    [-sin(theta),0,cos(theta)]])
          gprime=g*A.inv()
          gprime=np.array(gprime,dtype=float)
          gxx=np.absolute(np.linalg.eigvals(gprime)[0])
          gyy=np.absolute(np.linalg.eigvals(gprime)[2])
          gzz=np.absolute(np.linalg.eigvals(gprime)[1])
          return gxx,gyy,gzz, gprime
#%% test
if __name__ == "__main__":
    alpha=0.01*10.0*4/(45*35)
    beta=0.01*100.0*2/(11*15*273)
    gamma=0.01*10.0*8/(13**2*11**2*3**3*7)
    Stevens_idx=[[2,0],[2,1],[2,2],[4,0],[4,1],[4,2],[4,3],[4,4],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]
    scale      = [ 1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1  ,  1 ]
    test=pd.read_csv(f'C:/Users/qmc/OneDrive/ONRL/Data/CEF/Python/Eradam/Eradam_MPI_Newfit_goodsolution.csv',header=None)
    Parameter=dict()
    temperature=5
    field=[0,0,0]
    j=0

    Para=np.zeros(15)
    for i in range(15):
        Para[i]=test[i][0]
        
    Para[2]=10*Para[2] #22 -2
    Para[4]=0.1*Para[4] #41 - 4
    Para[6]=10*Para[6]  #43 -6
    Para[9]=0.1*Para[9] #61 -9
    Para[11]=10*Para[11] #63 -11
    Para[13]=10*Para[13] #65 -13
    Para[14]=10*Para[14] #66 -14
    
    CEF=CrysFieldExplorer('Er3+',Stevens_idx,alpha,beta,gamma,Para,temperature,field)
    ev,ef,H=CEF.Hamiltonian()
    #classmethod here doesn't associate Hamiltonian_scale to any instance so we doesn't need to create another instance to change scale
    # ev,ef,H=CrysFieldExplorer.Hamiltonian_scale('Er3+', Stevens_idx, alpha, beta, gamma, Para, scale, temperature, field)
    print(np.round(ev-ev[0],3))
    
    # Intensity=CEF.Neutron_Intensity(2, 0, True)
    Intensity=CEF.Intensity_fast(0)
    # print(Intensity)
    
    uti=Utilities('Er3+', Stevens_idx, alpha, beta, gamma, Para, temperature, field)
    ev1,_,_=uti.test(1, 3, 1)
    print(np.round(ev1-ev1[0],3))
    #plotting
    plot=Visulization.vis(15,10)
    plot.susceptibility(uti.susceptibility_VanVleck())

    mag=uti.magnetization(5)
    plot.magnetization(mag)
    
    plot.neutron_spectrum(ev-ev[0], Intensity, 0.5)