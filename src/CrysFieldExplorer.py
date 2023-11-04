import numpy as np
from numpy import sqrt
import pandas as pd
import pdb
import Operators as op

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
        self.B=field
        super().__init__(Magnetic_ion)

    def Hamiltonian(self):
        O=super().Stevens_hash(Stevens_idx)
        
        H=0
        for i in O:
            if i[0] == '2':
                H+=self.alpha*self.Parameter[i]*O[i]
            elif i[0] == '4':            
                H+=self.beta*self.Parameter[i]*O[i]
            elif i[0] == '6':            
                H+=self.gamma*self.Parameter[i]*O[i]                
        eigenvalues, eigenvectors=np.linalg.eigh(H)
        eigenvalues=np.sort(eigenvalues)
        eigenvectors=eigenvectors[:,np.argsort(eigenvalues)]
        return eigenvalues, eigenvectors, H
    
    # def magH(jx, jy, jz, Bx,By,Bz):
    def magnetic_Hamiltonian(self):
         jx=super().Jx()
         jy=super().Jy()
         jz=super().Jz()
         Bx=self.field[0]
         By=self.field[1]
         Bz=self.field[2]
         S=self.S;L=self.L;J=self.J
         gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
         Na=6.0221409e23
         muBT=5.7883818012e-2
         magH = -gj*muBT*(Bx*jx+By*jy+Bz*jz)
         return magH
     
    def magsovler(self):
         _,_,H=self.Hamiltonian()
         magH=self.magnetic_Hamiltonian()
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
            total=s_degen[N]
            S=dict()
            for i in s_degen:
                S.update({i:(s_degen[i]/total).item()})
            return S
        

class Utilities(CrysFieldExplorer):
    def __init__(self,Magnetic_ion,Stevens_idx,alpha,beta,gamma,Parameter,temperature,field):
        super().__init__(Magnetic_ion)
        
    # @staticmethod
    # def lorentzian(x,Area,width,x0):
    #     pi=np.pi
    #     Y=(Area/pi)*(width/2)/((x-x0)**2+(width/2)**2)
    #     return Y
    
    # @staticmethod
    # def chi(Obs,Exp):
    #     summation=0
    #     for i in range(len(Obs)):
    #         if Exp[i]<2e-5:continue
    #         else:
    #             summation+=(Obs[i]-Exp[i])**2/Exp[i]
    #     return summation
    
    # @staticmethod
    # def test(a):
    #     return print(a)
    
    # @staticmethod
    # def susceptibility_VanVleck(Eigenvectors, Jx, Jy, Jz, E):
    #       # S=3/2;L=6;J=15/2
    #       # gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
    #       Na=6.0221409e23
    #       muB=9.274009994e-21
    #       kb=1.38064852e-16
    #       C=(gj**2)*(Na)*(muB)**2/kb #X*Na* mu_b^2*X/kb in cgs unit
    #       Z=0
    #       # T=np.linspace(1, 300,150)
    #       T=TX[0][::50]
         
    #       for n in range(0,dim):
    #           Z=Z+np.exp(-E[n]/T)
    #       X=0
    #       for n in range(0,dim):
    #           for m in range(0,dim):
    #               if np.abs(E[m]-E[n])<1e-5: X=X+(np.absolute(Eigenvectors[:,n].H*Jx*Eigenvectors[:,m]).item())**2*(np.exp(-E[n]/T))/T
    #               else: X = X+ 2*(np.absolute(Eigenvectors[:,m].H*Jx*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T))/(E[m]-E[n])
    #       for n in range(0,dim):
    #           for m in range(0,dim):
    #               if  np.abs(E[m]-E[n])<1e-5: X=X+(np.absolute(Eigenvectors[:,n].H*Jy*Eigenvectors[:,m]).item())**2*(np.exp(-E[n]/T))/T
    #               else: X = X+ 2*(np.absolute(Eigenvectors[:,m].H*Jy*Eigenvectors[:,n]).item())**2*(np.exp(-E[n]/T))/(E[m]-E[n])
    #       for n in range(0,dim):
    #           for m in range(0,dim):
    #               if  np.abs(E[m]-E[n])<1e-5: X=X+(np.absolute(Eigenvectors[:,n].H*Jz*Eigenvectors[:,m]).item())**2*(np.exp(-E[n]/T))/T
    #               else: X = X+ 2*((np.absolute(Eigenvectors[:,m].H*Jz*Eigenvectors[:,n]).item())**2)*(np.exp(-E[n]/T))/(E[m]-E[n])
    #       X=C*X/(3*Z)
    #       return T,X
#%% test
if __name__ == "__main__":
    alpha=0.01*10.0*4/(45*35)
    beta=0.01*100.0*2/(11*15*273)
    gamma=0.01*10.0*8/(13**2*11**2*3**3*7)
    Stevens_idx=[[2,0],[2,1],[2,2],[4,0],[4,1],[4,2],[4,3],[4,4],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]
    test=pd.read_csv(f'C:/Users/qmc/OneDrive/ONRL/Data/CEF/Python/Eradam/Eradam_MPI_Newfit_goodsolution.csv',header=None)
    Parameter=dict()
    temp=5
    field=0
    j=0
    for i in Stevens_idx:
        Parameter.update({f'{i[0]}{i[1]}':test[j][0]})
        j+=1
    
    Parameter['22']=10*Parameter['22']
    Parameter['41']=0.1*Parameter['41']
    Parameter['43']=10*Parameter['43']
    Parameter['61']=0.1*Parameter['61']
    Parameter['63']=10*Parameter['63']
    Parameter['65']=10*Parameter['65']
    Parameter['66']=10*Parameter['66']
    
    CEF=CrysFieldExplorer('Er3+',Stevens_idx,alpha,beta,gamma,Parameter,temp,field)
    ev,ef,H=CEF.Hamiltonian()
    print(np.round(ev-ev[0],3))
    Intensity=CEF.Neutron_Intensity(2, 0, True)
    
    uti=Utilities('Er3+', Stevens_idx, alpha, beta, gamma, Parameter, temp, field)
    # uti.test()
    
# B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66
