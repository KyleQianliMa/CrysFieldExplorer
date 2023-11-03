import numpy as np
from numpy import sqrt
import pandas as pd
import pdb
import Operators as op

class CrysFieldExplorer(op.Stevens_Operator,op.Quantum_Operator):
    '''alpha,beta,gamma:conversion to get the crystal field parameters between -1000 to 1000.
       Double inheriting from Stevens_Operator and Quantum_Operator from Operators to have access of quantum operators.
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
        O['22']=10*O['22']
        O['41']=0.1*O['41']
        O['43']=10*O['43']
        O['61']=0.1*O['61']
        O['63']=10*O['63']
        O['65']=10*O['65']
        O['66']=10*O['66']
        
        H=0
        for i in O:
            if i[0] == '2':
                H+=self.alpha*self.Parameter[i]*O[i]
            elif i[0] == '4':            
                H+=self.beta*self.Parameter[i]*O[i]
            elif i[0] == '6':            
                H+=self.gamma*self.Parameter[i]*O[i]                
        eigenvalues, eigenvectors=np.linalg.eig(H)
        eigenvalues=np.sort(eigenvalues)
        eigenvectors=eigenvectors[:,np.argsort(eigenvalues)]
        return np.matrix(eigenvalues), np.matrix(eigenvectors), H
    
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
        ev=ev-ev[0]
        s=dict()
        for i in range(gs,int(2*super().J+1)):
            s.update({i:self.tempfac(ev, ev[gs])*self.scattering(gs,i)}) #s:transition probability from ground state gs to excited state. This include the probability from gs to gs
        return s
    
            
#%%
if __name__ == "__main__":
    alpha=0.01*10.0*4/(45*35)
    beta=0.01*100.0*2/(11*15*273)
    gamma=0.01*10.0*8/(13**2*11**2*3**3*7)
    Stevens_idx=[[2,0],[2,1],[2,2],[4,0],[4,1],[4,2],[4,3],[4,4],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]
    test=pd.read_csv(f'C:/Users/qmc/OneDrive/ONRL/Data/CEF/Python/Eradam/Eradam_MPI_Newfit_goodsolution.csv',header=None)
    
    Parameter=dict()
    j=0
    for i in Stevens_idx:
        Parameter.update({f'{i[0]}{i[1]}':test[j][0]})
        j+=1
    
    CEF=CrysFieldExplorer('Er3+',Stevens_idx,alpha,beta,gamma,Parameter)
    ev,ef,H=CEF.Hamiltonian()
    print(np.round(ev-ev[0],3))
# B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66
#%%


#vertical rolls are eigenvectors
def solver(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,T):
    a=H1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66)
    HH=a[2]
    Energy=a[0]
    E=Energy-Energy[0]
    Eigenvectors=np.matrix(a[1])
    
    scattering1=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,2],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,3],jx,jy,jz)
    scattering1=scattering1*tempfac(T,E,E[0])
    
    scattering2=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,4],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,5],jx,jy,jz)
    scattering2=scattering2*tempfac(T,E,E[0])
    
    scattering3=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,6],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,7],jx,jy,jz)
    scattering3=scattering3*tempfac(T,E,E[0])
    
    scattering4=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,8],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,9],jx,jy,jz)
    scattering4=scattering4*tempfac(T,E,E[0])
    
    scattering5=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,10],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,11],jx,jy,jz)
    scattering5=scattering5*tempfac(T,E,E[0])
    
    scattering6=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,12],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,13],jx,jy,jz)
    scattering6=scattering6*tempfac(T,E,E[0])
    
    scattering7=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,14],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,15],jx,jy,jz)
    scattering7=scattering7*tempfac(T,E,E[0])
    
    scattering12=scattering(np.transpose(np.conj(Eigenvectors[:,2])),Eigenvectors[:,4],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,2])),Eigenvectors[:,5],jx,jy,jz)
    scattering12=scattering12*tempfac(T, E, E[2]) #5.29-1.75=3.54meV, major fit observable!!!!!!!!!!!!!!!!!!!!!
    
    scattering13=scattering(np.transpose(np.conj(Eigenvectors[:,2])),Eigenvectors[:,6],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,2])),Eigenvectors[:,7],jx,jy,jz)
    scattering13=scattering12*tempfac(T, E, E[2]) #7.10-1.75=5.35meV, add to scattering 2 intensity
    
    scattering14=scattering(np.transpose(np.conj(Eigenvectors[:,2])),Eigenvectors[:,8],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,2])),Eigenvectors[:,9],jx,jy,jz)
    scattering14=scattering14*tempfac(T, E, E[2]) #13.73-1.75=11.98meV, don't fit but check intensity
    
    scattering23=scattering(np.transpose(np.conj(Eigenvectors[:,4])),Eigenvectors[:,6],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,4])),Eigenvectors[:,7],jx,jy,jz)
    scattering23=scattering23*tempfac(T, E, E[4]) #7.10-5.29=1.81meV, not going to fit
    
    scattering24=scattering(np.transpose(np.conj(Eigenvectors[:,4])),Eigenvectors[:,8],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,4])),Eigenvectors[:,9],jx,jy,jz)
    scattering24=scattering12*tempfac(T, E, E[4]) #13.73-5.29=8.44meV, not goingto fit
    
    
    
    N=scattering1
    s1=scattering1/N
    s2=(scattering2+scattering13)/N
    s3=scattering3/N
    s4=scattering4/N
    s5=scattering5/N
    s6=scattering6/N
    s7=scattering7/N
    
    s12=scattering12/N #3.54meV
    s14=scattering14/N #11.98meV
    # s23=scattering23/N #1.81meV
    s24=scattering24/N #8.44meV
    #return s1, s2, s3, s4, Energy, Eigenvectors,
    return np.array([s1, s2, s3,s4,s5,s6,s7, s12, s14, s24]).squeeze(), Energy, Eigenvectors,jx,jy,jz,HH


def magH(jx, jy, jz, Bx,By,Bz):
     S=3/2;L=6;J=15/2
     gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
     Na=6.0221409e23
     muBT=5.7883818012e-2
     muB=9.274009994e-21
     kb=1.38064852e-16
     magH = -gj*muBT*(Bx*jx+By*jy+Bz*jz)
     return magH

def magsovler1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66,Bx,By,Bz):
    a=H1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66)
    b=magH(jx, jy, jz, Bx, By, Bz)
    Eigenvalues, Eigenvectors = (np.linalg.eigh(a[2]+b))
    Energy=Eigenvalues-Eigenvalues[0]
    return Energy, Eigenvectors, jx,jy,jz,
