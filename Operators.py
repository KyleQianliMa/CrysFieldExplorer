import numpy as np
from numpy import sqrt
import pdb

print('-'*55 + '\n'+
    '|                CrysFieldExplorer 1.0.1              |\n' +
    '|   A program for fast optimization of CEF Parameters |\n' +
    '|   -Developed by Kyle Ma                             |\n' +
    '|   Please cite  J. Appl. Cryst. (2023). 56, 1229-124 |\n' +
    '|    https://doi.org/10.1107/S1600576723005897        |\n' + '-'*55 +'\n')



REion= {}   # The [S, L, J] quantum numbers for rare earth elements
#Rare earth (Thanks to PyCrystalField for summarizing the data)
REion['Ce3+'] = [0.5, 3., 2.5]
REion['Pr3+'] = [1., 5., 4.]
REion['Nd3+'] = [1.5, 6., 4.5]
REion['Pm3+'] = [2., 6., 4.]
REion['Sm3+'] = [2.5, 5, 2.5]
REion['Eu3+'] = [3, 3, 0]
REion['Gd3+'] = [7/2, 0, 7/2]
REion['Tb3+'] = [3., 3., 6.]
REion['Dy3+'] = [2.5, 5., 7.5]
REion['Ho3+'] = [2., 6., 8.]
REion['Er3+'] = [1.5, 6., 7.5]
REion['Tm3+'] = [1., 5., 6.]
REion['Yb3+'] = [0.5, 3., 3.5]

class Quantum_Operator():
    '''The total angular momentum operator'''
    def __init__(self, Magnetic_ion):
        self.S=REion[Magnetic_ion][0]
        self.L=REion[Magnetic_ion][1]
        self.J=REion[Magnetic_ion][2]
        
    def Jplus(self):
        Jplus = np.zeros((int(2*self.J+1),int(2*self.J+1)))
        for i in range(int(2*self.J)):
            Jplus[i+1,i] = sqrt((self.J-(-self.J+i))*(self.J+(-self.J+i)+1))
        return Jplus
       
    def Jminus(self):
        Jminus =np.zeros((int(2*self.J+1),int(2*self.J+1)))
        for i in range(int(2*self.J)):
            Jminus[i,i+1] = sqrt((self.J+(-(self.J-1)+i))*(self.J-(-(self.J-1)+i)+1))
        return Jminus   
       
    def Jz(self):
        Jz = np.zeros((int(2*self.J+1),int(2*self.J+1)))
        for i in range(int(2*self.J+1)):
         Jz[i,i] = (-self.J+i)
        return Jz
       
    def Jx(self):
        jx=0.5*(self.Jplus()+self.Jminus())       
        return jx
    
    def Jy(self):
        jy=(self.Jminus()-self.Jplus())/(2*1j)
        return jy
       
    def Jsquare(self):
        jsquare=np.dot(self.Jx(),np.transpose(np.conj(self.Jx())))+np.dot(self.Jy(),np.transpose(np.conj(self.Jy())))+np.dot(self.Jz(),np.transpose(np.conj(self.Jz())))
        return jsquare
        
    def Jplussquare(self):
        jplussquare = self.Jplus()*self.Jplus()
        return jplussquare
       
    def Jminussquare(self):
        jminussquare = self.Jminus()*self.Jminus()
        return jminussquare
    
    def Matrix(self):
        quantum_matrix={'jx':self.Jx(),'jy':self.Jy(),'jz':self.Jz(),'jplus':self.Jplus(),'jminus':self.Jminus(),'jsquare':self.Jsquare()} 
        return quantum_matrix

class Stevens_Operator():
    '''Calculate the Stevens Operator from the quantum operator'''
    
    def __init__(self,matrix):
        self.matrix=matrix

    def Stevens(self,n,m):
       jx=self.matrix['jx']
       jy=self.matrix['jy']
       jz=self.matrix['jz'] 
       jplus=self.matrix['jplus']
       jminus=self.matrix['jminus']
       jsquare=self.matrix['jsquare']
       
       O20 = 3.0*(jz**2) - jsquare
       O21 = 0.25*(jz*(jplus+jminus)+(jplus+jminus)*jz)
       O22 = 0.5*(jplus**2 + jminus**2)
       O40 = 35.0*(jz**4) - 30.0*(jsquare*(jz**2)) + 25.0*(jz**2) -6.0*(jsquare) + 3.0*(jsquare**2)
       O41 = 0.25*((jplus+jminus)*(7*jz**3-(3*jsquare*jz+jz))+(7*jz**3-(3*jsquare*jz+jz))*(jplus+jminus))
       O42 = 0.25*((jplus**2 + jminus**2)*(7*jz**2-jsquare) -(jplus**2 + jminus**2)*5+
                   (7*jz**2-jsquare)*(jplus**2 + jminus**2)-5*(jplus**2 + jminus**2))
       O43 = 0.25*(jz*(jplus**3+jminus**3)+(jplus**3+jminus**3)*jz)
       O44 = 0.5*(jplus**4 + jminus**4)
       O60 = 231*(jz**6) - 315*(jsquare*(jz**4)) + 735*(jz**4) + 105*((jsquare**2)*(jz**2)) - 525*(jsquare*(jz**2)) + 294*(jz**2) - 5*(jsquare**3) + 40*(jsquare**2) - 60*(jsquare)
       O61 =0.25*((jplus+jminus)*(33*jz**5-(30*jsquare*jz**3-15*jz**3)+(5*jsquare**2*jz-10*jsquare*jz+12*jz))+(33*jz**5-(30*jsquare*jz**3-15*jz**3)+(5*jsquare**2*jz-10*jsquare*jz+12*jz))*(jplus+jminus))
       O62 = 0.25*((jplus**2 + jminus**2)*(33*jz**4 -(18*jsquare*jz**2+123*jz**2) +jsquare**2 +10*jsquare) +((jplus**2 + jminus**2)*102) +
                   (33*jz**4 -(18*jsquare*jz**2+123*jz**2) +jsquare**2 +10*jsquare)*(jplus**2 + jminus**2)+102*(jplus**2 + jminus**2))
       O63 = 0.25*((11*jz**3-3*jsquare*jz-59*jz)*(jplus**3+jminus**3) + (jplus**3+jminus**3)*(11*jz**3 - 3 * jsquare*jz-59*jz))
       O64 = 0.25*((jplus**4 + jminus**4)*(11*jz**2 -jsquare) -(jplus**4 + jminus**4)*38 + (11*jz**2 -jsquare)*(jplus**4 + jminus**4)-38*(jplus**4 + jminus**4))
       O65 = 0.25*((jplus**5+jminus**5)*jz+jz*(jplus**5+jminus**5))
       O66 = 0.5*((jplus**6+jminus**6))
       O2n2 = (-0.5j)*(jplus**2-jminus**2)
       O4n2 = (-0.25j)*((jplus**2 - jminus**2)*(7*jz**2-jsquare) -(jplus**2 - jminus**2)*5+
                   (7*jz**2-jsquare)*(jplus**2 - jminus**2)-5*(jplus**2 - jminus**2))
       O4n3 = (-0.25j) *((jplus**3 - jminus**3)*jz + jz*(jplus**3 - jminus**3))
       O4n4 = (-0.5j)*(jplus**4 - jminus**4)
       O6n2 = (-0.25j)*((jplus**2 - jminus**2)*(33*jz**4 -(18*jsquare*jz**2+123*jz**2) +jsquare**2 +10*jsquare) +((jplus**2 - jminus**2)*102) +
                   (33*jz**4 -(18*jsquare*jz**2+123*jz**2) +jsquare**2 +10*jsquare)*(jplus**2 - jminus**2)+102*(jplus**2 - jminus**2))
       O6n3 =  -0.25j*((11*jz**3-3*jsquare*jz-59*jz)*(jplus**3-jminus**3) + (jplus**3-jminus**3)*(11*jz**3 - 3 * jsquare*jz-59*jz))
       O6n4 = -0.25j*((jplus**4 - jminus**4)*(11*jz**2 -jsquare) -(jplus**4 - jminus**4)*38 + (11*jz**2 -jsquare)*(jplus**4 - jminus**4)-38*(jplus**4 - jminus**4))
       O6n6 = (-0.5j)*(jplus**6-jminus**6)
       
       if   [n,m] == [2,0]:
           matrix= O20
       elif [n,m] == [2,1]:
           matrix= O21
       elif [n,m] == [2,2]:
           matrix= O22
       elif [n,m] == [4,0]:
           matrix= O40
       elif [n,m] == [4,1]:
           matrix= O41
       elif [n,m] == [4,2]:
           matrix= O42
       elif [n,m] == [4,3]:
           matrix= O43
       elif [n,m] == [4,4]:
            matrix= O44
       elif [n,m] == [6,0]:
           matrix= O60   
       elif [n,m] == [6,1]:
           matrix= O61
       elif [n,m] == [6,2]:
           matrix= O62
       elif [n,m] == [6,3]:
           matrix= O63
       elif [n,m] == [6,4]:
           matrix= O64
       elif [n,m] == [6,5]:
           matrix= O65
       elif [n,m] == [6,6]:
           matrix= O66
       elif [n,m] == [2,-2]:
           matrix= O2n2
       elif [n,m] == [4,-2]:
           matrix= O4n2
       elif [n,m] == [4,-3]:
           matrix= O4n3
       elif [n,m] == [4,-4]:
           matrix= O4n4
       elif [n,m] == [6,-2]:
           matrix= O6n2
       elif [n,m] == [6,-3]:
           matrix= O6n3
       elif [n,m] == [6,-4]:
           matrix= O6n4
       elif [n,m] == [6,-6]:
           matrix= O6n6
       return {f'{n}{m}':matrix}



class CrysFieldExplorer():
    '''alpha,beta,gamma:conversion to get the crystal field parameters between -1000 to 1000'''
    
    def __init__(self,Stevens_Matrix,alpha,beta,gamma):
        self.Stevens_Matrix=Stevens_Matrix
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
    
    def Parameter(self):
        return dict.fromkeys(self.Stevens_Matrix.keys(),[])
        
    def Hamiltonian(self):
        H=0
        for i in self.Stevens_Matrix:
            if i[0] == '2':
                H+=self.alpha*self.Parameter()[i]*self.Stevens_Matrix[i]
            elif i[0] == '4':
                H+=self.beta*self.Parameter()[i]*self.Stevens_Matrix[i]
            elif i[0] == '6':
                H+=self.gamma*self.Parameter()[i]*self.Stevens_Matrix[i]
            
        eigenvalues, eigenvectors=np.linalg.eig(H)
        return eigenvalues, eigenvectors, H

if __name__ == "__main__":  
    Quantum_matrix=Quantum_Operator('Nd3+').Matrix() 
    obj=Stevens_Operator(Quantum_matrix)
    
    Steven=[[2,0],[2,1],[2,2],[4,0],[4,1],[4,2],[4,3],[4,4],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]
    O=dict()
    for i in Steven:
        O.update(obj.Stevens(i[0],i[1]))
#D20,D21,D22,D40,D41,D42,D43,D44,D60,D61,D62,D63,D64,D65,D66
#%%
def H1(D20,D21,D22,D40,D41,D42,D43,D44,D60,D61,D62,D63,D64,D65,D66):
    alpha=0.01*10.0*4/(45*35)
    beta=0.01*100.0*2/(11*15*273)
    gamma=0.01*10.0*8/(13**2*11**2*3**3*7)
    H=alpha*(D20*O20 + D21*O21 + 10*D22*O22) + beta*(D40*O40 + 0.1*D41*O41 + D42*O42 + 10*D43*O43 + D44*O44) + gamma * (D60*O60 + 0.1*D61*O61 + D62*O62 + 10*D63*O63 + D64*O64 + 10*D65*O65 + 10*D66*O66)
    Eigenvalues, Eigenvectors = (np.linalg.eig(H))
    # Energy=Eigenvalues-Eigenvalues[0]
    # pdb.set_trace()
    return Eigenvalues, Eigenvectors, H



def scattering(i,j,jx,jy,jz):
    #k=8.6173324*10^(-2)
    x=np.dot(np.dot(i,jx),j)
    y=np.dot(np.dot(i,jy),j)
    z=np.dot(np.dot(i,jz),j)
    S=np.dot(np.conj(x),x)+np.dot(np.conj(y),y)+np.dot(np.conj(z),z)
    return S

def tempfac(T,E,Ei):
    beta = 1/(8.61733e-2*T)  # Boltzmann constant is in meV/K
    Z=sum([np.exp(-beta*en) for en in E])
    prefac=np.exp(-beta*Ei)/Z
    return prefac

#Stevens operator conversion to [-1000,1000]range
k20=1#-281/0.912
k40=1#-344/(1.25e-2)
k60=1#-88/(2.09e-4)
k43=1
#k44=93/(-2.82e-2)
#k64=104/(-2.77e-3)
k63=1
k66=1

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
