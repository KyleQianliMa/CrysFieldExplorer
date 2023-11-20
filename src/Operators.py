import numpy as np
from numpy import sqrt
import pandas as pd
import pdb

print('-'*55 + '\n'+
    '|                CrysFieldExplorer 1.0.0              |\n' +
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
    '''The total angular momentum operator Constructed from Magnetic Ion
       Input Description:
       
       Magnetic_ion[Str]: Magnetic ion. Example: "Er3+"
     '''
    
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

class Stevens_Operator(Quantum_Operator):
    
    '''Child class of Quantum_Operator.
    
       Calculate the Stevens Operator from the quantum operator by inheriting Jx,Jy,Jz,Jplus,Jminus,Jsquare etc from Parent Class.
       Using hashmap data struct for readability.
       
       Input Description:
          
       Magnetic_ion[Str]: Magnetic ion. Example: "Er3+"
    '''
    
    def __init__(self,Magnetic_ion):
        super().__init__(Magnetic_ion)

    def Stevens(self,n,m):
       jx=np.matrix(super().Matrix()['jx'])
       jy=np.matrix(super().Matrix()['jy'])
       jz=np.matrix(super().Matrix()['jz']) 
       jplus=np.matrix(super().Matrix()['jplus'])
       jminus=np.matrix(super().Matrix()['jminus'])
       jsquare=np.matrix(super().Matrix()['jsquare'])
       
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
   

    def Stevens_hash(self,Stevens_idx):
        O=dict()
        for i in Stevens_idx:
            O.update(self.Stevens(i[0],i[1]))
        return O

