import numpy as np
from numpy import sqrt
import pdb

class Operator():
 def Jplus(self,J):
    Jplus = np.zeros((int(2*J+1),int(2*J+1)))
    for i in range(int(2*J)):
        Jplus[i+1,i] = sqrt((J-(-J+i))*(J+(-J+i)+1))
    return Jplus

 def Jminus(self,J):
    Jminus =np.zeros((int(2*J+1),int(2*J+1)))
    for i in range(int(2*J)):
        Jminus[i,i+1] = sqrt((J+(-(J-1)+i))*(J-(-(J-1)+i)+1))
    return Jminus

 def Jz(self,J):
    Jz = np.zeros((int(2*J+1),int(2*J+1)))
    for i in range(int(2*J+1)):
     Jz[i,i] = (-J+i)
    return Jz

def H1(D20,D21,D22,D40,D41,D42,D43,D44,D60,D61,D62,D63,D64,D65,D66):
 H=D20*O20+D21*O21+D22*O22+D40*O40+D41*O41+D42*O42+D43*O43+D44*O44+D60*O60+D61*O61+D62*O62+D63*O63+D64*O64+D65*O65+D66*O66
 Eigenvalues, Eigenvectors = (np.linalg.eigh(H))
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

#start calculating Nd3+ matrces
Nd=Operator()
J=15/2
jplus=np.matrix(Nd.Jplus(J))
jminus=np.matrix(Nd.Jminus(J))
jz=np.matrix(Nd.Jz(J))
jx=0.5*(jplus+jminus)
jy=(jminus-jplus)/(2*1j)
jsquare=np.dot(jx,np.transpose(np.conj(jx)))+np.dot(jy,np.transpose(np.conj(jy)))+np.dot(jz,np.transpose(np.conj(jz)))
jplussquare = jplus*jplus
jminussquare = jminus*jminus
O20 = 3*(jz**2) - jsquare
O21 = 0.25*(jz*(jplus+jminus)+(jplus+jminus)*jz)
O22 = 0.5*(jplus**2 + jminus**2)
O40 = 35*(jz**4) - 30*(jsquare*(jz**2)) + 25*(jz**2) -6*(jsquare) + 3*(jsquare**2)
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
def solver(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66):
    a=H1(B20,B21,B22,B40,B41,B42,B43,B44,B60,B61,B62,B63,B64,B65,B66)
    HH=a[2]
    Energy=a[0]
    Eigenvectors=np.matrix(a[1])
    scattering1=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,2],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,3],jx,jy,jz)
    scattering2=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,4],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,5],jx,jy,jz)
    scattering3=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,6],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,7],jx,jy,jz)
    scattering4=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,8],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,9],jx,jy,jz)
    scattering5=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,10],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,11],jx,jy,jz)
    scattering6=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,12],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,13],jx,jy,jz)
    scattering7=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,14],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,15],jx,jy,jz)
    N=scattering1
    s1=scattering1/N
    s2=scattering2/N
    s3=scattering3/N
    s4=scattering4/N
    s5=scattering5/N
    s6=scattering6/N
    s7=scattering7/N

    #return s1, s2, s3, s4, Energy, Eigenvectors,
    return np.array([s1, s2, s3,s4,s5,s6,s7]).squeeze(), Energy, Eigenvectors,jx,jy,jz,HH


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
