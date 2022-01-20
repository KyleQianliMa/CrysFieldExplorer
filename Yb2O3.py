import numpy as np
from numpy import sqrt

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

def H1(D20,D40,D43,D4n3,D60,D63,D6n3,D66,D6n6):
 H=D20*O20+D40*O40+D43*O43+D4n3*O4n3+D60*O60+D63*O63+D6n3*O6n3+D66*O66+D6n6*O6n6
 Eigenvalues, Eigenvectors = (np.linalg.eigh(H))
 Energy=Eigenvalues-Eigenvalues[0]
 return Energy, Eigenvectors

def H2(D20,D22,D2n2,D40,D42,D4n2,D44,D4n4,D60,D62,D6n2,D64,D6n4,D66,D6n6):
 H=D20*O20+D22*O22+D2n2*O2n2+D40*O40+D42*O42+D4n2*O4n2+D44*O44+D4n4*O4n4+D60*O60+D62*O62+D6n2*O6n2+D64*O64+D6n4*O6n4+D66*O66+D6n6*O6n6
 Eigenvalues, Eigenvectors = (np.linalg.eigh(H))
 Energy=Eigenvalues-Eigenvalues[0]
 return Energy, Eigenvectors

def scattering(i,j,jx,jy,jz):
    #k=8.6173324*10^(-2)
    x=np.dot(np.dot(i,jx),j)
    y=np.dot(np.dot(i,jy),j)
    z=np.dot(np.dot(i,jz),j)
    S=np.dot(np.conj(x),x)+np.dot(np.conj(y),y)+np.dot(np.conj(z),z)
    return S

#start calculating Nd3+ matrces
Nd=Operator()
J=7/2
jplus=np.matrix(Nd.Jplus(J))
jminus=np.matrix(Nd.Jminus(J))
jz=np.matrix(Nd.Jz(J))
jx=0.5*(jplus+jminus)
jy=(jminus-jplus)/(2*1j)
jsquare=np.dot(jx,np.transpose(np.conj(jx)))+np.dot(jy,np.transpose(np.conj(jy)))+np.dot(jz,np.transpose(np.conj(jz)))
jplussquare = jplus*jplus
jminussquare = jminus*jminus
O20 = 3*(jz**2) - jsquare
O22 = 0.5*(jplus**2 + jminus**2)
O40 = 35*(jz**4) - 30*(jsquare*(jz**2)) + 25*(jz**2) -6*(jsquare) + 3*(jsquare**2)
O42 = 0.25*((7*jz**2-jsquare -5)*(jplussquare + jminussquare) + (jplussquare + jminussquare)*(7*jz**2-jsquare -5))
O43 = 0.25*(jz*(jplus**3+jminus**3)+(jplus**3+jminus**3)*jz)
O60 = 231*(jz**6) - 315*(jsquare*(jz**4)) + 735*(jz**4) + 105*((jsquare**2)*(jz**2)) - 525*(jsquare*(jz**2)) + 294*(jz**2) - 5*(jsquare**3) + 40*(jsquare**2) - 60*(jsquare)
O44 = 0.5*(jplus**4 + jminus**4)
O64 = 0.25*( ((11*jz**2)*(jplus**4+jminus**4) -jsquare*(jplus**4+jminus**4) - 38*(jplus**4+jminus**4)) + ((jplus**4+jminus**4)*(11*jz**2) -(jplus**4+jminus**4)*jsquare - (jplus**4+jminus**4)*38) )
O62 = 0.25*((33*jz**4-(18*jsquare+123)*jz**2 + jsquare**2+ 10*jsquare+102)*(jplussquare+jminussquare) + (jplussquare+jminussquare)*(33*jz**4-(18*jsquare+123)*jz**2 + jsquare**2+ 10*jsquare+102))
O63 = 0.25*((11*jz**3-3*jsquare*jz-59*jz)*(jplus**3+jminus**3) + (jplus**3+jminus**3)*(11*jz**3 - 3 * jsquare*jz-59*jz))
O66 = 0.5*((jplus**6+jminus**6))
O2n2 = (0-0.5j)*(jplussquare-jminussquare)
O4n2 = (0-0.25j)*((7*jz**2-jsquare -5)*(jplussquare - jminussquare) + (jplussquare - jminussquare)*(7*jz**2-jsquare -5))
O4n3 = (0-0.25j)*(jz*(jplus**3-jminus**3)+(jplus**3-jminus**3)*jz)
O4n4 = (0-0.5j)*(jplus**4 - jminus**4)
O6n2 = (0-0.25j)*((33*jz**4-(18*jsquare+123)*jz**2 + jsquare**2+ 10*jsquare+102)*(jplussquare-jminussquare) + (jplussquare-jminussquare)*(33*jz**4-(18*jsquare+123)*jz**2 + jsquare**2+ 10*jsquare+102))
O6n3 = (0-0.25j)*((11*jz**3-3*jsquare*jz-59*jz)*(jplus**3-jminus**3) + (jplus**3-jminus**3)*(11*jz**3 - 3 * jsquare*jz-59*jz))
O6n4 = (0-0.25j)*( ((11*jz**2)*(jplus**4-jminus**4) -jsquare*(jplus**4-jminus**4) - 38*(jplus**4-jminus**4)) + ((jplus**4-jminus**4)*(11*jz**2) -(jplus**4-jminus**4)*jsquare - (jplus**4-jminus**4)*38) )
O6n6 = (0-0.5j)*((jplus**6-jminus**6))
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
def solveryb1(B20,B40,B43,B4n3,B60,B63,B6n3,B66,B6n6):
    D20=B20
    D40=B40
    D43=B43
    D4n3=B4n3
    D60=B60
    D63=B63
    D6n3=B6n3
    D66=B66
    D6n6=B6n6
    a=H1(D20,D40,D43,D4n3,D60,D63,D6n3,D66,D6n6)
    Energy=a[0]
    Eigenvectors=np.matrix(a[1])
    scattering1=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,2],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,3],jx,jy,jz)
    scattering2=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,4],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,5],jx,jy,jz)
    scattering3=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,6],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,7],jx,jy,jz)
    #scattering4=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,8],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,9],jx,jy,jz)
    N=scattering3
    s1=scattering1/N
    s2=scattering2/N
    s3=scattering3/N
    #s4=scattering4/N
    #return s1, s2, s3, s4, Energy, Eigenvectors,
    return [s1, s2, s3], Energy, Eigenvectors,jx,jy,jz,

def solveryb2(B20,B22,B2n2,B40,B42,B4n2,B44,B4n4,B60,B62,B6n2,B64,B6n4,B66,B6n6):
    D20=B20
    D22=B22
    D2n2=B2n2
    D40=B40
    D42=B42
    D4n2=B4n2
    D44=B44
    D4n4=B4n4
    D60=B60
    D62=B62
    D6n2=B6n2
    D64=B64
    D6n4=B6n4
    D66=B66
    D6n6=B6n6
    a=H2(D20,D22,D2n2,D40,D42,D4n2,D44,D4n4,D60,D62,D6n2,D64,D6n4,D66,D6n6)
    Energy=a[0]
    Eigenvectors=np.matrix(a[1])
    scattering1=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,2],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,3],jx,jy,jz)
    scattering2=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,4],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,5],jx,jy,jz)
    scattering3=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,6],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,7],jx,jy,jz)
    #scattering4=scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,8],jx,jy,jz)+scattering(np.transpose(np.conj(Eigenvectors[:,0])),Eigenvectors[:,9],jx,jy,jz)
    N=scattering3
    s1=scattering1/N
    s2=scattering2/N
    s3=scattering3/N
    #s4=scattering4/N
    #return s1, s2, s3, s4, Energy, Eigenvectors,
    return [s1, s2, s3], Energy, Eigenvectors,jx,jy,jz,
