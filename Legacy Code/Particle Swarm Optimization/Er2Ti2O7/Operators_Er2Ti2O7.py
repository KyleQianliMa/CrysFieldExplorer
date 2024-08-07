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

def H(D20,D40,D43,D60,D63,D66):
 alpha=4/(45*35)
 beta=2/(11*15*273)
 gamma=8/(13*13*11*11*3*3*3*7)
 theta_6=-1/891891
 r2=0.6980
 r4=1.218
 r6=4.502
#----pay attention to the 100,1000 constants when converting back to the real parameters 
 H=alpha*r2*D20*O20*100 + beta*r4*D40*O40*100 + beta*r4*D43*O43*1000 + gamma*r6*D60*O60*100 + gamma*r6*D63*O63*100 + gamma*r6*D66*O66*100
 Eigenvalues, Eigenvectors = (np.linalg.eigh(H))
 # Energy=Eigenvalues-Eigenvalues[0]
 Energy=Eigenvalues
 return Energy, Eigenvectors, H

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
O22 = 0.5*(jplus**2 + jminus**2)
O40 = 35*(jz**4) - 30*(jsquare*(jz**2)) + 25*(jz**2) -6*(jsquare) + 3*(jsquare**2)
O42 = 0.25*((jplus**2 + jminus**2)*(7*jz**2-jsquare) -(jplus**2 + jminus**2)*5+
            (7*jz**2-jsquare)*(jplus**2 + jminus**2)-5*(jplus**2 + jminus**2))
O43 = 0.25*(jz*(jplus**3+jminus**3)+(jplus**3+jminus**3)*jz)
O60 = 231*(jz**6) - 315*(jsquare*(jz**4)) + 735*(jz**4) + 105*((jsquare**2)*(jz**2)) - 525*(jsquare*(jz**2)) + 294*(jz**2) - 5*(jsquare**3) + 40*(jsquare**2) - 60*(jsquare)
O44 = 0.5*(jplus**4 + jminus**4)
O64 = 0.25*((jplus**4 + jminus**4)*(11*jz**2 -jsquare) -(jplus**4 + jminus**4)*38 + (11*jz**2 -jsquare)*(jplus**4 + jminus**4)-38*(jplus**4 + jminus**4))
O62 = 0.25*((jplus**2 + jminus**2)*(33*jz**4 -(18*jsquare*jz**2+123*jz**2) +jsquare**2 +10*jsquare) +((jplus**2 + jminus**2)*102) + 
            (33*jz**4 -(18*jsquare*jz**2+123*jz**2) +jsquare**2 +10*jsquare)*(jplus**2 + jminus**2)+102*(jplus**2 + jminus**2))
O63 = 0.25*((11*jz**3-3*jsquare*jz-59*jz)*(jplus**3+jminus**3) + (jplus**3+jminus**3)*(11*jz**3 - 3 * jsquare*jz-59*jz))
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

#B20  =  212
#B40  =  211
#B44  =  -111
#B60  =  -206
#B64  =  79
#vertical rolls are eigenvectors
def solver(B20,B40,B43,B60,B63,B66):
 D20=B20/k20
 D40=B40/k40
 #D44=B44/k44
 D43=B43/k43
 D60=B60/k60
 #D64=B64/k64
 D63=B63/k63
 D66=B66/k66
 a=H(D20,D40,D43,D60,D63,D66)
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
 return np.array([s1, s2, s3,s4,s5,s6,s7]).squeeze(), Energy, Eigenvectors,jx,jy,jz, HH

def magH(jx, jy, jz, Bx,By,Bz):
     S=3/2;L=6;J=15/2;
     gj=(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) +(J*(J+1) + S*(S+1) - L*(L+1))/(J*(J+1))
     Na=6.0221409e23
     muBT=5.7883818012e-2
     muB=9.274009994e-21
     kb=1.38064852e-16
     magH = -gj*muBT*(Bx*jx+By*jy+Bz*jz)
     return magH

def magsovler1(B20,B40,B43,B60,B63,B66,Bx,By,Bz):
    D20=B20
    D40=B40
    D43=B43
    D60=B60
    D63=B63
    D66=B66
    a=H(D20,D40,D43,D60,D63,D66)
    b=magH(jx, jy, jz, Bx, By, Bz)
    Eigenvalues, Eigenvectors = (np.linalg.eigh(a[2]+b))
    Energy=Eigenvalues-Eigenvalues[0]
    return Energy, Eigenvectors, jx,jy,jz,