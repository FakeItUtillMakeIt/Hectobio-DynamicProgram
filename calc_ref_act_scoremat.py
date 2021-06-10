#input:
#reference  and  actually level vector
import numpy as np
from numpy.core.defchararray import mod


def logdet(A):
    A=np.reshape(A,newshape=(-1,1))
    R=np.linalg.cholesky(A)
    ld=2*np.sum(np.log(np.diag(R)))
    return ld

def calc_scoremat(x1,k1,x2,k2):
    x1_shape=x1.shape[0]
    x2_shape=x2.shape[0]
    s=-0.5*1*np.log(2*np.pi)*np.ones(shape=(x1_shape,x2_shape))
    pifactor=-0.5*1*np.log(2*np.pi)

    n_1=x1.shape[0]
    n_2=x2.shape[0]
    n_element=n_1*n_2

    for cE in range(1,n_element+1):
        c1=(cE % n_1)-1
        
        c2=int(np.ceil(cE/n_1))-1
        this_k1=k1[c1]
        this_x1=x1[c1]
        this_k1x1=this_k1*this_x1
        t_this_x1=np.transpose(this_x1)
        this_x1k1x1=t_this_x1*this_k1x1

        this_k2=k2[c2]
        this_x2=x2[c2]
        this_k2x2=this_k2*this_x2
        this_x2k2x2=np.transpose(this_x2)*this_k2x2

        this_kx=this_k1x1+this_k2x2
        this_k1k2=this_k1+this_k2

        l_k1=logdet(this_k1)
        l_k2=logdet(this_k2)
        l_k1k2=logdet(this_k1k2)
        t_this_kx=np.transpose(this_kx)
        #对应论文中的6小节中的公式16
        s[c1][c2]=pifactor+0.5*(l_k1+l_k2-l_k1k2-this_x1k1x1-this_x2k2x2+t_this_kx/this_k1k2*this_kx)
        #print(s[c1][c2])
    return s
    pass
