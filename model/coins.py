#Requires numpy 1.13 or later for block method
import numpy as np
from scipy import sparse

def grover(n):
    """
    Constructs the n x n Grover Coin.
    NOTE: THIS COIN IS ONLY UNITRAY FOR n=4!
    :param n: Dimensionality of the coin
    :return: Grover Coin in n dimensions
    """
    g=np.ones((n, n), dtype=float)
    np.fill_diagonal(g,-1)
    g=g/np.sqrt(n)
    return g

def groverDiffusion(n):
    g=np.ones((n,n))
    g *= 2./n
    np.fill_diagonal(g,-1+2./n)
    return g

def hadamard(k):
    """
    Iteratively constructs a Hadamard coin of the given order.
    :param k: The log_2 of the size of the coin
    :return: A 2^k dimensional coin of the form [[H_k-1,H_k-1],[H_k-1,-H_k-1]]
    """

    h=np.ones((1,1))
    for i in range(0, k):
        h=np.block([[h,h],[h,-h]])
    h=h/np.sqrt(2 ** k)
    return h

def RotationMatrix(n,thetas,compact=True,full=True):
    """
    Returns a real valued rotation matrix in n-dimensions.
    :param n: Dimensions
    :param thetas: Rotation about each n-2 dimensional hyperplane. Given dimensions i_0 to i_{n-1}, theta_0 rotates
    the (i_0,i_1) coordinate. Theta_1 rotates the (i_0,i_2) coordinates ... theta_n rotates the (i_1,i_2) coordinates...
    theta_{n(n-1)/2} rotates the (i_n-2,i_n-1) coordinate.
    :param compact: If True, return a single rotation matrix that is the product of each hyperplane rotation matrix.
    :param full: If true, return a sparse matrix, otherwise an ndarray
    :return: The rotation matrix or a list of hyperplane rotation matrices.
    """
    assert n*(n-1)/2==len(thetas)

    R=[]
    t=0
    for i in np.arange(0,n):
        for j in np.arange(i+1,n):
            if thetas[i]==0:
                if compact:
                    pass
                else:
                    R.append(sparse.eye(n,format='dia',dtype=float))
            else:
                diags=np.array([[0]*n,[1]*n,[0]*n],dtype=float)
                diags[[1,2,0,1],[i,j,i,j]]=[np.cos(thetas[t]),np.sin(thetas[t]),
                                             -np.sin(thetas[t]),np.cos(thetas[t])]
                R.append(sparse.dia_matrix((diags,[i-j,0,j-i]),shape=(n,n),dtype=float))
            t+=1
    if compact:
        rot = sparse.eye(n, format='coo',dtype=float)
        for r in R:
            rot = rot.dot(r)
        if full:
            return rot.toarray()
        return rot
    else:
        if full:
            return R.toarray()
        return R