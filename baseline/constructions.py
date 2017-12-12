import numpy as np
import collections
import scipy.sparse as sp

def createTorusShift(dim=1,w=3,splitDims=False):
    """
    Creates a the shift tensor for a lattice of given specifications.
    :param dim: Dimension of the hypertorus (i.e. 1=circle, 2=donut)
    :param w: Scaler or enumerable designating widths along each dimension. Scaler implies equiwidth dimensions.
    :param splitDims: For a 2d-torus, returns a (w[0],w[1],d,w[0],w[1],d) dimensional tensor instead of (w[0]w[1],d,w[0]w[1],d).
    :return: A 4(6) dimensional tensor
    """
    if not isinstance(w, collections.Iterable):
        w=[w]*dim
    if dim == 1:
        shift = np.zeros((w[0], 2 * dim, w[0], 2 * dim), dtype=float)
        for i in range(w[0]):
            shift[i, [0, 1], [(i + 1) % w[0], i - 1], [0, 1]] = [1, 1]
    elif dim == 2 and splitDims: #Useful for plotting
        # Shift Operator. If pos(i,j) and pos(k,l) are adjacent add a directional vector to the 2nd and 5th dimensions
        shift = np.zeros((w[0], w[1], 2 * dim, w[0], w[1], 2 * dim), dtype=float)
        for i in range(w[0]):
            for j in range(w[1]):
                shift[i, j, [0, 1, 2, 3],
                    [(i + 1) % w[0], i - 1, i, i],
                    [j, j, (j + 1) % w[1], j - 1], [0, 1, 2, 3]] =[1, 1, 1, 1]
    else:
        N = np.prod(w)
        d = 2 * dim
        shift = np.zeros((N, d, N, d), dtype=float)
        for j in np.arange(dim,dtype=int):
            offset = np.prod(w[:j], dtype=int)
            mod = np.prod(w[:j + 1], dtype=int)
            for i in np.arange(N,dtype=int):
                shift[i, 2*j, (i + offset) % mod + (i / mod) * mod, 2*j] = 1
                shift[i, 2*j+1, (i - offset) % mod + (i / mod) * mod, 2*j+1] = 1
    return shift

def createTorusAdj(dim=1,w=3):
    """
    Creates an adjacency matrix for a torus with the given parameters
    :param dim: Number of dimensions of the hypertorus
    :param w: Scaler or enumerable designating widths along each dimension. Scaler implies equiwidth dimensions.
    :return: N x N adjacency matrix where N=w[0]*w[1]*..*w[dim-1]
    """
    if not isinstance(w, collections.Iterable):
        w=[w]*dim
    N=np.prod(w)
    d = 2 * dim
    adj = np.zeros((N,N), dtype=float)
    for j in np.arange(dim,dtype=int):
        offset = np.prod(w[:j], dtype=int)
        mod = np.prod(w[:j + 1], dtype=int)
        for i in np.arange(N,dtype=int):
            adj[i, (i + offset) % mod + (i / mod) * mod] = 1
            adj[i, (i - offset) % mod + (i / mod) * mod] = 1
    return adj

def shiftToAdj(shift):
    return np.sum(np.sum(shift,np.ndim(shift)/2-1),-1)

def adjToSwap(adj,sparse=False):
    """
    Constructs a tensor operator that swaps 1 direction of amplitude for adjacent nodes
    :param adj: An n x n adjaceny matrix
    :return: An n x d x n x d tensor, where d is the max degree of the graph
    """
    n=adj.shape[0]
    d=np.int(np.max(np.sum(adj,axis=1)))
    if sparse:
        swap=sp.coo_matrix((n,d,n,d),dtype=int)
    else:
        swap=np.zeros((n,d,n,d))
    ind=np.zeros(n,dtype=int)

    for i in range(n):
        for j in np.arange(i+1,n):
            if adj[i,j]==1:
                swap[i,ind[i],j,ind[j]]=1
                swap[j,ind[j],i,ind[i]]=1
                ind[i]+=1
                ind[j]+=1
    return swap

def createRandomRing(w):
    shift=np.zeros((w,2,w,2),dtype=float)
    for i in np.arange(w,dtype=int):
        a=np.random.permutation([0,1])
        shift[i, a, [(i + 1) % w, i - 1], a] = [1, 1]
    return shift

def adj2list(adj,conversion=None):
    adj_list=[]
    for i in range(len(adj)):
        inds=np.where(adj[i]==1)[0]
        if conversion is not None:
            adj_list.append(conversion(inds))
        else:
            adj_list.append(inds)
    return adj_list