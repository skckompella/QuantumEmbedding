import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from coins import groverDiffusion

class qwLayer(nn.Module):
    def __init__(self,adj,num_walkers=None,learn_coin=True,
                 learn_amps=False,onGPU=False,time_steps=1,
                 **kwargs):
        super(qwLayer, self).__init__()
        self.nodes=len(adj)
        self.degrees=[len(a) for a in adj]
        self.maxdegree=np.max(self.degrees)
        self.time_steps=time_steps
        self.adj=adj #list of LongTensor of ind
        self.edges=np.sum(self.degrees)
        self.learn_amps=learn_amps
        self.learn_coin=learn_coin

        if num_walkers is None:
            self.walkers=self.nodes
        else:
            self.walkers=num_walkers

        #amp dimensions are [nodes, spins, walkers]
        init_amps=np.zeros((self.nodes,self.maxdegree,self.walkers))
        for i in range(self.nodes):
            #Sart Walkers across all nodes when fewer walkers than nodes
            if self.walkers<self.nodes:
                init_amps[i,:self.degrees[i]] = np.random.rand(self.degrees[i],self.walkers)*2-1
            else:
                init_amps[i,:self.degrees[i],i]=1./np.sqrt(self.degrees[i])
        if self.walkers<self.nodes:
            for w in range(self.walkers):
                init_amps[:,:,w]=init_amps[:,:,w]/np.linalg.norm(init_amps[:,:,w])
        if self.learn_amps:
            self.init_amps=nn.Parameter(torch.from_numpy(init_amps))
        else:
            self.init_amps=Variable(torch.from_numpy(init_amps),requires_grad=False)

        #Create Swap Tensor
        swap=np.zeros((2,self.nodes*self.maxdegree),dtype=int)
        a,b=[],[]
        inds=np.zeros(self.nodes)
        for i in range(len(adj)):
            for n in range(self.maxdegree):
                if n<adj[i].size()[0]:
                    node=adj[i][n]
                    a.append(node)
                    b.append(inds[node])
                    inds[node]+=1
                else:
                    a.append(i)
                    b.append(n)
        swap[0]=a
        swap[1]=b
        self.swap=torch.from_numpy(swap)

        #Create Coins
        if self.learn_coin:
            self.coins=nn.ParameterList()
            for i in range(self.nodes):
                self.coins.append(nn.Parameter(torch.from_numpy(groverDiffusion(self.degrees[i]))))
        else:
            self.coins=[]
            for i in range(self.nodes):
                self.coins.append(Variable(torch.from_numpy(groverDiffusion(self.degrees[i])),
                                           requires_grad=False))

        self.onGPU=onGPU
        if self.onGPU:
            self.toGPU()

    def forward(self,x):
        amps = self.init_amps
        for t in range(self.time_steps):
            if self.onGPU:
                a = Variable(torch.DoubleTensor(self.init_amps.size()).zero_().cuda(),
                             requires_grad=False)
            else:
                a=Variable(torch.DoubleTensor(self.init_amps.size()).zero_(),
                                     requires_grad=False)

            for i in range(len(self.coins)):
                #Coin Operator
                r=torch.matmul(self.coins[i],amps[i,:self.degrees[i]])
                #Shift Operator
                a[i,:self.degrees[i]]=r
            amps=a[self.swap[0],self.swap[1]].view(self.init_amps.size())
        d1=torch.sum(amps*amps,dim=1)
        d2=torch.sum(self.init_amps*self.init_amps,dim=1)
        self.D=torch.matmul(d2,torch.transpose(d1,0,1))
        return torch.matmul(torch.transpose(self.D,0,1),x)

    def toGPU(self):
        self.onGPU=True
        self.cuda()
        self.swap=self.swap.cuda()
        if not self.learn_amps:
            self.init_amps=self.init_amps.cuda()
        if not self.learn_coin:
            for i in range(len(self.coins)):
                self.coins[i]=self.coins[i].cuda()

class qwLayer1C(nn.Module):
    def __init__(self,adj,num_walkers=None,learn_coin=True,
                 learn_amps=False,onGPU=False,time_steps=1,
                 **kwargs):
        super(qwLayer1C, self).__init__()
        self.nodes=len(adj)
        self.degrees=[len(a) for a in adj]
        self.maxdegree=np.max(self.degrees)
        self.time_steps=time_steps
        self.adj=adj #list of LongTensor of ind
        self.edges=np.sum(self.degrees)
        self.learn_amps=learn_amps
        self.learn_coin=learn_coin

        if num_walkers is None:
            self.walkers=self.nodes
        else:
            self.walkers=num_walkers

        #amp dimensions are [nodes, spins, walkers]
        init_amps=np.zeros((self.nodes,self.maxdegree,self.walkers))
        for i in range(self.nodes):
            #Sart Walkers across all nodes when fewer walkers than nodes
            if self.walkers<self.nodes:
                init_amps[i,:self.degrees[i]] = np.random.rand(self.degrees[i],self.walkers)*2-1
            else:
                init_amps[i,:self.degrees[i],i]=1./np.sqrt(self.degrees[i])
        if self.walkers<self.nodes:
            for w in range(self.walkers):
                init_amps[:,:,w]=init_amps[:,:,w]/np.linalg.norm(init_amps[:,:,w])
        if self.learn_amps:
            self.init_amps=nn.Parameter(torch.from_numpy(init_amps))
        else:
            self.init_amps=Variable(torch.from_numpy(init_amps),requires_grad=False)

        #Create Swap Tensor
        swap=np.zeros((2,self.nodes*self.maxdegree),dtype=int)
        a,b=[],[]
        inds=np.zeros(self.nodes)
        for i in range(len(adj)):
            for n in range(self.maxdegree):
                if n<adj[i].size()[0]:
                    node=adj[i][n]
                    a.append(node)
                    b.append(inds[node])
                    inds[node]+=1
                else:
                    a.append(i)
                    b.append(n)
        swap[0]=a
        swap[1]=b
        self.swap=torch.from_numpy(swap)

        #Create Coins
        if self.learn_coin:
            self.coins = nn.ParameterList()
            for t in range(self.time_steps):
                self.coins.append(nn.Parameter(torch.from_numpy(
                    groverDiffusion(self.maxdegree))))
        else:
            self.coins=[Variable(torch.from_numpy(
                groverDiffusion(self.maxdegree),
                requires_grad=False)) for t in range(self.time_steps)]

        self.onGPU=onGPU
        if self.onGPU:
            self.toGPU()

    def forward(self,x):
        amps = self.init_amps
        for t in range(self.time_steps):
            #Coin Operator
            a=torch.matmul(self.coins[t],amps)
            #Swap Operator
            amps=a[self.swap[0],self.swap[1]].view(self.init_amps.size())
        d1=torch.sum(amps*amps,dim=1)
        d2=torch.sum(self.init_amps*self.init_amps,dim=1)
        self.D=torch.matmul(d2,torch.transpose(d1,0,1))
        return torch.matmul(torch.transpose(self.D,0,1),x)

    def toGPU(self):
        self.onGPU=True
        self.cuda()
        self.swap=self.swap.cuda()
        if not self.learn_amps:
            self.init_amps=self.init_amps.cuda()
        if not self.learn_coin:
            for t in range(self.time_steps):
                self.coins[t]=self.coins[t].cuda()


class dcLayer(nn.Module):
    def __init__(self,features,hops,adj=None,addBias=True,
                 nonlinearity=None,w=None,sumHops=False,onGPU=False,**kwargs):
        super(dcLayer, self).__init__()

        self.features=features
        self.hops=hops
        self.sumHops=sumHops

        if onGPU:
            self.A=Variable(adj.cuda(),requires_grad=False)
        self.sumHops=sumHops

        if w is None:
            w=np.random.normal(0,0.01,(1,self.hops+1,1,self.features))
        self.w=nn.Parameter(torch.from_numpy(w))

        self.addBias=addBias
        if self.addBias:
            b=np.zeros((1,self.hops+1,1,self.features))
            self.b=nn.Parameter(torch.from_numpy(b))

        self.applyNonlinearity=nonlinearity is not None
        if self.applyNonlinearity:
            self.h=nonlinearity

        if onGPU:
            self.cuda()

    def forward(self,x,A=None):
        if A is None:
            A=self.A

        y=torch.matmul(A[None],x[:,None])
        y=torch.mul(y,self.w)

        if self.addBias:
            y=y+self.b
        if self.sumHops:
            y=torch.sum(y,dim=1)
        if self.applyNonlinearity:
            y=self.h(y)

        return y

class spectralLayer(nn.Module):
    def __init__(self,U,V=None,addBias=True,nonlinearity=None,W=None,ongpu=False):
        super(spectralLayer,self).__init__()
        if ongpu:
			self.U=Variable(U.cuda())
        else:
			self.U=Variable(U)

        if W is None:
            if V is None:
			    W=np.random.normal(0,0.01,(self.U.size()[1],self.U.size()[1]))
            else:
                W=np.eye(len(V))*V
        
        self.W=nn.Parameter(torch.from_numpy(W))

        self.addBias=addBias
        if self.addBias:
            b=np.zeros((1,self.U.size()[0],1))
            self.b=nn.Parameter(torch.from_numpy(b))

        self.h=nonlinearity

        if ongpu:
            self.cuda()

    def forward(self,x):
        y=torch.matmul(torch.t(self.U),x)
        y=torch.matmul(self.W,y)
        y=torch.matmul(self.U,y)

        if self.addBias:
            y=y+self.b
        if self.h is not None:
            y=self.h(y)

        return y


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureExtractor, self).__init__()

        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        scores = self.softmax(out3)

        return scores