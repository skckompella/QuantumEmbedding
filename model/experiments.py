import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import constructions
from coins import groverDiffusion
from layers import *
from datasets import *
from datetime import datetime

class coraNet(nn.Module):
    def __init__(self,adj,num_walkers=None,learn_coin=True,learn_amps=False,onGPU=False,time_steps=1):
        super(coraNet,self).__init__()
        self.qw=qwLayer(adj,num_walkers=num_walkers,
                        learn_coin=learn_coin,learn_amps=learn_amps,
                        onGPU=onGPU,time_steps=time_steps)
        w=torch.DoubleTensor(1432,7)
        w=nn.init.xavier_normal(w)
        w=w-torch.mean(w,dim=0)
        b=np.zeros(7)
        if onGPU:
            self.w=nn.Parameter(w.cuda())
            self.b=nn.Parameter(torch.DoubleTensor(b).cuda())
        else:
            self.w=nn.Parameter(w)
            self.b=nn.Parameter(torch.DoubleTensor(b))
        print b.data

    def forward(self,x):
        x=self.qw(x)
        x=torch.transpose(x,0,1)
        x=x-torch.mean(x,dim=0)
        x=torch.transpose(x,0,1)
        x=torch.matmul(x,self.w)+self.b
        return x

    def toGPU(self):
        self.qw.toGPU()
        self.cuda()

def maskedLoss(lossF,mask,**kwargs):
    """
    Applys a mask over dim 1 of the input/target to lossF
    :param lossF: a loss function, should take an input and a target
    :param mask: An integer array of columns to keep unmasked
    :param kwargs: args to pass to lossF
    :return: A function that computes the masked loss
    """
    def mloss(input,target):
        l = lossF(input[:, mask[0]], target[:, mask[0]], **kwargs)
        for i in range(1, len(mask)):
            l += lossF(input[:, mask[i]], target[:, mask[i]], **kwargs)
        return l
    return mloss

def acc(pred,target):
    x=pred.data.cpu().numpy()
    l=np.argmax(x,axis=1)
    acc=np.sum(l==target)*1.0/len(target)
    y=[np.sum(l==i) for i in range(7)]
    return acc,y

def doExperiment(experiment,network,logging=False,epochs=32,batch_size=16,
                 ongpu=True,learn_amps=True ,learn_coin=True,walk_length=4,
                 train_ratio=0.5,feature_dropout=0.0,walkers=None,
                 shuffleEx=True,shuffleNodes=True):
    print "\nStarting Experiment with Parameters:",[experiment,network,walk_length,learn_amps,learn_coin]

    # Load Data and set experiment specific parameters
    if experiment == "weather":
        data = numpyDataset(datafile='ustemp/2009.npy',
                            adjfile='ustemp/adj.npy',
                            offset=1,
                            trainRatio=0.5,
                            shuffleEx=shuffleEx,
                            shuffleNodes=shuffleNodes,
                            dropout_p=feature_dropout,
                            asTensor=True)
        criterion = nn.MSELoss()
    elif experiment == "cora":
        data = coraDataset(dropout_p=feature_dropout)
        mask = np.arange(7)
        nnet = coraNet
        labs = data.dataY
        class_weights = [np.sum(labs[:np.int(len(labs) * train_ratio)] == i) for i in range(7)]
        class_weights = 1. / np.array(class_weights)
        if ongpu:
            class_weights = torch.from_numpy(class_weights / np.mean(class_weights)).cuda()
        else:
            class_weights = torch.from_numpy(class_weights / np.mean(class_weights))
        criterion = maskedLoss(F.cross_entropy, mask, weight=class_weights)

    if walkers is None:
        walkers = len(data.adj)

    if network == "qw":
        net = qwLayer(data.adj_list,
                      num_walkers=walkers, time_steps=walk_length,
                      learn_amps=learn_amps, learn_coin=learn_coin,
                      onGPU=ongpu)
    if network == "qw1c":
        net=qwLayer1C(data.adj_list,
                      num_walkers=walkers, time_steps=walk_length,
                      learn_amps=learn_amps, learn_coin=learn_coin,
                      onGPU=ongpu)
    elif network == "dc":
        A = [torch.DoubleTensor(np.eye(len(data.adj)))]
        adj=data.adj.double()
        adj=adj/(torch.sum(adj,dim=1)[:,None])
        A.append(adj)
        for i in range(walk_length-1):
            A.append(torch.matmul(A[-2], A[-1]))
        A = torch.stack(A)
        net = dcLayer(data.dataX.shape[-1], hops=walk_length, adj=A, sumHops=True, onGPU=ongpu)
    elif network == "spectral":
        adj=data.adj.numpy()
        adj=adj+np.eye(len(adj))
        d=np.sum(adj,axis=1)
        disqrt=(1./(np.sqrt(d)))*np.eye(len(d))
        L=disqrt.dot(adj).dot(disqrt)
        v,u=np.linalg.eigh(L)
        net = spectralLayer(torch.from_numpy(np.real(u[:,:walk_length])).double(),v[:walk_length],ongpu=ongpu)

    opt = optim.Adam(net.parameters())

    running_loss = 0.0
    print "Beginning Traning.."
    print "Epoch Batch Loss"
    besttest=1000
    if logging:
        f = open('results/'+
            datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "Results-" + experiment +"-"+network+"-"+ str(learn_amps) + "-" + str(
                learn_coin) + "-" + str(walk_length) + "-" + str(walkers) + ".log", "a+")
    dloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    patience=0
    los=[]
    for iter in range(epochs):
        for i_batch, (x, y) in enumerate(dloader):
            if ongpu:
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
            else:
                x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
            opt.zero_grad()  # zero the gradient buffers
            output = net(x)
            # loss = criterion(output,y)
            loss = criterion(output, y)
            loss.backward()
            opt.step()  # Does the update

            running_loss += loss.data[0]
            loss_batches = 4
            if i_batch % loss_batches == loss_batches - 1:  # print every 10 mini-batches
                print('%5d %5d %.3f' %
                      (iter + 1, i_batch + 1, running_loss / loss_batches))
                if logging:
                    f.write('%5d %5d %.3f\n' %
                            (iter + 1, i_batch + 1, running_loss / loss_batches))
                running_loss = 0.0
                # ac=acc(output[0,:train_size], data.dataY[:train_size])
                # print "Train Accuracy:",ac
                # ac=acc(output[0,train_size:], data.dataY[train_size:])
                # print "Test Accuracy:", ac
                # f.write(str(ac)+"\n")

        x, y = data.testSet()
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        out = net(x)
        # inds=np.argmax(out,axis=1)
        # acc=np.sum(inds[train_size:]==y[train_size:])
        testloss = criterion(out, y).data.cpu().numpy()[0]
        los.append(testloss)
        print "Test Loss: ", testloss
        print "Test Loss per Node:", testloss / len(data.adj)
        if logging:
            f.write("iter: " + str(iter) + " Test Loss: " + str(testloss)+"\n")
        if testloss<besttest:
            patience=0
            besttest=testloss
        if patience==8:
            break
    if logging:
        f.close()
    return besttest,los


if __name__=="__main__":
    experiment="weather" #"cora"/"weather"
    network="dc" #qw, dc, spectral
    logging=False
    epochs=32 #>=1
    batch_size=16 #>1
    ongpu=True #True/False
    learn_amps=True #True/False
    learn_coin=True #True/False
    walk_length=4 #>1
    train_ratio=0.5 #(0,1]
    feature_dropout=0.0 #[0:1)
    walkers=None #None or #, None uses maximum number of walkers
    shuffleEx=True
    shuffleNodes=True

    if network == "qw":
        nnet = qwLayer
    elif network == "dc":
       def summeddcLayer(**kwargs):
           return dcLayer(features=features,hops=hops,adj=adj,addBias=addBias,
                 nonlinearity=nonlinearity,w=w,sumHops=True)
       nnet=summeddcLayer
    elif network == "spectral":
        nnet = spectralLayer

    #Load Data and set experiment specific parameters
    print "Loading Dataset.."
    if experiment=="weather":
        data=numpyDataset(datafile='ustemp/2009.npy',
                          adjfile='ustemp/adj.npy',
                          offset=1,
                          trainRatio=0.5,
                          shuffleEx=shuffleEx,
                          shuffleNodes=shuffleNodes,
                          dropout_p=feature_dropout,
                          asTensor=True)
        criterion=nn.MSELoss()
    elif experiment=="cora":
        data = coraDataset(dropout_p=feature_dropout)
        mask = np.arange(7)
        nnet=coraNet
        labs=data.dataY
        class_weights=[np.sum(labs[:np.int(len(labs)*train_ratio)]==i) for i in range(7)]
        class_weights=1./np.array(class_weights)
        if ongpu:
            class_weights=torch.from_numpy(class_weights/np.mean(class_weights)).cuda()
        else:
            class_weights = torch.from_numpy(class_weights / np.mean(class_weights))
        criterion=maskedLoss(F.cross_entropy,mask,weight=class_weights)

    print "Building Network.."
    if walkers is None:
        walkers=len(data.adj)

    if network == "qw":
        net = qwLayer(data.adj_list,
                num_walkers=walkers, time_steps=walk_length,
                learn_amps=learn_amps, learn_coin=learn_coin,
                onGPU=ongpu)
    elif network == "dc":
        A=[torch.DoubleTensor(np.eye(len(data.adj)))]
        A.append(data.adj.double())
        A.append(torch.matmul(A[-2],A[-1]))
        A=torch.stack(A)
        net = dcLayer(data.dataX.shape[-1],hops=2,adj=A,sumHops=True,onGPU=ongpu)
    elif network == "spectral":
        net = spectralLayer


    opt=optim.Adam(net.parameters())

    running_loss=0.0
    print "\nBeginning Traning.."
    print "Epoch Batch Loss"
    if logging:
        f=open(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"Results-"+experiment+str(learn_amps)+"-"+str(learn_coin)+"-"+str(walk_length)+"-"+str(walkers)+".log","a+")
    dloader=DataLoader(data, batch_size=batch_size,shuffle=True,num_workers=1)
    los=[]
    for iter in range(epochs):
        for i_batch,(x,y) in enumerate(dloader):
            if ongpu:
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
            else:
                x,y=Variable(x,requires_grad=False),Variable(y,requires_grad=False)
            opt.zero_grad()  # zero the gradient buffers
            output = net(x)
            #loss = criterion(output,y)
            loss=criterion(output,y)
            loss.backward()
            opt.step()  # Does the update

            running_loss += loss.data[0]
            loss_batches=4
            if i_batch % loss_batches == loss_batches-1:  # print every 10 mini-batches
                print('%5d %5d %.3f' %
                      (iter + 1, i_batch + 1, running_loss / loss_batches))
                if logging:
                    f.write('%5d %5d %.3f\n' %
                            (iter + 1, i_batch + 1, running_loss / loss_batches))
                running_loss = 0.0
                #ac=acc(output[0,:train_size], data.dataY[:train_size])
                #print "Train Accuracy:",ac
                #ac=acc(output[0,train_size:], data.dataY[train_size:])
                #print "Test Accuracy:", ac
                #f.write(str(ac)+"\n")


        x,y=data.testSet()
        x=Variable(x.cuda())
        y=Variable(y.cuda())
        out=net(x)
        #inds=np.argmax(out,axis=1)
        #acc=np.sum(inds[train_size:]==y[train_size:])
        testloss=criterion(out,y).data.cpu().numpy()[0]
        los.append(testloss)
        print "Test Loss: ", testloss
        print "Test Loss per Node:",testloss/len(data.adj)
        if logging:
            f.write("iter: "+str(iter)+" Test Loss: "+str(testloss))
    if logging:
        f.close()