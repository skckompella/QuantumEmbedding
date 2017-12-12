import numpy as np
import torch
from torch.utils.data import Dataset
from model import constructions
from common import utils


class numpyDataset(Dataset):
    def __init__(self,datafile='ustemp/2009.npy',adjfile='ustemp/adj.npy',
                 offset=1,trainRatio=1,shuffleEx=False,shuffleNodes=False,
                 dropout_p=0.0,asTensor=False):

        data = np.load('ustemp/2009.npy').astype(float)
        if len(data.shape)==2:
            data=np.expand_dims(data,-1)

        self.dataX=data[:-offset]
        self.dataY=data[offset:]
        self.adj = np.load('ustemp/adj.npy').astype(int)
        self.dropout_p=dropout_p

        if shuffleEx:
            perm = np.random.permutation(len(self.dataX))
            self.dataX=self.dataY[perm]
            self.dataY=self.dataY[perm]

        if shuffleNodes:
            perm=np.random.permutation(len(self.adj))
            self.dataX=self.dataX[:,perm]
            self.dataY=self.dataY[:,perm]
            self.adj=self.adj[perm][:,perm]

        self.asTensor=asTensor
        if self.asTensor:
            self.dataX=torch.from_numpy(self.dataX)
            self.dataY=torch.from_numpy(self.dataY)
            self.adj_list=constructions.adj2list(self.adj,torch.from_numpy)
            self.adj=torch.from_numpy(self.adj)
        else:
            self.adj_list=constructions.adj2list(self.adj)

        self.offset=offset
        self.trainRatio=trainRatio

    def __len__(self):
        #Returns the length of the train set
        return np.int(len(self.dataX)*self.trainRatio)

    def __getitem__(self,idx):
        #Returns an item from the train set
        mask = np.random.choice([0, 1], self.dataX[idx].shape, p=[self.dropout_p, 1 - self.dropout_p])
        if self.asTensor:
            mask=torch.DoubleTensor(mask)
        return (self.dataX[idx] * mask, self.dataY[idx])

    def testSet(self):
        #Returns the entire test set
        return (self.dataX[self.__len__():],
                self.dataY[self.__len__():])

    def trainSet(self):
        #Returns the entire train set
        return (self.dataX[:self.__len__()],
                self.dataY[:self.__len__()])


class coraDataset(Dataset):
    def __init__(self,path="data/cora/",dataset="cora",
                 length=None,shuffleNodes=False,dropout_p=0.0):
        label2index = {
            'Case_Based': 0,
            'Genetic_Algorithms': 1,
            'Neural_Networks': 2,
            'Probabilistic_Methods': 3,
            'Reinforcement_Learning': 4,
            'Rule_Learning': 5,
            'Theory': 6
        }

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        self.dataX = np.array(idx_features_labels[:, 1:-2], dtype=float)
        self.dataY = np.array([label2index[i] for i in idx_features_labels[:, -1]])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)

        adj = np.zeros((self.dataY.shape[0], self.dataY.shape[0]))
        for e in edges:
            adj[e[0],e[1]]=1
        adj = adj + adj.T
        adj[adj>0]=1
        self.adj=adj

        if shuffleNodes:
            perm=np.random.permute(len(adj))
            self.adj=self.adj[perm][:,perm]
            self.dataX=self.dataX[perm]
            self.dataY=self.dataY[perm]

        self.adj_list=constructions.adj2list(self.adj,torch.from_numpy)

        if length is None:
            self.length=len(self.dataX)
        else:
            self.length=length

        self.dropout_p=dropout_p

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        mask=np.random.choice([0,1], self.dataX.shape, p=[self.dropout_p, 1 - self.dropout_p])
        out=(torch.from_numpy(self.dataX * mask),
             torch.from_numpy(self.dataY))
        return out

    def trainSet(self):
        out = (torch.from_numpy(self.dataX), torch.from_numpy(self.dataY))

    def testSet(self):
        out=(torch.from_numpy(self.dataX), torch.from_numpy(self.dataY))


class SentimentDataset(Dataset):

    def __init__(self, data_path, labels_path, max_len, train_test_ratio=0.8):

        self.data_x = np.load(data_path)
        self.data_y = np.load(labels_path)
        self.max_len = max_len
        self.train_test_ratio = train_test_ratio
        self.adj = utils.get_sentiment_adjacency_matrix(self.max_len)
        self.adj_list = constructions.adj2list(self.adj, torch.from_numpy)

    def __len__(self):
        return int(len(self.data_x) * self.train_test_ratio)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data_x[idx]), torch.from_numpy(self.data_y[idx])

    def get_train_set(self):
        return self.data_x[:self.__len__()], self.data_y[:self.__len__()]

    def get_test_set(self):
        return self.data_x[self.__len__():], self.data_y[self.__len__():]
