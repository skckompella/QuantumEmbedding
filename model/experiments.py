from datetime import datetime

import torch.optim as optim
from torch.utils.data import DataLoader

from common import constants
from layers import *
from model.datasets import *
import pickle


class SentimentNet(nn.Module):
    def __init__(self, word_to_idx, embedding_size,
                 adj, qw_network="qw1c", num_walkers=None, learn_coin=True, learn_amps=False, onGPU=False,
                 time_steps=1):
        super(SentimentNet, self).__init__()
        self.word_to_idx = word_to_idx
        self.NULL_IDX = 0
        self.embedding = nn.Embedding(len(self.word_to_idx), embedding_size, padding_idx=self.NULL_IDX)

        self.qw = qwLayer(adj, num_walkers=num_walkers,
                          learn_coin=learn_coin, learn_amps=learn_amps,
                          onGPU=onGPU, time_steps=time_steps)
        if qw_network == "qw1c":
            self.qw = qwLayer1C(adj,
                                num_walkers=num_walkers, time_steps=time_steps,
                                learn_amps=learn_amps, learn_coin=learn_coin,
                                onGPU=onGPU)

        self.mlp = FeatureExtractor(embedding_size, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.qw.forward(x)
        x = self.mlp.forward(x)
        return x


def doExperiment(experiment, qw_network, embedding_size=128, logging=False, epochs=32, batch_size=16,
                 ongpu=True, learn_amps=True, learn_coin=True, walk_length=4,
                 train_ratio=0.5, feature_dropout=0.0, walkers=None,
                 shuffleEx=True, shuffleNodes=True):
    print "\nStarting Experiment with Parameters:", [experiment, qw_network, walk_length, learn_amps, learn_coin]

    data = None
    net = None

    with open(constants.WORD_TO_IDX_PATH, "rb") as w_idx_fp:
        word_to_idx = pickle.load(w_idx_fp)

    # Load Data and set experiment specific parameters
    if experiment == "sentiment":
        data = SentimentDataset(constants.SENTIMENT_DATA_PATH, constants.SENTIMENT_LABELS_PATH, constants.MAX_LEN,
                                constants.TRAIN_RATIO)
        net = SentimentNet(word_to_idx, embedding_size, data.adj_list, qw_network, constants.MAX_LEN, learn_coin, learn_amps, ongpu, walk_length)
        criterion = nn.NLLLoss()

    opt = optim.Adam(net.parameters())

    running_loss = 0.0
    print "Beginning Traning.."
    print "Epoch Batch Loss"
    besttest = 1000
    if logging:
        f = open('results/' +
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "Results-" + experiment + "-" + qw_network + "-" + str(
                    learn_amps) + "-" + str(
                    learn_coin) + "-" + str(walk_length) + "-" + str(walkers) + ".log", "a+")
    dloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    patience = 0
    los = []
    for iter in range(epochs):
        for i_batch, (x, y) in enumerate(dloader):
            if ongpu:
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
            else:
                x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
            opt.zero_grad()  # zero the gradient buffers
            output = net.forward(x)
            loss = criterion(output, y)
            loss.backward()
            opt.step()

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

        x, y = data.get_test_set()
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        out = net.forward(x)

        testloss = criterion(out, y).data.cpu().numpy()[0]
        los.append(testloss)
        print "Test Loss: ", testloss
        print "Test Loss per Node:", testloss / len(data.adj)
        if logging:
            f.write("iter: " + str(iter) + " Test Loss: " + str(testloss) + "\n")
        if testloss < besttest:
            patience = 0
            besttest = testloss
        if patience == 8:
            break
    if logging:
        f.close()
    return besttest, los


if __name__ == "__main__":
    pass
