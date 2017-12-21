from __future__ import division
from datetime import datetime

import torch.optim as optim
from torch.utils.data import DataLoader

from common import constants
from layers import *
from model.datasets import *
import pickle
import logging


class SentimentNet(nn.Module):
    def __init__(self, word_to_idx, embedding_size,
                 adj, qw_network="qw", num_walkers=None, learn_coin=True, learn_amps=False, onGPU=False,
                 time_steps=1):
        super(SentimentNet, self).__init__()
        self.word_to_idx = word_to_idx
        # print "length", len(self.word_to_idx)
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
        x = x.double()
        x = self.qw.forward(x)
        # print x[0][3]
        x = x.sum(1).float()
        # exit()
        x = self.mlp.forward(x)
        return x


def doExperiment(experiment, qw_network, embedding_size=128, do_logging=False, epochs=32, batch_size=16,
                 ongpu=True, learn_amps=True, learn_coin=True, walk_length=4,
                 train_ratio=0.5, feature_dropout=0.0, walkers=None,
                 shuffleEx=True, shuffleNodes=True):
    logging.basicConfig(filename=constants.LOGS_DIR + "/qw/qw" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".log",
                        level=logging.INFO,
                        format="%(asctime)s %(message)s")

    if do_logging:
        message = "Starting Experiment with Parameters:", [experiment, qw_network, walk_length, learn_amps, learn_coin]
        logging.info(message)

    with open(constants.WORD_TO_IDX_PATH, "rb") as w_idx_fp:
        word_to_idx = pickle.load(w_idx_fp)

    data = SentimentDataset(constants.SENTIMENT_DATA_PATH, constants.SENTIMENT_LABELS_PATH, constants.MAX_LEN,
                            constants.TRAIN_RATIO, constants.VALID_RATIO)
    net = SentimentNet(word_to_idx, embedding_size, data.adj_list, qw_network, constants.MAX_LEN, learn_coin,
                       learn_amps, ongpu, walk_length)
    criterion = nn.NLLLoss()

    opt = optim.RMSprop(net.parameters(), lr=1e-5)

    bestvalid = 1000
    dloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    los = []

    for iter in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i_batch, (x, y) in enumerate(dloader):
            if ongpu:
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
            else:
                x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

            opt.zero_grad()
            output = net.forward(x)
            loss = criterion(output, y)
            loss.backward()
            opt.step()

            running_loss += loss.data[0]
            running_acc += utils.get_accuracy(output.max(1)[1], y)

        if do_logging:
            message = "Train: Epoch: %d Loss: %.3f Accuracy: %.3f" % (
                iter + 1, running_loss / len(dloader), running_acc / len(dloader))
            logging.info(message)
        x, y = data.get_valid_set()
        if ongpu:
            x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
        else:
            x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

        out = net.forward(x)
        validation_loss = criterion(out, y).data[0]
        los.append(validation_loss)
        _, preds = out.max(1)

        if do_logging:
            message = "Validation Loss: %.3f, Validation Accuracy: %f" % (validation_loss, utils.get_accuracy(preds, y))
            logging.info(message)

        if validation_loss < bestvalid:
            bestvalid = validation_loss

    exit()
    x_test, y_test = data.get_test_set()
    if ongpu:
        x_test, y_test = Variable(x_test.cuda(), requires_grad=False), Variable(y_test.cuda(), requires_grad=False)
    else:
        x_test, y_test = Variable(x_test, requires_grad=False), Variable(y_test, requires_grad=False)

    out = net.forward(x_test)
    _, preds = out.max(1)
    if do_logging:
        message = "Test Accuracy: %f" % (utils.get_accuracy(preds, y_test))
        logging.info(message)

    return bestvalid, los


if __name__ == "__main__":
    pass
