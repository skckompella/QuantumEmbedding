from model import Baseline
from preprocessing import utils
from preprocessing import sentiment_preprocessing as sp
import torch
from torch.autograd import Variable
import numpy as np

DATA_FILE = "../data/len12_sents,txt"
LABEL_FILE = "../data/len12_sentlabels.txt"

MAX_LEN = 12
NUM_EPOCHS = 100
BATCH_SIZE = 128
TRAIN_RATIO = 0.8


def get_accuracy(predictions, labels):
    return float(sum(predictions == labels).data[0]) / labels.size()[0]


def main():
    sentences, labels = utils.read_data(DATA_FILE, LABEL_FILE)
    vocab = sp.get_vocabulary(sentences)
    word_to_idx = dict()
    word_to_idx["__PAD__"] = 0

    for idx, w in enumerate(vocab):
        word_to_idx[w] = idx + 1

    X = torch.LongTensor(len(sentences), MAX_LEN).fill_(0)

    for i in range(len(sentences)):
        # offset = MAX_LEN - len(sentences[i])
        for j in range(len(sentences[i])):
            X[i][j] = word_to_idx[sentences[i][j]]

    X = Variable(X)
    Y = Variable(torch.LongTensor(labels))

    model = Baseline(word_to_idx, BATCH_SIZE, MAX_LEN)
    for epoch in range(NUM_EPOCHS):
        np.random.seed()
        batch_indices = torch.from_numpy(np.random.randint(0, X.size()[0], BATCH_SIZE))

        print "-----------------------"
        train_loss, train_predictions = model.train(X[batch_indices], Y[batch_indices])
        print "Epoch: %d, Training Loss: %f, Training Accuracy: %f" % (epoch, train_loss.data[0], get_accuracy(
            train_predictions, Y[batch_indices]))


if __name__ == '__main__':
    main()
