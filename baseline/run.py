from model import Baseline
from preprocessing import utils
from preprocessing import sentiment_preprocessing as sp
import torch
from torch.autograd import Variable
import numpy as np

DATA_FILE = "../data/len12_sents,txt"
LABEL_FILE = "../data/len12_sentlabels.txt"

MAX_LEN = 12
NUM_EPOCHS = 50
TRAIN_RATIO = 0.9
BATCH_RATIO = 0.1


def get_accuracy(predictions, labels):
    return float(sum(predictions == labels).data[0]) / labels.size()[0]


def get_train_test_split(data, labels, word_to_idx, train_ratio, max_len):
    train_len = int(len(data) * train_ratio)

    train_data = torch.LongTensor(train_len, max_len).fill_(0)
    test_data = torch.LongTensor(len(data) - train_len, max_len).fill_(0)
    train_labels = torch.LongTensor(train_len).fill_(0)
    test_labels = torch.LongTensor(len(data) - train_len).fill_(0)

    for i in range(len(data)):
        if i < train_len:
            train_labels[i] = labels[i]
        else:
            test_labels[i - train_len] = labels[i]
        for j in range(len(data[i])):
            if i < train_len:
                train_data[i][j] = word_to_idx[data[i][j]]
            else:
                test_data[i - train_len][j] = word_to_idx[data[i][j]]

    return Variable(train_data), Variable(train_labels), Variable(test_data), Variable(test_labels)


def main():
    sentences, labels = utils.read_data(DATA_FILE, LABEL_FILE)
    vocab = sp.get_vocabulary(sentences)
    word_to_idx = dict()
    word_to_idx["__PAD__"] = 0
    BATCH_SIZE = int(len(sentences) * BATCH_RATIO)

    for idx, w in enumerate(vocab):
        word_to_idx[w] = idx + 1

    train_data, train_labels, test_data, test_labels = get_train_test_split(sentences, labels, word_to_idx, TRAIN_RATIO,
                                                                            MAX_LEN)

    model = Baseline(word_to_idx, BATCH_SIZE, MAX_LEN)
    for epoch in range(NUM_EPOCHS):
        np.random.seed()
        batch_indices = torch.from_numpy(np.random.randint(0, train_data.size()[0], BATCH_SIZE))

        print "-----------------------"
        train_loss, train_predictions = model.train(train_data[batch_indices], train_labels[batch_indices])
        print "Epoch: %d, Training Loss: %f, Training Accuracy: %f" % (epoch, train_loss.data[0], get_accuracy(
            train_predictions, train_labels[batch_indices]))

    test_predictions = model.test(test_data)
    print "Test Accuracy: %f" % get_accuracy(test_predictions, test_labels)


if __name__ == '__main__':
    main()
