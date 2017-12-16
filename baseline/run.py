import cPickle as pickle

import numpy as np
import torch
from torch.autograd import Variable

from common import utils, constants
import datasets
from network import Baseline


def run_baseline():
    sentiment_data = datasets.SentimentDataset(constants.SENTIMENT_DATA_PATH, constants.SENTIMENT_LABELS_PATH, constants.MAX_LEN)

    train_dataset = sentiment_data.get_train_set()
    test_dataset = sentiment_data.get_test_set()
    train_data, train_labels = Variable(torch.from_numpy(train_dataset[0])), Variable(
        torch.from_numpy(train_dataset[1].flatten()))
    test_data, test_labels = Variable(torch.from_numpy(test_dataset[0])), Variable(torch.from_numpy(test_dataset[1].flatten()))

    with open(constants.WORD_TO_IDX_PATH, "rb") as w_idx_fp:
        word_to_idx = pickle.load(w_idx_fp)

    batch_size = int((train_data.size()[0] + test_data.size()[0]) * constants.BATCH_RATIO)

    model = Baseline(word_to_idx, batch_size, constants.MAX_LEN)
    for epoch in range(constants.NUM_EPOCHS):
        np.random.seed()
        batch_indices = torch.from_numpy(np.random.randint(0, train_data.size()[0], batch_size))

        print "-----------------------"
        train_loss, train_predictions = model.train(train_data[batch_indices], train_labels[batch_indices])
        print "Epoch: %d, Training Loss: %f, Training Accuracy: %f" % (epoch, train_loss.data[0], utils.get_accuracy(
            train_predictions, train_labels[batch_indices]))

    test_predictions = model.test(test_data)
    print "Test Accuracy: %f" % utils.get_accuracy(test_predictions, test_labels)

    pass


def main():
    run_baseline()


if __name__ == '__main__':
    main()
