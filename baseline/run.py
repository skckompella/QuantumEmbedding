import cPickle as pickle

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import logging

from common import utils, constants
import datasets
from network import Baseline


def run_baseline():
    logging.basicConfig(filename=constants.LOGS_DIR + '/baseline/baseline_rnn.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')

    sentiment_data = datasets.SentimentDataset(constants.SENTIMENT_DATA_PATH, constants.SENTIMENT_LABELS_PATH,
                                               constants.MAX_LEN, constants.TRAIN_RATIO, constants.VALID_RATIO)
    sentiment_loader = DataLoader(sentiment_data, batch_size=constants.BATCH_SIZE, shuffle=True, num_workers=1)

    test_data, test_labels = sentiment_data.get_test_set()
    if constants.GPU:
        test_data, test_labels = Variable(test_data.cuda(), requires_grad=False), Variable(test_labels.cuda(),
                                                                                           requires_grad=False)
    else:
        test_data, test_labels = Variable(test_data, requires_grad=False), Variable(test_labels, requires_grad=False)

    with open(constants.WORD_TO_IDX_PATH, "rb") as w_idx_fp:
        word_to_idx = pickle.load(w_idx_fp)

    model = Baseline(word_to_idx, constants.MAX_LEN)
    for epoch in range(constants.NUM_EPOCHS):
        running_loss = 0.0
        running_acc = 0.0
        for batch, (x, y) in enumerate(sentiment_loader):
            if constants.GPU:
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
            else:
                x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

            train_loss, train_predictions = model.train(x, y)

            if batch % 10 == 0:
                message = "Epoch: %5d Batch: %5d Loss: %.3f Accuracy: %.3f" % (
                    epoch + 1, int(batch) + 1, running_loss / 10, running_acc / 10)
                logging.info(message)
                print message
                running_loss = 0.0
                running_acc = 0.0

            running_loss += train_loss.data[0]
            running_acc += utils.get_accuracy(train_predictions, y)

        test_predictions = model.test(test_data)
        message = "Test Accuracy: %f" % utils.get_accuracy(test_predictions, test_labels)
        logging.info(message)
        print message

    pass


def main():
    run_baseline()


if __name__ == '__main__':
    main()
