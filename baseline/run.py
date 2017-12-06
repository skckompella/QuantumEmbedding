from model import Baseline
from preprocessing import utils
from preprocessing import sentiment_preprocessing as sp
import torch
from torch.autograd import Variable

DATA_FILE = "../data/len12_sents,txt"
LABEL_FILE = "../data/len12_sentlabels.txt"

MAX_LEN = 12
NUM_EPOCHS = 10


def main():
    sentences, labels = utils.read_data(DATA_FILE, LABEL_FILE)
    vocab = sp.get_vocabulary(sentences)
    word_to_idx = dict()
    word_to_idx["__PAD__"] = 0
    # for i in range(len(vocab)):
    #     word_to_idx[vocab[i]] = i+1

    for idx, w in enumerate(vocab):
        word_to_idx[w] = idx+1




    X = torch.LongTensor(10, MAX_LEN).fill_(0) #TODO- change to batch size

    for i in range(10):  #TODO- Implement batching
        for j in range(len(sentences[i])):
            X[i][j] = word_to_idx[sentences[i][j]]
    X = Variable(X)
    Y = Variable(torch.LongTensor(labels[:10])) #TODO- change to batch size

    # print(X.shape)
    model = Baseline(word_to_idx, X.shape[0], MAX_LEN)
    for e in range(NUM_EPOCHS):
        print("-----------------------" )
        print("  Epoch %d" % e)
        model.train(X, Y)
        print("-----------------------")


if __name__ == '__main__':
    main()