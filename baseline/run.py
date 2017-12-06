from .model import Baseline
from preprocessing import sentiment_preprocessing as sp
import torch

DATA_FILE = "../data/combined_data_no_oov.txt"
LABEL_FILE = "../data/data_labels.txt"

MAX_LEN = 20


def main():
    sentences, labels = sp.get_sentences_and_labels(DATA_FILE, LABEL_FILE)
    vocab = sp.get_vocabulary(sentences)

    word_to_idx = dict()
    word_to_idx["__PAD__"] = 0
    # for i in range(len(vocab)):
    #     word_to_idx[vocab[i]] = i+1

    for w, idx in enumerate(vocab):
        word_to_idx[w] = idx+1

    model = Baseline(word_to_idx)

    X = torch.LongTensor()










if __name__ == '__main__':
    main()