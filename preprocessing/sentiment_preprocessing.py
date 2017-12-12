from nltk import word_tokenize
from common import utils, constants
import re
import numpy as np
import cPickle as pickle


def get_sentences_and_labels(data_file, label_file):
    """

    :param data_file: File containing the data
    :param label_file: File to store labels of processed sentences
    :return: list of sentences, where each sentence is a list of words
    """

    sentences = []
    labels = []

    with open(data_file, "r") as data_fp:
        for line in data_fp:
            sentence = line.strip().split("\t")[0].lower()
            current_sentence = []
            for word in word_tokenize(sentence):
                if not bool(re.search(r'\d', word)) and len(word) > 1:
                    for filtered_word in re.split('\W+', word):
                        current_sentence.append(filtered_word)
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(line.strip().split("\t")[1])

    with open(label_file, "w") as label_fp:
        for label in labels:
            label_fp.write(label + "\n")

    return sentences


def get_vocabulary(sentences):
    """

    :param sentences: list of sentences, where each sentence is a list of words
    :return: set object containing vocabulary
    """
    vocabulary = set()

    for sentence in sentences:
        for word in sentence:
            vocabulary.add(word)

    return vocabulary


def remove_oov_words(sentences, embeddings_file):
    """

    :param sentences:  list of sentences, where each sentence is a list of words
    :param embeddings_file: file containing words in the vocabulary, along with their embeddings
    :return:
    """

    vocab_with_embeddings = set()

    with open(embeddings_file, "r") as emb_fp:
        for line in emb_fp:
            vocab_with_embeddings.add(line.strip().split(" ")[0])

    for sentence in sentences:
        for word in [oov_word for oov_word in sentence if oov_word not in vocab_with_embeddings]:
            sentence.remove(word)

    return sentences


def get_data_less_than_length(num_len, sentences_file, labels_file, subset_sentences_file, subset_label_file):
    """

    :param num_len: desired max length
    :param sentences_file: input file containing the sentences
    :param labels_file: input file containing the labels
    :param subset_sentences_file: file where the sentences with length less than num_len will be stored
    :param subset_label_file: file where labels of above sentences will be stored
    :return:
    """

    count = 0
    labels = []
    subset_sentences = []
    subset_labels = []

    with open(labels_file, "r") as lb_fp:
        for line in lb_fp:
            labels.append(line.strip().split(" ")[0])

    counter = 0
    with open(sentences_file, "r") as fp:
        for line in fp:
            sentence = line.strip().split()
            if len(sentence) <= num_len:
                count += 1
                subset_sentences.append(sentence)
                subset_labels.append(labels[counter])
            counter += 1

    assert count == len(subset_sentences) == len(subset_labels)

    with open(subset_sentences_file, "w") as s_fp:
        for sentence in subset_sentences:
            s_fp.write(" ".join(sentence) + "\n")

    with open(subset_label_file, "w") as l_fp:
        for label in subset_labels:
            l_fp.write(label + "\n")

    return count


def prepare_datasets(input_data_files, max_len):
    """
    Reads the input data, and prepares the train, and test files
    :param input_data_files: tuple containing (<path_to_data_file>, <path_to_labels_file>)
    :param max_len: maximum length of the sentence (number of words) in input data
    :return:
    """

    sentences, sentence_labels = utils.read_data(input_data_files[0], input_data_files[1])
    vocab = get_vocabulary(sentences)
    word_to_idx = dict()
    word_to_idx["__PAD__"] = 0

    data = np.zeros(shape=(len(sentences), max_len), dtype=long)
    labels = np.zeros(shape=len(sentences), dtype=long)

    for idx, w in enumerate(vocab):
        word_to_idx[w] = idx + 1

    for i in range(len(sentences)):
        labels[i] = sentence_labels[i]
        offset = max_len - len(sentences[i])
        for j in range(len(sentences[i])):
            data[i][offset+j] = word_to_idx[sentences[i][j]]

    np.save(constants.DATA_PATH, data)
    np.save(constants.LABELS_PATH, labels)

    with open(constants.WORD_TO_IDX_PATH, "wb") as w_idx_fp:
        pickle.dump(word_to_idx, w_idx_fp)

    print "saved the train data, labels, and test data, labels"

    pass


def main():

    input_data_files = ("../data/text_corpora/len12_sents.txt", "../data/text_corpora/len12_sentlabels.txt")
    max_length, train_test_ratio = 12, 0.9
    prepare_datasets(input_data_files, max_length, train_test_ratio)

    pass


if __name__ == '__main__':
    main()
