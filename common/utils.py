import numpy as np


def write_data_to_file(data, file_name, join=False):
    """

    :param data: some data in the form of a list
    :param file_name: file where the above data has to be stored
    :param join: if the list is single or double dimensional array
    :return:
    """
    with open(file_name, "w") as fp:
        for line in data:
            if join:
                fp.write(" ".join(line) + "\n")
            else:
                fp.write(line + "\n")


def read_data(data_file, label_file):
    """

    :param data_file: File containing the data
    :param label_file: File to store labels of processed sentences
    :return: list of sentences, where each sentence is a list of words
    """

    sentences = []
    labels = []

    with open(data_file, "r") as data_fp:
        for line in data_fp:
            sentence = line.strip().split()
            sentences.append(sentence)

    with open(label_file, "r") as label_fp:
        for line in label_fp:
            labels.append(int(line.strip().split()[0]))

    return sentences, labels


def get_accuracy(predictions, labels):
    """
    Returns the accuracy of the predictions, provided the labels
    :param predictions: predictions from the model
    :param labels: gold labels for the respective predictions
    :return:
    """

    return float(sum(predictions == labels).data[0]) / labels.size()[0]


def get_sentiment_adjacency_matrix(max_len):

    adj = np.zeros(shape=(max_len, max_len))

    for i in range(max_len):
        prev_idx = max(0, i-1)
        next_idx = min(max_len-1, i+1)
        adj[i, prev_idx] = 1
        adj[i, next_idx] = 1

    return adj


if __name__ == '__main__':
    print get_sentiment_adjacency_matrix(12)
