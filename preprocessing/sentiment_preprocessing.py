from nltk import word_tokenize
import re


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


def main():

    # get_data_less_than_length()
    pass


if __name__ == '__main__':
    main()
