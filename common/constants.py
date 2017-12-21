MAX_LEN = 70
NUM_EPOCHS = 50
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
BATCH_SIZE = 16
GPU = False

ROOT_DIR = "../"
PROCESSED_DATA_DIR = ROOT_DIR + "/data/processed"
TEXT_CORPORA_DIR = ROOT_DIR + "/data/text_corpora"
LOGS_DIR = ROOT_DIR + "/logs"

SENTIMENT_DATA_PATH = PROCESSED_DATA_DIR + "/sentiment_data_" + str(MAX_LEN) + ".npy"
SENTIMENT_LABELS_PATH = PROCESSED_DATA_DIR + "/sentiment_labels_" + str(MAX_LEN) + ".npy"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR + "/train_data.npy"
TEST_DATA_PATH = PROCESSED_DATA_DIR + "/test_data.npy"
TRAIN_LABELS_PATH = PROCESSED_DATA_DIR + "/train_labels.npy"
TEST_LABELS_PATH = PROCESSED_DATA_DIR + "/test_labels.npy"
WORD_TO_IDX_PATH = PROCESSED_DATA_DIR + "/word_to_idx"

INPUT_DATA_FILES = (TEXT_CORPORA_DIR + "/data_sentences.txt", TEXT_CORPORA_DIR + "/data_labels.txt")
PROCESSED_DATA_FILES = (TEXT_CORPORA_DIR + "/data_sentences_len_" + str(MAX_LEN) + ".txt",
                        TEXT_CORPORA_DIR + "/data_labels_len_" + str(MAX_LEN) + ".txt")
