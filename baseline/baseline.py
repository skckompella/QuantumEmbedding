import torch.nn as nn

from . import layers


class Baseline():
    def __init__(self):
        self.use_cuda = False
        self.embedding_size = 300
        self.encoder_hsz = 256
        self.num_rnn_layers = 128
        self.learning_rate = 0.1
        self.rnn_dropout = 0.1

        self.dict = None #TODO- dictionary
        self.NULL_IDX = None #TODO - Fill in the null idx

        self.embeds = nn.Embedding(len(self.dict), self.embedding_size, padding_idx=self.NULL_IDX,
                                   scale_grad_by_freq=True)
        self.encoder = layers.RNNEncoder(self.embedding_size, self.encoder_hsz, self.num_rnn_layers, self.rnn_dropout)
        self.mlp = nn.Linear(self.encoder_hsz, 2)
        self.criterion = nn.NLLLoss()



    def train(self):
        pass

    def test(self):
        pass

    def batchify(self):
        pass

