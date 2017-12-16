from __future__ import division
from common import constants
import torch.nn as nn
import torch.optim as optim
import layers


class Baseline:
    def __init__(self, word_to_idx, batch_size, seq_max_len):
        self.use_cuda = False
        self.embedding_size = 300
        self.encoder_hsz = self.embedding_size
        self.num_rnn_layers = 2
        self.learning_rate = 0.01
        self.rnn_dropout = 0.1
        self.batch_size = batch_size
        self.seq_max_len = seq_max_len
        self.dict = word_to_idx
        self.NULL_IDX = 0

        self.embeds = nn.Embedding(len(self.dict), self.embedding_size, padding_idx=self.NULL_IDX)
        # self.embeds = nn.Linear(constants.MAX_LEN, self.embedding_size)
        self.encoder = layers.RNNEncoder(self.embedding_size, self.encoder_hsz, self.num_rnn_layers, self.rnn_dropout)
        self.mlp = layers.FeatureExtractor(self.encoder_hsz*self.seq_max_len, 2)

        self.optims = {
            'encoder': optim.Adam(self.encoder.parameters()),
            'mlp': optim.Adam(self.mlp.parameters())
        }
        self.criterion = nn.NLLLoss()

    def zero_grad(self):
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        for optimizer in self.optims.values():
            optimizer.step()

    def train(self, x, y):
        self.zero_grad()
        h0 = self.encoder.initHidden()
        xes = self.embeds(x)
        output, hn = self.encoder.forward(xes, h0)
        mlp_in = output.contiguous().view(-1, self.encoder_hsz*self.seq_max_len)
        scores = self.mlp.forward(mlp_in)
        _, preds = scores.max(1)
        loss = self.criterion(scores, y)
        loss.backward()
        self.update_params()
        return loss, preds

    def test(self, x):
        h0 = self.encoder.initHidden()
        xes = self.embeds(x)
        output, hn = self.encoder.forward(xes, h0)
        mlp_in = output.contiguous().view(-1, self.encoder_hsz*self.seq_max_len)
        scores = self.mlp.forward(mlp_in)
        _, preds = scores.max(1)
        return preds


