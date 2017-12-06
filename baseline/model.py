import torch.nn as nn
import torch.optim as optim

from . import layers



class Baseline():
    def __init__(self, word_to_idx):
        self.use_cuda = False
        self.embedding_size = 300
        self.encoder_hsz = 256
        self.num_rnn_layers = 128
        self.learning_rate = 0.1
        self.rnn_dropout = 0.1

        self.dict = word_to_idx
        self.NULL_IDX = 0

        self.embeds = nn.Embedding(len(self.dict), self.embedding_size, padding_idx=self.NULL_IDX,
                                   scale_grad_by_freq=True)
        self.encoder = layers.RNNEncoder(self.embedding_size, self.encoder_hsz, self.num_rnn_layers, self.rnn_dropout)
        self.mlp = nn.Linear(self.encoder_hsz, 2)

        self.optims = {
            'encoder': optim.Adam(self.encoder.parameters(), lr=self.learning_rate),
            'mlp': optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        }
        self.criterion = nn.NLLLoss()

    def zero_grad(self):
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        for optimizer in self.optims.values():
            optimizer.step()



    def train(self, x, y):
        h0 = self.encoder.initHidden()
        xes = self.embeds(x)
        output, hn = self.encoder.forward(xes, h0)
        scores = self.mlp.forward(output)
        _, preds = scores.max(1)

        loss = self.criterion(preds, y)
        loss.backward()
        self.update_params()


    def test(self, x):
        h0 = self.encoder.initHidden()
        xes = self.embeds(x)
        output, hn = self.encoder.forward(xes, h0)
        scores = self.mlp.forward(output)
        _, preds = scores.max(1)

        return preds


