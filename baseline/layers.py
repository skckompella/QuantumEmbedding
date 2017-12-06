import torch
from torch.autograd import Variable
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(RNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        return output, hidden

    def initHidden(self, use_cuda=False):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureExtractor, self).__init__()

        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)


    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        return out3


