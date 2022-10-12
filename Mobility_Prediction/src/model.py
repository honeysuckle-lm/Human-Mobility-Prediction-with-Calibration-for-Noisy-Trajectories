# coding: utf-8

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class TrajRNN(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajRNN, self).__init__()
        self.loc_size = parameters.loc_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.loc_emb_size = parameters.loc_emb_size
        self.hidden_size = parameters.hidden_size
        self.batch_size = parameters.batch_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size + 1, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size + 1, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        #self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):

        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim):
        h1 = torch.zeros(1, tim.size()[-1], self.hidden_size, requires_grad=True)
        c1 = torch.zeros(1, tim.size()[-1], self.hidden_size, requires_grad=True)
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        tim_emb = self.emb_tim(tim) # embedding之后 tim_emb是[sequence length, batch size, embedding size]维度
        loc_emb = self.emb_loc(loc)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))

        out = F.selu(out)
        out = self.dropout(out)
        y = self.fc(out)
        score = F.log_softmax(y, dim=2)
        return score
