# coding: utf-8

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TrajRNN(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajRNN, self).__init__()
        self.loc_size = parameters.loc_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.loc_emb_size = parameters.loc_emb_size
        self.hidden_size = parameters.hidden_size
        self.rnn_type = parameters.rnn_type
        self.input_size = self.tim_emb_size + self.loc_emb_size

        self.emb_loc = nn.Embedding(self.loc_size + 1, self.loc_emb_size, padding_idx= 0)
        self.emb_tim = nn.Embedding(self.tim_size + 1, self.tim_emb_size, padding_idx= 0)
        self.concat_layer = nn.Linear(self.hidden_size * 5, self.hidden_size)
        self.rnn = nn.GRU(self.input_size, self.hidden_size, 1)
        self.output_layer = nn.Linear(self.hidden_size, 1)

        self.dropout = nn.Dropout(p=parameters.dropout_p)
        #self.init_weights()


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


    def forward(self, user):
        batch_size = user['mon_tim'].size()[-1]
        mon_tim = self.emb_tim(user['mon_tim'].cuda())
        mon_loc = self.emb_loc(user['mon_loc'].cuda())
        mon = torch.cat((mon_tim, mon_loc),2)
        tue_tim = self.emb_tim(user['tue_tim'].cuda())
        tue_loc = self.emb_loc(user['tue_loc'].cuda())
        tue = torch.cat((tue_tim, tue_loc), 2)
        wed_tim = self.emb_tim(user['wed_tim'].cuda())
        wed_loc = self.emb_loc(user['wed_loc'].cuda())
        wed = torch.cat((wed_tim, wed_loc), 2)
        thu_tim = self.emb_tim(user['thu_tim'].cuda())
        thu_loc = self.emb_loc(user['thu_loc'].cuda())
        thu = torch.cat((thu_tim, thu_loc), 2)
        sat_tim = self.emb_tim(user['sat_tim'].cuda())
        sat_loc = self.emb_loc(user['sat_loc'].cuda())
        sat = torch.cat((sat_tim, sat_loc), 2)

        mon_h = torch.zeros(1, batch_size, self.hidden_size, requires_grad=True).cuda()
        tue_h = torch.zeros(1, batch_size, self.hidden_size, requires_grad=True).cuda()
        wed_h = torch.zeros(1, batch_size, self.hidden_size, requires_grad=True).cuda()
        thu_h = torch.zeros(1, batch_size, self.hidden_size, requires_grad=True).cuda()
        sat_h = torch.zeros(1, batch_size, self.hidden_size, requires_grad=True).cuda()
        temp, out_mon = self.rnn(mon, mon_h)
        temp, out_tue = self.rnn(tue, tue_h)
        temp, out_wed = self.rnn(wed, wed_h)
        temp, out_thu = self.rnn(thu, thu_h)
        temp, out_sat = self.rnn(sat, sat_h)
        out = torch.cat([out_mon,out_tue,out_wed,out_thu,out_sat],dim=2)
        out = out.squeeze(0)
        out = self.dropout(out)
        out = self.concat_layer(out)
        out = torch.tanh(out)
        out = self.output_layer(out)
        out = torch.sigmoid(out)
        out = out.squeeze()

        return out #[batch_size]
