# coding: utf-8
from typing import Dict, List

import torch
import torch.nn
import numpy as np
import random
import json
import time
from collections import deque, Counter
import torch.nn.functional as F
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.data as data
import sys
import gc

class RnnParameterData(object):
    def __init__(self, loc_emb_size,tim_emb_size, hidden_size,
                 lr, batch_size, lr_step, lr_decay, dropout_p, L2, clip, optim, epoch_max, rnn_type,
                 data_path, save_path, data_name='Oidd'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        print('*' * 15 + 'start loading' + '*' * 15)
        with open(self.data_path + self.data_name + '.json', 'r') as f:
            data = json.load(f)

        self.data_neural = data['data_neural']  # data_filter: 每个用户对应的轨迹(session)

        self.tim_size = 96
        self.loc_size = data['vid_len'] + 1

        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.lr = lr
        self.batch_size = batch_size
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip
        self.rnn_type = rnn_type
        del data
        gc.collect()


def generate_input_trajectory(data_neural, batch_size):
    data_train = []
    count = 0
    data_weighting = {}
    for u in data_neural.keys():
        if len(data_neural[u]['sessions']) < 7 :
            continue
        sessions = data_neural[u]['sessions']
        count += 1
        data_weighting[str(count)] = data_neural[u]
        session = sessions[0]
        mon_tim = np.array([s[1] + 1 for s in session])
        mon_loc = np.array([s[0] + 1 for s in session])
        session = sessions[1]
        tue_tim = np.array([s[1] + 1 for s in session])
        tue_loc = np.array([s[0] + 1 for s in session])
        session = sessions[2]
        wed_tim = np.array([s[1] + 1 for s in session])
        wed_loc = np.array([s[0] + 1 for s in session])
        session = sessions[3]
        thu_tim = np.array([s[1] + 1 for s in session])
        thu_loc = np.array([s[0] + 1 for s in session])
        session = sessions[5]
        sat_tim = np.array([s[1] + 1 for s in session])
        sat_loc = np.array([s[0] + 1 for s in session])
        trace = {}
        trace['mon_tim'] = torch.LongTensor(mon_tim)
        trace['mon_loc'] = torch.LongTensor(mon_loc)
        trace['tue_tim'] = torch.LongTensor(tue_tim)
        trace['tue_loc'] = torch.LongTensor(tue_loc)
        trace['wed_tim'] = torch.LongTensor(wed_tim)
        trace['wed_loc'] = torch.LongTensor(wed_loc)
        trace['thu_tim'] = torch.LongTensor(thu_tim)
        trace['thu_loc'] = torch.LongTensor(thu_loc)
        trace['sat_tim'] = torch.LongTensor(sat_tim)
        trace['sat_loc'] = torch.LongTensor(sat_loc)
        trace['user_id'] = count
        data_train.append(trace)

    print('*' * 15, 'End Trajectory Transformation', '*' * 15)
    print('*' * 15, 'Total User:', count, '*' * 15)
    data = MyData(data_train)
    return data_weighting, DataLoader(data, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=16, pin_memory=True)


class MyData(data.Dataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        negative_id = idx
        while negative_id == idx:
            negative_id = random.choice(range(self.__len__()))
        sample_id = ['mon', 'tue', 'wed', 'thu', 'sat']
        negative_sample = {}
        for key in sample_id:
            negative_sample[key + '_tim'] = self.data_source[negative_id][key + '_tim'].detach()
            negative_sample[key + '_loc'] = self.data_source[negative_id][key + '_loc'].detach()

        for id in random.sample(range(0,5), np.random.randint(3,5)):
            key = sample_id[id]
            negative_sample[key + '_tim'] = self.data_source[idx][key + '_tim'].detach()
            negative_sample[key + '_loc'] = self.data_source[idx][key + '_loc'].detach()
        return self.data_source[idx], negative_sample

def collate_fn(data):
    #data.sort(key=lambda x: len(x['target']), reverse=True)
    mon_tim = [s['mon_tim'] for s,t in data]
    mon_loc = [s['mon_loc'] for s,t in data]
    tue_tim = [s['tue_tim'] for s,t in data]
    tue_loc = [s['tue_loc'] for s,t in data]
    wed_tim = [s['wed_tim'] for s,t in data]
    wed_loc = [s['wed_loc'] for s,t in data]
    thu_tim = [s['thu_tim'] for s,t in data]
    thu_loc = [s['thu_loc'] for s,t in data]
    sat_tim = [s['sat_tim'] for s,t in data]
    sat_loc = [s['sat_loc'] for s,t in data]
    positive_sample = {}
    positive_sample['mon_tim'] = torch.nn.utils.rnn.pad_sequence(mon_tim, batch_first=False, padding_value=0)
    positive_sample['mon_loc'] = torch.nn.utils.rnn.pad_sequence(mon_loc, batch_first=False, padding_value=0)
    positive_sample['tue_tim'] = torch.nn.utils.rnn.pad_sequence(tue_tim, batch_first=False, padding_value=0)
    positive_sample['tue_loc'] = torch.nn.utils.rnn.pad_sequence(tue_loc, batch_first=False, padding_value=0)
    positive_sample['wed_tim'] = torch.nn.utils.rnn.pad_sequence(wed_tim, batch_first=False, padding_value=0)
    positive_sample['wed_loc'] = torch.nn.utils.rnn.pad_sequence(wed_loc, batch_first=False, padding_value=0)
    positive_sample['thu_tim'] = torch.nn.utils.rnn.pad_sequence(thu_tim, batch_first=False, padding_value=0)
    positive_sample['thu_loc'] = torch.nn.utils.rnn.pad_sequence(thu_loc, batch_first=False, padding_value=0)
    positive_sample['sat_tim'] = torch.nn.utils.rnn.pad_sequence(sat_tim, batch_first=False, padding_value=0)
    positive_sample['sat_loc'] = torch.nn.utils.rnn.pad_sequence(sat_loc, batch_first=False, padding_value=0)
    positive_sample['user_id'] = [s['user_id'] for s,t in data]

    mon_tim = [t['mon_tim'] for s, t in data]
    mon_loc = [t['mon_loc'] for s, t in data]
    tue_tim = [t['tue_tim'] for s, t in data]
    tue_loc = [t['tue_loc'] for s, t in data]
    wed_tim = [t['wed_tim'] for s, t in data]
    wed_loc = [t['wed_loc'] for s, t in data]
    thu_tim = [t['thu_tim'] for s, t in data]
    thu_loc = [t['thu_loc'] for s, t in data]
    sat_tim = [t['sat_tim'] for s, t in data]
    sat_loc = [t['sat_loc'] for s, t in data]
    negative_sample = {}
    negative_sample['mon_tim'] = torch.nn.utils.rnn.pad_sequence(mon_tim, batch_first=False, padding_value=0)
    negative_sample['mon_loc'] = torch.nn.utils.rnn.pad_sequence(mon_loc, batch_first=False, padding_value=0)
    negative_sample['tue_tim'] = torch.nn.utils.rnn.pad_sequence(tue_tim, batch_first=False, padding_value=0)
    negative_sample['tue_loc'] = torch.nn.utils.rnn.pad_sequence(tue_loc, batch_first=False, padding_value=0)
    negative_sample['wed_tim'] = torch.nn.utils.rnn.pad_sequence(wed_tim, batch_first=False, padding_value=0)
    negative_sample['wed_loc'] = torch.nn.utils.rnn.pad_sequence(wed_loc, batch_first=False, padding_value=0)
    negative_sample['thu_tim'] = torch.nn.utils.rnn.pad_sequence(thu_tim, batch_first=False, padding_value=0)
    negative_sample['thu_loc'] = torch.nn.utils.rnn.pad_sequence(thu_loc, batch_first=False, padding_value=0)
    negative_sample['sat_tim'] = torch.nn.utils.rnn.pad_sequence(sat_tim, batch_first=False, padding_value=0)
    negative_sample['sat_loc'] = torch.nn.utils.rnn.pad_sequence(sat_loc, batch_first=False, padding_value=0)
    return positive_sample, negative_sample

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, positive, negative):
        batch_size = positive.size()
        threshold = torch.full(batch_size,0.4).cuda()
        loss = torch.mean(F.relu(threshold - positive + negative))
        return loss

def run_simple(data_loader,clip, model, optimizer):

    total_loss = []
    criterion = MyLoss()
    #batch_acc = {}
    #print("*"*10,"Total Batch :",data_loader.__len__(),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),"*"*10)
    model.train()
    for i,(positive_sample,negative_sample) in enumerate(data_loader):
        #print("*" * 10, "Start Batch :",i , time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        #          "*" * 10)
        #check = data[69]
        optimizer.zero_grad()
        positive_score = model(positive_sample)
        negative_score = model(negative_sample)
        loss = criterion(positive_score,negative_score)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    return model, avg_loss
