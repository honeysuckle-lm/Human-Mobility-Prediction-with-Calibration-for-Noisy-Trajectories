# coding: utf-8

import torch
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.data as data


class RnnParameterData(object):
    def __init__(self, loc_emb_size,tim_emb_size, hidden_size,
                 lr, batch_size, lr_step, lr_decay, dropout_p, L2, clip, optim, epoch_max, rnn_type,
                 data_path, save_path, data_name='Oidd'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        with open(self.data_path + self.data_name + '.json', 'r') as f:
            data = json.load(f)

        self.data_neural = data

        self.tim_size = 96
        self.loc_size = 779
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.batch_size = batch_size
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip
        self.rnn_type = rnn_type


def generate_input_trajectory(data_neural, mode,batch_size, candidate):
    data_train = []
    count = 0
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        count += 1
        if mode == 'test':
            for i in train_id:
                session = sessions[i]
                tim_np = np.array([s[1] + 1 for s in session[:-1]])
                loc_np = np.array([s[0] + 1 for s in session[:-1]])
                target = np.array([s[0] + 1 for s in session[1:]])  # 留出开始一条记录
                trace = {}
                trace['loc'] = torch.LongTensor(loc_np)
                trace['tim'] = torch.LongTensor(tim_np)
                trace['target'] = torch.LongTensor(target)
                trace['score'] = data_neural[u]['score']
                data_train.append(trace)
        else:
            tim_np = []
            loc_np = []
            target = []
            for i in train_id:
                session = sessions[i]
                tim_np.extend([s[1] + 1 for s in session[:-1]])
                loc_np.extend([s[0] + 1 for s in session[:-1]])
                target.extend([s[0] + 1 for s in session[1:]])
            tim_np = np.array(tim_np)
            loc_np = np.array(loc_np)
            target = np.array(target)
            trace = {}
            trace['loc'] = torch.LongTensor(loc_np)
            trace['tim'] = torch.LongTensor(tim_np)
            trace['target'] = torch.LongTensor(target)
            trace['score'] = data_neural[u]['score']
            data_train.append(trace)

    print('*' * 15, 'End Trajectory Transformation', mode, '*' * 15)
    print('*' * 15, 'Total user:', count, '*' * 15)
    data = MyData(data_train)
    return DataLoader(data, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_fn, num_workers=16,pin_memory=True)


class MyData(data.Dataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        return {
                'tim': self.data_source[idx]['tim'],
                'loc': self.data_source[idx]['loc'],
                'target': self.data_source[idx]['target'],
                'score': self.data_source[idx]['score']
                }

def collate_fn(data):
    #data.sort(key=lambda x: len(x['target']), reverse=True)
    #data_length = [len(sample['target']) for sample in data]
    tim = [s['tim'] for s in data]
    loc = [s['loc'] for s in data]
    target = [s['target'] for s in data]
    trandata = {}
    trandata['tim'] = torch.nn.utils.rnn.pad_sequence(tim, batch_first=False, padding_value=0)
    trandata['loc'] = torch.nn.utils.rnn.pad_sequence(loc, batch_first=False, padding_value=0)
    trandata['target'] = torch.nn.utils.rnn.pad_sequence(target, batch_first=False, padding_value=0)
    trandata['score'] = torch.FloatTensor([s['score'] for s in data])
    return trandata


def get_acc(target, scores):
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = [0,0,0]
    count = 0
    for i, p in enumerate(predx):
        t = target[i]
        if t > 0:
            count += 1
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return count,acc

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, scores, target, weighting):
        batch_size = scores.size()[1]
        loss = torch.zeros(batch_size, requires_grad= True).cuda()
        for i in range(batch_size):
            loss[i] = F.nll_loss(scores[:,i],target[:,i])
        loss = torch.mul(loss,weighting)
        return torch.mean(loss)

def run_simple(data_loader, mode, clip, model, optimizer):
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.eval()
    total_loss = []
    batch_acc = {}
    #print("Total Batch :",data_loader.__len__())
    if mode == 'train':
        criterion = MyLoss().cuda()
    else:
        criterion = torch.nn.NLLLoss().cuda()

    for i,batch_list in enumerate(data_loader):
        #print("*" * 10, "Start Batch :",i , time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        #     "*" * 10)
        optimizer.zero_grad()
        tim = batch_list['tim'].cuda()
        loc = batch_list['loc'].cuda()
        weighting = batch_list['score'].cuda()
        weighting = torch.div(weighting, torch.mean(weighting))
        target = batch_list['target'].cuda()
        scores = model(loc, tim) #loc和tim都是[sequence length, batch size]维度的tensor
        if mode == 'train':
            loss = criterion(scores, target, weighting)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        elif mode == 'test':
            scores = scores.reshape(tim.size()[0]*tim.size()[1],-1)
            target = target.reshape(tim.size()[0]*tim.size()[1])
            loss = criterion(scores, target)
            batch_acc[i] = [0,0]
            batch_acc[i][0],acc = get_acc(target, scores)
            batch_acc[i][1] = acc[2]

        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        tmp_0 = sum([batch_acc[s][0] for s in batch_acc])
        tmp_1 = sum([batch_acc[s][1] for s in batch_acc])
        avg_acc = tmp_1 / tmp_0
        return avg_loss, avg_acc
