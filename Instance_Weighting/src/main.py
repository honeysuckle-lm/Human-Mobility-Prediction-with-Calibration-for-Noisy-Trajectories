# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import argparse
import numpy as np
from json import encoder
import gc

from train import run_simple, RnnParameterData, generate_input_trajectory
from model import TrajRNN

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size,
                                  tim_emb_size=args.tim_emb_size, hidden_size=args.hidden_size,
                                  dropout_p=args.dropout_p,data_name=args.data_name, lr=args.learning_rate,
                                  batch_size=args.batch_size,lr_step=args.lr_step, lr_decay=args.lr_decay,
                                  L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, clip=args.clip, epoch_max=args.epoch_max, data_path=args.data_path,
                                  save_path=args.save_path)

    print('*' * 15 + 'start training' + '*' * 15)

    model = TrajRNN(parameters=parameters).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %d' % (total))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr)
                           #weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-4)

    lr = parameters.lr
    metrics = {'train_loss': []}

    data_weighting, data_train = generate_input_trajectory(parameters.data_neural, parameters.batch_size)
    print("Total Batch :", data_train.__len__())
    del parameters.data_neural
    gc.collect()

    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    try:
        os.mkdir(SAVE_PATH + tmp_path)
    except:
        for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(SAVE_PATH + tmp_path)
        os.mkdir(SAVE_PATH + tmp_path)

    for epoch in range(parameters.epoch):
        st = time.time()
        model, avg_loss = run_simple(data_train, parameters.clip, model, optimizer)
        print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
        metrics['train_loss'].append(avg_loss)
        print('*' * 20)

        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        scheduler.step(avg_loss)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmin(metrics['train_loss'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            print('load epoch={} model state'.format(load_epoch))
            print('*' * 20)
        if epoch == 0:
            print('single epoch time cost:{}'.format(time.time() - st))
        if lr < 1.0 * 1e-7:
            break
        if avg_loss < 0.02:
            break
    
    mid = np.argmin(metrics['train_loss'])
    #final_loss = metrics['train_loss'][int(mid)]
    load_name_tmp = 'ep_' + str(mid) + '.m'
    model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
    save_name = 'result'
    json.dump({'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.txt', 'w'), indent=4)
    torch.save(model.state_dict(), SAVE_PATH + save_name + '.m')

    #model.load_state_dict(torch.load(SAVE_PATH + 'ep_497.m'))
    score_list = {}
    for i, (positive_sample, negative_sample) in enumerate(data_train):
        positive_score = model(positive_sample).cpu().detach().numpy().tolist()
        user_id = positive_sample['user_id']
        for id in range(len(user_id)):
            score_list[str(user_id[id])] = positive_score[id]

    for key in data_weighting.keys():
        data_weighting[key]['score'] = score_list[key]
    json.dump(data_weighting, fp=open(SAVE_PATH + 'data_weighting.json', 'w'))

if __name__ == '__main__':
    #np.random.seed(1)
    #torch.manual_seed(1)  # 随机初始化种子
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=300, help="location embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=50, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='Oidd')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_step', type=int, default=20)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=500)
    parser.add_argument('--rnn_type', type=str, default='GRU', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--save_path', type=str, default='../results/')
    args = parser.parse_args()
    ours_acc = run(args)
