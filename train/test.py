# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random

from config_args.args_test import argparse
from models.model_test import *
from dataloader.dataloader import data_test

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = argparse()

print(args)

#model = model_test_resnet_7()
model = VGG10()
model_name = 'vgg10'

save_base_dir = '../save/' + model_name

if not os.path.exists(save_base_dir):
    os.mkdir(save_base_dir)
    print('Creat direction:', save_base_dir)
else:
    print('Direction exist:', save_base_dir)
    
if not os.path.exists(save_base_dir+'/models_saved'):
    os.mkdir(save_base_dir+'/models_saved')
    print('Creat direction:', save_base_dir+'/models_saved')
else:
    print('Direction exist:', save_base_dir+'/models_saved')
    
if not os.path.exists(save_base_dir+'/figs_saved'):
    os.mkdir(save_base_dir+'/figs_saved')
    print('Creat direction:', save_base_dir+'/figs_saved')
else:
    print('Direction exist:', save_base_dir+'/models_saved')
    
epoch_loss_fig_dir = save_base_dir + '/figs_saved/train_loss_test_resnet.png'
epoch_acc_fig_dir = save_base_dir + '/figs_saved/train_acc_test_resnet.png'
model_save_dir = save_base_dir + '/models_saved/test_resnet.pth'

print(model)

train_dataloader = torch.utils.data.DataLoader(                 # vision.utils : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网。
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),       # 图像转化为Tensor
                           transforms.Normalize((0.1307,), (0.3081,))       # 标准化
                       ])),
        batch_size=128, shuffle=True)            # shuffle() 方法将序列的所有元素随机排序

# 测试集
test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)            # shuffle() 方法将序列的所有元素随机排序
valid_dataloader = test_dataloader
#train_dataset = data_test(flag='train')
#train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
#valid_dataset = data_test(flag='valid')
#valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)

#print(train_dataset)

model = model.to(args.device)
#criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
train_acces = []
valid_acces = []


# 训练
for epoch in range(args.epochs):
    model.train()
    train_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(train_dataloader,0):
        print(data_x.numpy().shape)
        data_x = data_x.to(torch.float32).to(args.device)
        #print(data_x.shape)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = model(data_x)
        #print(data_x.shape)
        #print(outputs.shape)
        #print(data_y.shape)
        optimizer.zero_grad()
        #loss = criterion(data_y,outputs)
        loss = F.nll_loss(outputs, data_y.long())
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        
        _, pred = torch.max(outputs.data,1) # Calculation accuracy
        num_correct = (pred == data_y.long()).sum().item()
        train_acc = num_correct / data_x.shape[0]
        train_acces.append(train_acc*100)
        
        if idx%(len(train_dataloader)//2)==0:
            print("epoch={}/{},{}/{}of train, loss={}, acc={}".format(
                epoch, args.epochs, idx, len(train_dataloader),loss.item(), np.mean(train_acces)))
    train_epochs_loss.append(np.average(train_epoch_loss))
    
    #=====================valid============================
    model.eval()
    valid_epoch_loss = []
    
    valid_acc = []
    
    for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = model(data_x)
        #loss = criterion(outputs,data_y)
        loss = F.nll_loss(outputs, data_y.long())
        valid_epoch_loss.append(loss.item())
        valid_loss.append(loss.item())
        
        _,valid_pred=torch.max(outputs.data,1) # test_pred = output.data.max(1)[1]
        num_correct = (valid_pred == data_y.long()).sum().item()
        valid_acc.append(num_correct / data_x.shape[0])
        
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    
    valid_acces.append(np.mean(valid_acc))
    
    #==================early stopping======================
    #early_stopping(valid_epochs_loss[-1],model=model,path=r'c:\\your_model_to_save')
    #if early_stopping.early_stop:
    #    print("Early stopping")
    #    break
    #====================adjust lr========================
    # lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    # lr_adjust = {
    #         2: args.learning_rate*(1e-1), 4: args.learning_rate*(1e-1)*0.5, 6: args.learning_rate*(1e-2), 
    #         8: args.learning_rate*(1e-3), 10: args.learning_rate*(1e-4), 15: args.learning_rate*(1e-5), 
    #         20: args.learning_rate*(1e-6)
    #     }
    # if epoch in lr_adjust.keys():
    #     lr = lr_adjust[epoch]
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     print('Updating learning rate to {}'.format(lr))
        
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(train_loss[:])
plt.title("train_loss")
plt.subplot(122)
plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
plt.plot(valid_epochs_loss[1:],'-o',label="valid_loss")
plt.title("epochs_loss")
plt.legend()
#plt.show()
plt.savefig(epoch_loss_fig_dir)

plt.figure(figsize=(12,4))
plt.plot(train_acces[1:],'-o',label="train_acc")
plt.plot(valid_acces[1:],'-o',label="valid_acc")
plt.title("epochs_acc")
plt.legend()
#plt.show()
plt.savefig(epoch_acc_fig_dir)


torch.save(model, model_save_dir)

# 预测
#model.eval()
#predict = model(data)
