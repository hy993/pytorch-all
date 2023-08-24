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
import cv2

import random

from config_args.args_test import argparse
from models.model_test import *
from dataloader.dataloader import data_test
from loss.loss_test import *
from dataset.dataset import *

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from torchviz import make_dot

seed = 2242
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#print(CatDog('../data/test_cat_dog/'))
# 定义配置文件
args = argparse()

train_data = CatDog('..//data//test_cat_dog//')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

test_data = CatDog('..//data//test_cat_dog//')
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

#print(enumerate(test_dataloader, 0))
# 定义模型
model = model_test_resnet_7_cat_dog_cls()
model = model.to(args.device)

# 定义保存地址
model_name = 'model_test_resnet_7_cat_dog_cls'
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

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
# 定义损失变量
train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
train_acces = []
valid_acces = []
# 定义损失函数
#criterion = nn.CrossEntropyLoss()
criterion = TestLoss()
#model.eval()
# 可视化网络
#input = torch.rand(1, 1, 28, 28)
#output = model(input)
#print(output)
#g = make_dot(model(input.requires_grad_(True)),
#        params=dict(list(model.named_parameters()) + [('x', input)]))
#g.render('vgg10', view=False, format='pdf')
#print(g)
#print(model)
#model.train()
for epoch in range(args.epochs):
    model.train()
    #train_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(train_dataloader, 0):
        #print(data_x.numpy().shape)
        # 发送数据到指定设备
#        print(data_y.t())
        data_y = data_y.t()[0]
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)

        # 清除优化器的梯度
        optimizer.zero_grad()

        # 前向推导 计算损失 损失反向传播 权重优化
        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()
        
        # 保存损失
        train_loss.append(loss.item())
        
        # 计算准确率
        _, pred = torch.max(outputs.data, 1)
        num_correct = (pred == data_y.long()).sum().item()
        train_acc = num_correct / data_x.shape[0]
        train_acces.append(train_acc * 100)
        
    print('epoch:', epoch, 'accuracy:', train_acc, 'loss:', loss.item())
        #print(loss.item())
#    print('idx:', idx, 'img:', data_x[0], 'label:', data_y)
#    outputs = model(data_x)
#    print(outputs)
    
#    if idx == 11:
#        print(data_x[0].numpy().shape)
#        print(data_x[0])
#        #print(data_y[0])
#        data_x = data_x.cpu()
#        img2show = data_x[0].numpy()
#        font = cv2.FONT_HERSHEY_SIMPLEX
#        cv2.putText(img2show, data_y[0], (10, 45), font, 2, (0, 0, 255), 3)
#        cv2.imshow('image', img2show)
#        cv2.waitKey(0)

plt.figure(figsize=(12,4))
plt.subplot(111)
plt.plot(train_loss[:], label='train loss')
plt.title("train_loss")
plt.legend()
#plt.show()
plt.savefig(epoch_loss_fig_dir)

# 保存模型
torch.save(model, model_save_dir)