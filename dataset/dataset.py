import sys
sys.path.append('..')

from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import torch

class CatDog(Dataset):
    # 新建猫狗分类数据集
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data_files = []
        for ii in (os.listdir(self.data_path + 'cat')):
            self.data_files.append([ii, 'cat'])
        for jj in (os.listdir(self.data_path + 'dog')):
            self.data_files.append([jj, 'dog'])
        #print(data_path)
        #for iii in range(len(self.data_files)):
        #    print(self.data_files[iii])
    def __getitem__(self, idx):
        img = cv2.imread(self.data_path + self.data_files[idx][1] + '//' + self.data_files[idx][0])
        img = cv2.resize(img, (28, 28))
        label = self.data_files[idx][1]
        if label == 'cat':
            label_ = torch.Tensor([0])
        if label == 'dog':
            label_ = torch.Tensor([1])
        #return torch.unsqueeze(torch.Tensor(img), 0), label_ # 没有用
        return torch.Tensor(img).view(3, 28, 28), label_
    def __len__(self):
        cat_num = len(os.listdir(self.data_path + 'cat'))
        dog_num = len(os.listdir(self.data_path + 'dog'))
        #print(cat_num + dog_num)
        return cat_num + dog_num
    