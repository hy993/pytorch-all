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

model = model_test_resnet_5()

print(model)