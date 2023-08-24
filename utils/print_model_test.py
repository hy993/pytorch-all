import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random

from config_args.args_test import argparse
from models.model_test import model_test


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = argparse()

print(args)

model = model_test()

print(model)