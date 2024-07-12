import torch
import torch.nn as nn

import torch.nn.functional as F
from copy import deepcopy

import numpy as np
import json

import os
import sentencepiece as spm

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange

import math

from typing import Dict, Tuple

import torch.optim as optim
from torch.optim import AdamW as Adam
from torch.nn import LayerNorm

from ast import literal_eval

from torch.utils.data import Dataset, DataLoader
from abstract_structure.config.config import Config

def divide_file():
    path = '/storage/hjchoi/kowiki/'
    line_cnt = 0
    with open('/home/hjchoi/PycharmProjects/GPT-2/abstract_structure/dataset/kowiki_preprocessed.json', 'r') as f:
        with tqdm(total=11390914, desc=f"Devide file") as pbar:
            for line in f:
                line = literal_eval(line)
                line_cnt += 1
                with open(path + str(line_cnt)+'.json', 'w') as f:
                    json.dump(line, f)
                pbar.update(1)

if __name__ == '__main__':
    divide_file()