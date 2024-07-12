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

# This function that allows access to dictionary keys using '.' instead of '[]'
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.load(f)
            return Config(config)

def get_config_dict():
    dataset_info = dict(
        name = 'corpus.txt',
        path = '/home/hjchoi/PycharmProjects/GPT-2/ko_wiki/all',
        #
        n_dec_vocab = 8007,
        n_dec_seq = 768,
        n_layer = 12,
        d_hidn = 256,
        i_pad = 0,
        d_ff = 1024,
        n_head = 12,
        d_head = 64,
        dropout = 0.1,
        layer_norm_eps = 1e-12,
        #
        batch_size = 64,
        n_epoch = 20,
        best_epoch = 0
    )

    path = dict(
        save_base_path = 'runs',
        save_pretrain = '/home/hjchoi/PycharmProjects/GPT-2/abstract_structure/save_gpt_pretrain.json'
    )


    model = dict(
        name = 'GPT'
    )

    solver = dict(
        name = 'Adam',
        gpu_id = 0,
        lr0 = 5e-5,
        weight_decay = 5e-4,
        max_epoch = 20
    )

    scheduler = dict(
        name = 'LambdaLR',
        lr_lambda = lambda epoch:0.95 ** epoch
    )

    weight_info = dict(
        name = 'last_weight.pth'
    )

    # Merge all info into a dictionary variable
    config = dict(
        dataset_info = dataset_info,
        path = path,
        #subtokenizer = subtokenizer,
        model = model,
        solver = solver,
        scheduler = scheduler,
        weight_info = weight_info,
        device = 1
    )

    return config


