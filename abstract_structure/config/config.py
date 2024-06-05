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

def get_config_dict():
    dataset_info = dict(
        name = 'corpus.txt',
        path = '/home/hjchoi/PycharmProjects/GPT-2/ko_wiki/all',
        #
        n_dec_vocab = 8007,
        n_dec_seq = 256,
        n_layer = 6,
        d_hidn = 256,
        i_pad = 0,
        d_ff = 1024,
        n_head = 4,
        d_head = 64,
        dropout = 0.1,
        layer_norm_eps = 1e-12
    )

    path = dict(
        save_base_path = 'runs'
    )

    #subtokenizer = dict(
        # load model

    #)

    model = dict(
        name = 'gpt'
    )

    solver = dict(
        name = 'Adam',
        gpu_id = 0,
        lr0 = 1e-4,
        weight_decay = 5e-4,
        max_epoch = 10
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
        weight_info = weight_info
    )

    return config




