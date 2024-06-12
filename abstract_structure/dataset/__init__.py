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

from torch.utils.data import Dataset, DataLoader

def create_dataloader(config):
    if config.dataset_info.name == 'corpus.txt'
        from abstract_structure.dataset.dataset import PreTrainDataset as dataset_class
    else:
        raise ValueError('Invalid dataset name, currently supported [ corpus.txt ]')
    #
    data_path = config.dataset_info.data_path + 'corpus.txt'
    #
    train_object = dataset_class(
        corpus_file=data_path, dataset_name="kowiki_small",
        vocab_size=8000,
        out_preprocessed_file='kowiki_preprocessed.json'
    )
    #
    train_loader = DataLoader(
        dataset_object,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset_class.collate_fn
    )
    #
    #val_object
    #
    #val_loader

    return train_loader