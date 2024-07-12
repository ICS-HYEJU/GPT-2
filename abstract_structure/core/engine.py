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
import time

import math

from typing import Dict, Tuple

import torch.optim as optim
from torch.optim import AdamW as Adam
from torch.nn import LayerNorm

from torch.utils.data import Dataset, DataLoader

from abstract_structure.config.config import Config

class Trainer():
    def __init__(self, cfg, device):
        self.cfg = Config(cfg)
        self.device = device

        #===== Save Path =====
        self.save_path = self.make_save_path()

        #===== Tensorboard =====
        # self.tblogger = SummaryWirtier(self.save_path)

        #===== DataLoader =====
        self.train_loader = self.get_dataloader()

        #===== Model =====
        self.model = self.build_model()

        #===== Optimizer =====
        self.optimizer = self.build_optimizer()

        #===== Scheduler =====
        self.scheduler = self.build_scheduler()

        #===== Loss =====
        self.criterion = self.set_criterion()

        #===== Parameters =====
        self.max_epoch = self.cfg.solver['max_epoch']
        self.max_stepnum = len(self.train_loader)

    def cal_loss(self, logits, labels):
        logits = logits.view(-1, logits.size(2))
        labels = labels.view(-1)
        return self.criterion(logits, labels)

    def set_criterion(self):
        return nn.CrossEntropyLoss().to(self.device)

    def build_scheduler(self):
        from abstract_structure.solver.fn_scheduler import build_scheduler
        return build_scheduler(self.cfg, self.optimizer)

    def build_optimizer(self):
        from abstract_structure.solver.fn_optimizer import build_optimizer
        return build_optimizer(self.cfg, self.model)

    def build_model(self):
        name = self.cfg.model['name']
        if name == "GPT":
            from abstract_structure.model.gpt import GPTPretrain
            model = GPTPretrain(self.cfg)
        else:
            raise NotImplementedError(f'The required model is not implemented yet...')
        return model.to(self.device)

    def get_dataloader(self):
        from abstract_structure.dataset import create_dataloader
        train_loader = create_dataloader(self.cfg)
        return train_loader

    def make_save_path(self):
        save_pretrain = os.path.join(self.cfg.path['save_pretrain'],
                                     self.cfg.model['name'] + '_pretrain' )
        os.makedirs(save_pretrain, exist_ok=True)
        return save_pretrain

    def start_train(self):
        best_epoch = self.cfg.dataset_info['best_epoch']
        n_epoch = self.cfg.dataset_info['n_epoch']
        self.save_pretrain = "save_gpt_pretrain.json"

        try:
            print(f'Training Start..')
            start_time = time.time()
            #
            losses = []
            offset = best_epoch
            #
            for step in tqdm(range(n_epoch), desc="Epoch"):
                epoch = step + offset
                #
                loss = self.train_one_epoch(epoch=epoch)
                #
                losses.append(loss)
                #
                self.model.gpt.save(epoch, loss, self.save_path)
                #
                self.scheduler.step()
                # #
                # best_epoch, best_loss = 0, 0
                # if os.path.isfile(self.save_pretrain):
                #     best_epoch, best_loss = self.model.gpt.load(self.save_pretrain)
                #     print(f'load pretrain from: {self.save_pretrain}, epoch: {best_epoch}, loss={best_loss}')
                #     best_epoch += 1
                #
            print(f'\nTraining completed in {(time.time() - start_time) / 3600:.3f} hours.')
        except Exception as _:
            print('ERROR in training loop or eval/save model.')
            raise

    def train_one_epoch(self,epoch):
        losses = []
        self.model.train()

        with tqdm(total=len(self.train_loader), desc=f'Train({epoch}', leave=True) as pbar:
            for i, value in enumerate(self.train_loader):
                dec_inputs, _ = map(lambda v: v.to(self.device), value)
                labels = dec_inputs[:, 1:].contiguous()
                # shape of labels_lm :

                self.optimizer.zero_grad()
                outputs = self.model(dec_inputs)
                logits = outputs[0]
                # shape of logits_lm :

                # shape of logits_lm.view()
                # shape of labels_lm.view(-1)
                loss = self.cal_loss(logits, labels)

                loss_val = loss.item()
                losses.append(loss_val)

                loss.backward()
                self.optimizer.step()

                pbar.update(1)
        pbar.set_description(f'Loss: {loss_val:.3f} ({np.mean(losses):.3f})')
        return np.mean(losses)

if __name__ == '__main__':
    from abstract_structure.config.config import get_config_dict

    # Get configuration
    cfg = get_config_dict()
    # Set Device
    if cfg['device'] is not None:
        device = torch.device('cuda:{}'.format(cfg['device']))
        torch.cuda.set_device(cfg['device'])
    else:
        device = torch.device('cpu')
    # Get Trainer
    trainer = Trainer(cfg, device=device)
    # Start train
    trainer.start_train()
