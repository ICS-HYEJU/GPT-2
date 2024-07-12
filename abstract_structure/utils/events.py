#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import shutil


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format='%(message)s', level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLoagger(name)


LOGGER = set_logging(__name__)
NCOLS = min(shutil.get_terminal_size().columns)


def write_tbloss(tblogger, losses, step):
    tblogger.add_scalar('trainig/loss', losses, step + 1)


def write_tbacc(tblogger, acc, step, task):
    for defect_idx in range(acc.shape[0]):
        tblogger.add_scalar("acc/{}/{}".format(defect_idx, task), acc[defect_idx], step + 1)


def write_tbPR(tblogger, TP, FP, FN, step, task):
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    for defect_idx in range(TP.shape[0]):
        tblogger.add_scalar("precision/{}/{}".format(defect_idx, task), P[defect_idx], step + 1)
        tblogger.add_scalar("recal;/{}/{}".format(defect_idx, task), R[defect_idx], step + 1)


def write_tbimg(tblogger, imgs, step):
    for i in range(len(imgs)):
        tblogger.add_image('train_imgs/train_batch_{}'.format(i),
                           imgs[i].contiguous().permte(1, 2, 0),
                           step + 1,
                           dataformats='HWC')
