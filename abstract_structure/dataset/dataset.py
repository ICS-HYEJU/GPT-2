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


def load_subtokenzier(corpus, vocab_size):
    prefix = "kowiki_small"
    spm.SentencePieceTrainer.Train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
        " --model_type=bpe" +
        " --max_sentence_length=999999" +
        " --pad_id=0 --pad_piece=[PAD]" +
        " --unk_id=1 --unk_piece=[UNK]" +
        " --bos_id=2 --bos_piece=[BOS]" +
        " --eos_id=3 --eos_piece=[EOS]" +
        " --user_defined_symbols=[SEP],[CLS],[MASK]")


class PreTrainDataset(Dataset):
    def __init__(self, corpus_file, dataset_name, vocab_size,
                 out_preprocessed_file, n_seq=256, saved_vocab_file=None,
                saved_preprocessed_file=None):

        # Make subtokenizer using sentencepiece
        # if os.path.exists('/home/hjchoi/PycharmProjects/GPT-2/abstract_structure/dataset/kowiki_small.model'):
        #     self.saved_vocab_file = '/home/hjchoi/PycharmProjects/GPT-2/abstract_structure/dataset/kowiki_small.model'
        # else:
        self.saved_vocab_file = saved_vocab_file
        self.vocab = self.make_subtokenizer(
            corpus_file, dataset_name, vocab_size, self.saved_vocab_file)

        # Make PreTrained Dataset
        # if os.path.exists('/storage/hjchoi/debug_json/preprocessed_debug.json'):
        #     self.saved_preprocessed_file = '/storage/hjchoi/debug_json/preprocessed_debug.json'
        # else:
        self.saved_preprocessed_file = saved_preprocessed_file

        self.sentences = self.make_pretrain_data(
            corpus_file, out_preprocessed_file, n_seq=n_seq,
            saved_file=self.saved_preprocessed_file
        )

    def __len__(self):
        assert len(self.sentences) != 0, "No sentences"
        return len(self.sentences)

    def __getitem__(self, item):
        # return (torch.tensor(self.sentences), torch.tensor(item))
        return (torch.tensor(self.sentences[item]), torch.tensor(item))
    def make_subtokenizer(self, corpus_file, dataset_name, vocab_size, saved_vocab_file=None):
        # Load saved vocab file when it exists.
        if saved_vocab_file is not None:
            assert os.path.exists(saved_vocab_file), "There is no file...{}".format(saved_vocab_file)

            vocab = spm.SentencePieceProcessor()
            vocab.Load(saved_vocab_file)
            #
            assert vocab.vocab_size()-7 == vocab_size, "The size of vocabulary is not the same..."
            #
            print("[Sub Tokenizer] Loading completed...")
        #
        # Generate a new vocabulary subtokenizer
        else:
            assert os.path.exists(corpus_file), "There is no file...{}".format(corpus_file)
            spm.SentencePieceTrainer.Train(
                f"--input={corpus_file} --model_prefix={dataset_name} --vocab_size={vocab_size + 7}" +
                " --model_type=bpe" +
                " --max_sentence_length=999999" +
                " --pad_id=0 --pad_piece=[PAD]" +
                " --unk_id=1 --unk_piece=[UNK]" +
                " --bos_id=2 --bos_piece=[BOS]" +
                " --eos_id=3 --eos_piece=[EOS]" +
                " --user_defined_symbols=[SEP],[CLS],[MASK]")
            #
            vocab = spm.SentencePieceProcessor()
            vocab.Load(dataset_name + '.model')

        return vocab

    def create_pretrain_instances(self, doc, n_seq):
        # for [BOS], [EOS]
        max_seq = n_seq - 2
        tgt_seq = max_seq
        #
        instances = []
        current_chunk = []
        current_length = 0

        for i in range(len(doc)):
            current_chunk.append(doc[i])
            current_length += len(doc[i])
            if i == len(doc) - 1 or current_length >= tgt_seq:
                if 0 < len(current_chunk):
                    tokens = []
                    for chunk in current_chunk: tokens.extend(chunk)
                    tokens = tokens[:tgt_seq]
                    if 1 < len(tokens):
                        instance = {
                            "tokens": self.vocab.PieceToId(["[BOS]"] + tokens + ["[EOS]"])
                        }
                        instances.append(instance)
                current_chunk = []
                current_length = 0

        return instances

    def make_pretrain_data(self, in_file, out_file, n_seq, saved_file=None):
        sentences = []

        if saved_file is not None:
            assert os.path.exists(saved_file), 'There is no file...{}'.format(saved_file)
            #
            print("[PreTrain Data] Counting the number of` lines...")
            line_cnt = 0
            with open(saved_file, 'r') as f:
                for line in f:
                    line_cnt += 1
            print("[PreTrain Data] Counting Completed...")
            #
            with open(saved_file, 'r') as f:
                for i, line in enumerate(tqdm(f, total=line_cnt, desc="Loading and Making Pretrain Dataset", unit="lines")):
                    instance = json.loads(line)
                    sentences.append(instance['tokens'])
                    if i > 128:
                        break

            print("[PreTrain Data] Loading Completed...")

        else:
            assert os.path.exists(in_file), "There is no file..{}".format(in_file)

            line_cnt = 0
            with open(in_file, "r") as in_f:
                for line in in_f:
                    line_cnt += 1

            docs = []
            with open(in_file, "r") as f:
                doc = []
                with tqdm(total=line_cnt, desc=f"Loading") as pbar:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line == "":
                            if 0 < len(doc):
                                docs.append(doc)
                                doc = []
                        else:
                            pieces = self.vocab.EncodeAsPieces(line)
                            if 0 < len(pieces):
                                doc.append(pieces)
                            pbar.update(1)
                        if doc:
                            docs.append(doc)

            # with open(out_file, "w") as out_f:
            with open('/home/hjchoi/PycharmProjects/GPT-2/abstract_structure/dataset/kowiki_preprocessed.json', "w") as out_f:
                with tqdm(total=len(docs), desc=f"Making") as pbar:
                    for i, doc in enumerate(docs):
                        instances = self.create_pretrain_instances(doc, n_seq)
                        for instance in instances:
                            out_f.write(json.dumps(instance))
                            out_f.write("\n")
                            #
                            sentences.append(instance['tokens'])
                        pbar.update(1)
        print("[PreTrain Data] Making Completed...")

        return sentences

    @staticmethod
    def collate_fn(inputs):
        # for batch data processing
        dec_inputs, item = list(zip(*inputs)) # Values in the same column are zipped together by using '*'

        # This function returns a Tensor of size T x B x * or B x T x * where T is the length of the longest sequence.
        # This function assumes trailing dimensions and type of all the Tensors in sequences are same.
        # if batch_first = True: return B x T x *
        # else : return Tx B x *
        dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

        # Generate batch data
        batch = [
            dec_inputs,
            torch.stack(item, dim=0)
        ]
        return batch






if __name__ == "__main__":
    from abstract_structure.config.config import get_config_dict

    config = Config(get_config_dict())
    config = Config(config.dataset_info)

    path = config.path

    obj = PreTrainDataset(corpus_file=path+'/corpus.txt', dataset_name="kowiki_small",
                          vocab_size=8000,
                          out_preprocessed_file='kowiki_preprocessed.json',
                          )
    #
    obj.__getitem__(10)
    obj_dataloader = DataLoader(
        obj,
        batch_size=3,
        shuffle=True,
        collate_fn=PreTrainDataset.collate_fn
    )
    for i, data in enumerate(obj_dataloader):
        print(data)

