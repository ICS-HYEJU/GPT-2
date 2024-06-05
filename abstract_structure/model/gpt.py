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

# Positional Encoding
"""
 sinusoid position encoding
    - PE(pos,2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos,2i+1) = cos(pos / 10000^(2i/d_model)
"""
def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2*(i_hidn // 2)/d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    # Calculate angle for each position, hidden_index(i_hdin)
    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    # For even index
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:, 0::2])
    # For odd index
    sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])

    return sinusoid_table

"""Multi-Head Attention"""
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_hidn = config.d_hidn
        self.n_head = config.n_head
        self.d_head = config.d_head

        self.W_Q = nn.Linear(self.d_hidn, self.n_head*self.d_head)
        self.W_K = nn.Linear(self.d_hidn, self.n_head*self.d_head)
        self.W_V = nn.Linear(self.d_hidn, self.n_head*self.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.linear = nn.Linear(self.n_head*self.d_head, self.d_hidn)

    def forward(self, input, attn_mask):
        # Input shape : (bs, n_seq, d_hidn)
        #                |          |          | ----------> dimension of a sequence vector or dimension of embedded vector
        #                |          | -------------------> sequence length = len(position)
        #                | ----------------------------> batch size
        batch_size = input.size(0)
        #
        Q, K, V = input, input, input
        #
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        "Decoder : making future tokens"
        # Attn_mask shape : (bs, n_seq, n_seq)
        # The score is (query * key)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        # (bs, n_head, n_q_seq, n_k_seq)

        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_haed*self.d_head)
        # (bs, n_q_seq,  n_head, d_head).view(bs, -1, n_head*d_head) = (bs, 256, 256) = (bs, n_seq, n_head*d_head)
        output = self.linear(context)

        # (bs, n_q_seq, d_hidn),  (bs, n_head, n_seq, n_k_seq)
        return output, attn_prob

"""Scaled Dot Product Attention"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head:int):
        super().__init__()
        self.scale =1 / (d_head**0.5)

    def forward(self, Q, K, V, attn_mask):
        # Q, K, V shape is [bs, n_head, n_x_seq, d_head], (x = q, k, v)

        # 1) score = Quries * Keys / Scale
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        #  | ----> (bs, n_head, n_q_seq, n_k_seq)

        # 2) Masked Scores
        scores.masked_fill_(attn_mask, -1e9)
        #  | ----> (bs, n_head, n_q_seq, n_k_seq)

        # 3) Softmax along rows(=n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)  # softmax along row
        #  | ----> (bs, n_head, n_q_seq, n_k_seq)

        # 4) Softmax * Value
        context = torch.matmul(attn_prob, V)
        #  | ----> (bs, n_head, n_q_seq, d_head)
        #
        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_v_seq)
        # context is the score
        # attn_prob = softmax value, before (* Value)
        return context, attn_prob

"""Feed Forward"""
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=config.d_hind,
                               out_channels=config.d_hind*4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=config.d_hind * 4,
                               out_channels=config.d_hind, kernel_size=1)
        self.active = F.gelu

    def forward(self,inputs):
        output = self.active(self.conv1(inputs.transpose(1,2)))
        # (bs, d_ff, n_seq)

        output = self.conv2(output).transpose(1,2)
        # (bs, n_seq, d_hidn)
        return output

"""Decoder Layer"""
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hind, eps=self.config.layer_norm_eps)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hind, eps=self.config.layer_norm_eps)

    def forward(self, dec_inputs, self_attn_mask):
        # dec_inputs shape : (bs,n_seq, d_hidn)
        #
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, self_attn_mask)
        #(bs,n_seq,d_hidn), (bs,n_head,n_seq, n_seq)

        self_att_outputs = self.layer_norm1(self_att_outputs + dec_inputs) #skip connection
        # (bs,n_seq,d_hidn)

        ffn_outputs = self.pos_ffn(self_att_outputs)
        #(bs,n_seq, d_hidn)
        ffn_outputs =self.layer_nomr2(ffn_outputs + self_att_outputs)      #skip connection

        return ffn_outputs, self_attn_prob

""" Decoder """
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ======== Embedding Functions ========
        #   1) vocabulary embeddings
        #      : (bs, n_dec_seq, n_dec_vocab) ---(nn.Embedding)---> (bs, n_dec_seq, d_hidn)
        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hdin)
        #
        #   2) positional encoding
        #       : (bs, n_dec_seq+1, 1) ---(postional_encoding)--> (bs, n_dec_seq+1, d_hidn)
        #       : def get_sinusoid_encoding_table(n_dec_seq, d_hidn) : Generate [n_dec_seq+1, d_hidn] table
        #           - This table has positinal encoding vector at each position
        sinusoid_table = torch.FloatTensor(
            get_sinusoid_encoding_table(self.config.n_dec_seq+1, self.config.d_hidn)
        )
        # The self.pos_emb below simply uses the sinusoid_table and is not trained.
        # When using nn.Embedding, the weights in the table are updated through training.
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        # ====================================

        # ======= Structure of Decoder =======
        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
        # ====================================

    def forward(self, dec_inputs):
        # dec_inputs.shape = [bs, n_dec_seq, n_dec_vocab]
        positions = torch.arange(dec_inputs.size(1),
                                device=dec_inputs.device,
                                dtype=dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        # position.shape = (bs, n_dec_seq)
        # Values in position tensor =  [0 ~ n_dec_seq-1], so do '+1'
        """
        Example) n_dec_seq = 5, bs = 2
                 torch.arange(n_dec_seq) = tensor([0, 1, 2, 3, 4]) --> shape: (5)
                 torch.arange(5).expand(bs, n_dec_seq) = tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]) --> shape: (2, 5)

        => So, by adding '+1' to torch.arange(5).expand(bs, n_deq_seq),
              positions = tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        """

        pos_mask = dec_inputs.eq(self.config.i_pad)
        # is self.config.i_pad = 0,
        # dec_inputs.eq(self.config.i_pad) : If values in dec_inputs are same as i_pad, return True
        """
        Example) IF dec_inputs = tensor( [ [ 13, 41, 3, 0, 0 ] , [11,  24, 0, 0, 0 ] ] ),
                    dec_inputs.eq(0) = tensor( [ [F, F, F, T, T ],  [F, F, T, T, T] ])
        """

        positions.masked_fill_(pos_mask, 0)
        # 0 instead of True of pos_mask

        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)
        # |                            |                         | ---------> (bs, n_dec_seq)
        # |                            | ------> (bs, n_dec_seq, n_dec_vocab)
        # | -----------> (bs, n_dec_seq, n_dec_vocab) = (bs, n_dec_seq, d_hidn)

        # Masking
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_attn_decoder_mask + dec_attn_decoder_mask), 0)
        # torch.gt : if value is bigger than 0, True. else: False
        # (bs, n_dec_seq, n_dec_seq)

        self_attn_probs = []
        for layer in self.layers:
            dec_outputs, self_attn_prob = layer(dec_outputs, dec_self_attn_mask)
            # (bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq)
            self_attn_probs.append(self_attn_prob)
        # (bs, n_dec_seq, d_hidn), ([bs, n_dec_seq, n_dec_seq])
        return dec_outputs, self_attn_probs

""" Mask Functions """
" 1) padding mask"
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    # (bs, len_k)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    # (bs, len_q, len_k)
    return pad_attn_mask

" 2) Attention Decoder Mask"
def get_attn_decoder_mask(seq):
    # the shape of seq : (bs, n_seq)

    # Generate Q-len, K-len table with all values set to 1
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    # Make upper triangular part of a matrix(2D)
    subsequent_mask = subsequent_mask.tril(diagonal=1)
    return subsequent_mask

""" GPT Model """
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = Decoder(self.config)

    def forward(self, dec_inputs):
        # Shape of dec_inputs : (bs, n_dec_seq, n_dec_seq)
        dec_outputs, dec_self_attns_prob = self.decoder(dec_inputs)
        # (bs, n_seq, d_hidn), [(bs, n_head,n_dec_seq, n_dec_seq)]
        #
        return dec_outputs, dec_self_attns_prob

    # def save(self, epoch, loss, path):
    #     torch.save({
    #         'epoch': epoch,
    #         'loss':loss,
    #         'state_dict':self.state_dict()
    #     }, path)
    #
    # def load(self, path):
    #     save = torch.load(path)
    #     self.load_state_dict(save["state_dict"])
    #     return save["epoch"], save["loss"]

""" GPTPretrain"""
class  GPTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gpt = GPT(self.config)
        # The shape of output of GPT : (bs, n_dec_seq, d_hidn)

        # Save Score for each words
        self.projection_lm = nn.Linear(self.config.d_hidn, self.config.n_dec_vocab, bias=False)
        #
        # projection_lm share weight with Embedding of Decoder
        # dec_emb(dec_inputs) = [vocab -> vector]
        #                        : (bs, n_dec_seq, n_dec_vocab) -> (bs, n_dec_sq, d_hidn)
        # projection_lm(dec_outputs) = [vector -> vocab]
        #                                : (bs, n_dec_sq, d_hidn) -> (bs, n_dec_seq, n_dec_vocab)
        # Therefore, it's fine to use the same weights (assuming they are well-trained)
        # Using the same weights ensures consistent weights, which can improve training performance
        # If different weights are used for vocab -> vector and vector -> vocab,
        # the network may struggle to learn the weights effectively to enhance performance.
        self.projection_lm.weight = self.gpt.decoder.dec_emb.weight

    def forward(self, dec_inputs):
        dec_outputs, dec_self_attn_probs = self.gpt(dec_inputs)
        logits_lm = self.projection_lm(dec_outputs)

        return logits_lm[:, :-1, :].contiguous(), dec_self_attn_probs


"config = config.dataset_info"