# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

from os import listdir
from os.path import isfile, join

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn.functional as F

from torch import nn

import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    # new imports
    RobertaModel,
    BertPreTrainedModel,
)

from transformers import glue_compute_metrics as compute_metrics
#from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
import copy
import csv
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
class ClickEmbeddingHead(nn.Module):
    """This is same for now as Head for sentence-level classification tasks."""

    def __init__(self, config, embeddingDim):
        super().__init__()
        # embedding dimension 100
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, embeddingDim)
        self.out_proj = nn.Linear(embeddingDim, embeddingDim)

    def forward(self, features, **kwargs):
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

def compute_attention_mask(attention_mask, model):
    if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(model.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    return extended_attention_mask


class ScoringHead(nn.Module):
    """This is same for now as Head for sentence-level classification tasks."""

    def __init__(self, config, embeddingDim, outclass = 2):
        super().__init__()
        # embedding dimension 100
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(embeddingDim, embeddingDim)
        self.out_proj = nn.Linear(embeddingDim, outclass)

    def forward(self, features, **kwargs):
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class LinearScoringHead(nn.Module):
    """This is same for now as Head for sentence-level classification tasks."""

    def __init__(self, config, embeddingDim, outclass = 2):
        super().__init__()
        # embedding dimension 100
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.dense = nn.Linear(embeddingDim, embeddingDim)
        self.out_proj = nn.Linear(embeddingDim, outclass)

    def forward(self, features, **kwargs):
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class RobertaEmbSimLSTM(RobertaForSequenceClassification):

    def __init__(self, config):
        super(RobertaEmbSimLSTM,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        self.hidden_dim = config.hidden_size

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        

        self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 512)

    # def init_hidden(self):
    #     return (torch.randn(2, 1, self.hidden_dim // 2),
    #             torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, seq_out):
        seq_out = torch.transpose(seq_out, 0, 1)
        self.lstm.flatten_parameters()
        output, (h_n, c_n)= self.lstm(seq_out)

        hiddenout = torch.transpose(h_n, 0, 1) #batch, 2, hidden/2
        hiddenout = torch.reshape(hiddenout, (-1, self.hidden_dim)) #batch, hidden

        return hiddenout

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

        # try separate embeddings
        emb_1 = self.embeddingHead(self._get_lstm_features(outputs1[0]))

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)

        # try separate embeddings
        emb_2 = self.embeddingHead(self._get_lstm_features(outputs2[0]))

        cosinesim_fct = nn.CosineSimilarity(dim=-1, eps=1e-6)
        logits = cosinesim_fct(emb_1, emb_2)

        if labels is not None:
            loss_fct = torch.nn.CosineEmbeddingLoss()
            
            # [0, 1] => [-1, 1]
            normalized_label = (labels * 2 - 1).float()

            loss = loss_fct(emb_1, emb_2, normalized_label)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSim(RobertaForSequenceClassification):

    def __init__(self, config):
        super(RobertaEmbSim,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 200)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

        # try separate embeddings
        emb_1 = self.embeddingHead(outputs1[0][:, 0, :])

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)

        # try separate embeddings
        emb_2 = self.embeddingHead(outputs2[0][:, 0, :])

        cosinesim_fct = nn.CosineSimilarity(dim=-1, eps=1e-7)
        logits = cosinesim_fct(emb_1, emb_2)

        if labels is not None:
            # loss_fct = torch.nn.MSELoss() # tested, doesn't seem to perform as well as the built in embedding loss
            loss_fct = torch.nn.CosineEmbeddingLoss(margin=0.2)
            
            # [0, 1] => [-1, 1]
            normalized_label = (labels * 2 - 1).float()

            loss = loss_fct(emb_1, emb_2, normalized_label)
            # loss = loss_fct(logits, normalized_label)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimV3(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

        self.norm = nn.LayerNorm(200)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = complex_emb1[:, :, 2, :] * mask1
        value2 = complex_emb2[:, :, 2, :] * mask2

        # use symetrical attention
        act = nn.Sigmoid()
        
        a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb1 + emb2) / 2

        value_agg = self.norm(value_agg)

        # output should be [B, 1]
        logits = self.score_out(value_agg.squeeze(1))

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimV3_LinearDirectional(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_LinearDirectional,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = LinearScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = complex_emb1[:, :, 2, :] * mask1
        value2 = complex_emb2[:, :, 2, :] * mask2

        # use symetrical attention
        act = nn.Sigmoid()
        
        a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        # value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        # emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = emb2 #(emb1 + emb2) / 2

        # output should be [B, 1]
        logits = self.score_out(value_agg.squeeze(1))

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV5(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV5,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        #self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        #self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        #self.downsample3 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        # this uses softmax 
        self.attention_sim = transformers.modeling_bert.BertSelfAttention(config = config)

        self.score_out = ScoringHead(config, config.hidden_size , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    # def _self_run_cnn(self, sequence_out):
    #     # sequence_out [batch, len, embedding]
    #     sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
    #     x = sequence_out
    #     # TODO: better compression experiment needed
    #     x = self.downsample1(x)
    #     # x = self.downsample1(x) # repeat same downsampler exp
    #     x = self.downsample2(x)
    #     x = self.downsample3(x) # group is 3

    #     x = torch.transpose(x, 1, 2)
    #     return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = outputs1[0] #self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        # [batchS, compressed_len, embeddingS] = compressed_output1.size()
        # complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = compute_attention_mask(attention_mask_a, self)  #self._downsample_mask(attention_mask_a, 1).unsqueeze(2).float()
        
        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = outputs2[0] #self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        # complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = compute_attention_mask(attention_mask_b, self)  #self._downsample_mask(attention_mask_b, 1).unsqueeze(2).float()

        output_sim = self.attention_sim(hidden_states = compressed_output1,
                            attention_mask=mask1,
                            encoder_hidden_states=compressed_output2,
                            encoder_attention_mask=mask2)

        value_agg = output_sim[0][:,0,:]
        value_agg = self.norm(value_agg)

        # output should be [B, 1]
        logits = self.score_out(value_agg.squeeze(1))

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimV6(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV6,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        #self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        #self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        #self.downsample3 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        # this uses softmax 
        self.attention_sim = transformers.modeling_bert.BertSelfAttention(config = config)

        self.score_out = ScoringHead(config, config.hidden_size , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    # def _self_run_cnn(self, sequence_out):
    #     # sequence_out [batch, len, embedding]
    #     sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
    #     x = sequence_out
    #     # TODO: better compression experiment needed
    #     x = self.downsample1(x)
    #     # x = self.downsample1(x) # repeat same downsampler exp
    #     x = self.downsample2(x)
    #     x = self.downsample3(x) # group is 3

    #     x = torch.transpose(x, 1, 2)
    #     return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()
        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))
        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = outputs1[0] #self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        # [batchS, compressed_len, embeddingS] = compressed_output1.size()
        # complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        # mask1 = compute_attention_mask(attention_mask_a, self)  #self._downsample_mask(attention_mask_a, 1).unsqueeze(2).float()
        
        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = outputs2[0] #self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        # complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        # mask2 = compute_attention_mask(attention_mask_b, self)  #self._downsample_mask(attention_mask_b, 1).unsqueeze(2).float()
        
        compressed_output = torch.cat([compressed_output1, compressed_output2], dim=1)
        mask = torch.cat([attention_mask_a, attention_mask_b], dim=1)

        mask = compute_attention_mask(mask, self)

        output_sim = self.attention_sim(hidden_states = compressed_output,
                            attention_mask=mask)

        value_agg = output_sim[0][:,0,:]
        value_agg = self.norm(value_agg)

        # output should be [B, 1]
        logits = self.score_out(value_agg.squeeze(1))

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3N(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3N,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        with torch.no_grad():
            outputs1 = self.roberta(input_ids=input_ids_a,
                                attention_mask=attention_mask_a)

            outputs2 = self.roberta(input_ids=input_ids_b,
                    attention_mask=attention_mask_b)

        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())


        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = complex_emb1[:, :, 2, :] * mask1
        value2 = complex_emb2[:, :, 2, :] * mask2

        # use symetrical attention
        act = nn.Sigmoid()
        
        a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb1 + emb2) / 2
        value_agg = self.norm(value_agg)

        # output should be [B, 1]
        logits = self.score_out(value_agg.squeeze(1))

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_Linear(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_Linear,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = complex_emb1[:, :, 2, :] * mask1
        value2 = complex_emb2[:, :, 2, :] * mask2

        # use symetrical attention
        act = nn.Tanh()
        
        a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb1 + emb2) / 2
        value_agg = value_agg

        # output should be [B, 1]
        logits = self.score_out(value_agg.squeeze(1))

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class BertEmbSimV3_LinearEmb(BertForSequenceClassification):
    def __init__(self, config):
        super(BertEmbSimV3_LinearEmb,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.bert(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.bert(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = self.score_out(complex_emb1[:, :, 2, :])  * mask1
        value2 = self.score_out(complex_emb2[:, :, 2, :])  * mask2

        # use symetrical attention
        act = nn.Sigmoid()
        
        a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb1 + emb2) / 2
        value_agg = value_agg

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)
    
class RobertaMeanEmbedTripleLoss(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaMeanEmbedTripleLoss,self).__init__(config)
        self.triplet_margin = 1e-5
    
    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def mean_embedding(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        outputs_q = self.roberta(input_ids=query_ids,
                            attention_mask=attention_mask_q)

        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        
        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)

        mean_emb_q = self.masked_mean(outputs_q[0], attention_mask_q)
        mean_emb1 = self.masked_mean(outputs1[0], attention_mask_a)
        mean_emb2 = self.masked_mean(outputs2[0], attention_mask_b)

        return mean_emb_q, mean_emb1, mean_emb2

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        mean_emb_q, mean_emb1, mean_emb2 = self.mean_embedding(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        distance_pos = 1 - F.cosine_similarity(mean_emb_q, mean_emb1)
        distance_neg = 1 - F.cosine_similarity(mean_emb_q, mean_emb2)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return (losses.mean(),)

class RobertaMeanEmbedTripleLoss200(RobertaMeanEmbedTripleLoss):
    def __init__(self, config):
        super(RobertaMeanEmbedTripleLoss200,self).__init__(config)
        self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 200)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # with torch.no_grad():
        mean_emb_q, mean_emb1, mean_emb2 = self.mean_embedding(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        mean_emb_q = self.embeddingHead(mean_emb_q)
        

        distance_pos = 1 - F.cosine_similarity(mean_emb_q, mean_emb1)
        distance_neg = 1 - F.cosine_similarity(mean_emb_q, mean_emb2)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return (losses.mean(),)            

class RobertaEmbSimV3_Directional(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_Directional,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        # value1 = self.score_out(complex_emb1[:, :, 2, :])  * mask1
        value2 = self.score_out(complex_emb2[:, :, 2, :])  * mask2

        # use symetrical attention
        act = nn.Sigmoid()
        
        # a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        # value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        # emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = emb2

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimV3_DirectionalSym(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_DirectionalSym,self).__init__(config)
       
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = self.score_out(complex_emb1[:, :, 2, :])  * mask1
        value2 = self.score_out(complex_emb2[:, :, 2, :])  * mask2

        # use symetrical attention
        act = nn.Sigmoid()
        
        a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb2 + emb1) / 2

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_DirectionalBound(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_DirectionalBound,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] 
        query1 = query1 / torch.norm(query1, p=2, dim=-1, keepdim=True) # output should be [batch, compressed_len, emb/3]
        # query2 = complex_emb2[:, :, 0, :] 

        # key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :] 
        key2 = key2 / torch.norm(key2, p=2, dim=-1, keepdim=True)

        # zero out values of padding
       #  value1 = self.score_out(complex_emb1[:, :, 2, :])  * mask1
        value2 = self.score_out(complex_emb2[:, :, 2, :])  * mask2

        # use symetrical attention
        #act = nn.Sigmoid()
        #value1 = value1

        #a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2], L2 normalized, becomes cosine sim
        a21 = torch.matmul(key2, query1.transpose(1,2))
        
        #value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        # emb1 = value12[:, 0, :]

        # single logit bounded minimization instead of maximization [-1 to N - 1]
        # emb2 = value12[:, 0, :]
        value_agg = value21[:, 0, :]

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbDotSim_Diff_Encoder(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbDotSim_Diff_Encoder,self).__init__(config)
        self.dot_encoder = RobertaForSequenceClassification.from_pretrained("distilroberta-base")

        self.embeddingHead1 = ClickEmbeddingHead(config, embeddingDim = 200)
        self.embeddingHead2 = ClickEmbeddingHead(config, embeddingDim = 200)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.dot_encoder.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        
        # try separate embeddings
        emb_1 = self.embeddingHead1(outputs1[0][:, 0, :])

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)

        # try separate embeddings
        emb_2 = self.embeddingHead2(outputs2[0][:, 0, :])

        #cosinesim_fct = nn.CosineSimilarity(dim=-1, eps=1e-6)
        dot_score = torch.matmul(emb_1.unsqueeze(1), emb_2.unsqueeze(2))

        # act = nn.Sigmoid()
        # dot_prob = act(dot_score)

        logits = dot_score.squeeze().squeeze()

        if labels is not None:
            #loss_fct = MSELoss()
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbDotSim_Diff_FullEncoder(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbDotSim_Diff_FullEncoder,self).__init__(config)
        self.dot_encoder = RobertaForSequenceClassification.from_pretrained("roberta-base")

        self.embeddingHead1 = ClickEmbeddingHead(config, embeddingDim = 200)
        self.embeddingHead2 = ClickEmbeddingHead(config, embeddingDim = 200)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.dot_encoder.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        
        # try separate embeddings
        emb_1 = self.embeddingHead1(outputs1[0][:, 0, :])

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)

        # try separate embeddings
        emb_2 = self.embeddingHead2(outputs2[0][:, 0, :])

        #cosinesim_fct = nn.CosineSimilarity(dim=-1, eps=1e-6)
        dot_score = torch.matmul(emb_1.unsqueeze(1), emb_2.unsqueeze(2))

        # act = nn.Sigmoid()
        # dot_prob = act(dot_score)

        logits = dot_score.squeeze().squeeze()

        if labels is not None:
            #loss_fct = MSELoss()
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbDotSim(RobertaForSequenceClassification):

    def __init__(self, config):
        super(RobertaEmbDotSim,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)

        # initialize second roberta

        self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 200)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

        # try separate embeddings
        emb_1 = self.embeddingHead(outputs1[0][:, 0, :])

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)

        # try separate embeddings
        emb_2 = self.embeddingHead(outputs2[0][:, 0, :])

        #cosinesim_fct = nn.CosineSimilarity(dim=-1, eps=1e-6)
        dot_score = torch.matmul(emb_1.unsqueeze(1), emb_2.unsqueeze(2))

        # act = nn.Sigmoid()
        # dot_prob = act(dot_score)

        logits = dot_score.squeeze().squeeze()

        if labels is not None:
            #loss_fct = MSELoss()
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_Dot(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_Dot,self).__init__(config)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = LinearScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = complex_emb1[:, :, 2, :] * mask1
        value2 = complex_emb2[:, :, 2, :] * mask2

        # use symetrical attention
        # act = nn.Sigmoid()
        
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        a21 = torch.matmul(query2, key1.transpose(1,2))
        
        value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb1 + emb2) / 2

        # output should be [B, 1]
        logits = self.score_out(value_agg.squeeze(1))

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_PureDot(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDot,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        #self.score_out = LinearScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]
        
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        a21 = torch.matmul(query2, key1.transpose(1,2))

        emb1 = a12[:, 0, 0].unsqueeze(1)
        emb2 = a21[:, 0, 0].unsqueeze(1)

        value_agg = emb1 + emb2
        value_agg = value_agg / 2

        logits = value_agg.squeeze().squeeze()

        if labels is not None:
            #loss_fct = MSELoss()
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimV3_PureDot_NoWindow(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDot_NoWindow,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        self.norm = nn.LayerNorm(200)
        
        # self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 1, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        # self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 1, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        #self.score_out = LinearScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        # x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        # x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]
        
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        a21 = torch.matmul(query2, key1.transpose(1,2))

        emb1 = a12[:, 0, 0].unsqueeze(1)
        emb2 = a21[:, 0, 0].unsqueeze(1)

        value_agg = emb1 + emb2
        value_agg = value_agg / 2

        logits = value_agg.squeeze().squeeze()

        if labels is not None:
            #loss_fct = MSELoss()
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_PureDotDirectional(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional,self).__init__(config)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]
        
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        # a21 = torch.matmul(query2, key1.transpose(1,2))

        emb1 = a12[:, 0, 0].unsqueeze(1)
        # emb2 = a21[:, 0, 0].unsqueeze(1)

        value_agg = emb1 #+ emb2
        # value_agg = value_agg / 2

        logits = value_agg.squeeze().squeeze()

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_PureDotDirectional_BOW(RobertaEmbSimV3_PureDotDirectional):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_BOW, self).__init__(config)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()
        
        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        key2 = complex_emb2[:, :, 1, :] * mask2
        
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]

        emb1 = a12[:, 0, :].unsqueeze(1).sum(dim = -1, keepdim = True) / mask2.sum(dim = -1, keepdim = True)

        value_agg = emb1 
        logits = value_agg.squeeze().squeeze()

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_QQ(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_QQ,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        self.norm = nn.LayerNorm(200)
        
        # self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        # self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 200, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        #self.score_out = LinearScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        # x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        # x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        # complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))
        # mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        # complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))
        # mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = compressed_output1[:, :, :] # output should be [batch, compressed_len, emb/3]
        query2 = compressed_output2[:, :, :]

        # key1 = complex_emb1[:, :, 1, :]
        # key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        # value1 = complex_emb1[:, :, 2, :] * mask1
        # value2 = complex_emb2[:, :, 2, :] * mask2

        # use symetrical attention
        # act = nn.Sigmoid()
        
        a12 = torch.matmul(query1, query2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        
        #value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        #value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = a12[:, 0, 0].unsqueeze(1)

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = emb1
        value_agg = value_agg
        logits = torch.cat([-value_agg, value_agg], dim = 1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertEmbSimV3_WeightedCosine(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertEmbSimV3_WeightedCosine,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        #act = nn.Sigmoid()

        value1 = self.score_out(complex_emb1[:, :, 2, :]) * mask1
        value2 = self.score_out(complex_emb2[:, :, 2, :]) * mask2        
        
        # a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        # a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        # try first token cosine-sim
        cos_sim_fn = nn.CosineSimilarity(dim=-1, eps=1e-6)

        query1 = query1[:,0,:]
        key2 = key2[:,0,:]
        value2 = value2[:,0,:]

        cos12 = cos_sim_fn(query1, key2).unsqueeze(1)
        emb2 = cos12 * value2

        query2 = query2[:,0,:]
        key1 = key1[:,0,:]
        value1 = value1[:,0,:]

        cos21 = cos_sim_fn(query2, key1).unsqueeze(1)
        emb1 = cos21 * value1


        # value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        # value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        # emb1 = value12[:, 0, :]
        # emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb1 + emb2) / 2
        value_agg = value_agg

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_PureDotBOW(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotBOW,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        #self.score_out = LinearScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()
        
        # query, key, value
        query1 = complex_emb1[:, :, 0, :] * mask1 # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :] * mask2

        key1 = complex_emb1[:, :, 1, :] * mask1
        key2 = complex_emb2[:, :, 1, :] * mask2

        # zero out values of padding
        # value1 = complex_emb1[:, :, 2, :] * mask1
        # value2 = complex_emb2[:, :, 2, :] * mask2

        # use symetrical attention
        # act = nn.Sigmoid()
        
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        a21 = torch.matmul(query2, key1.transpose(1,2))
        
        #value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        #value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        # emb1 = a12[:, 0, 0].unsqueeze(1)
        # emb2 = a21[:, 0, 0].unsqueeze(1)

        emb1 = a12[:, 0, :].sum(1, keepdim = True)  / mask2.sum(1)
        emb2 = a21[:, 0, :].sum(1, keepdim = True)  / mask1.sum(1)

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = emb1 + emb2
        value_agg = value_agg / 2
        logits = torch.cat([-value_agg, value_agg], dim = 1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)



class RobertaEmbSimV3_Directional2(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_Directional2,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        # value1 = self.score_out(complex_emb1[:, :, 2, :])  * mask1
        act = nn.Sigmoid()
        value2 = act(self.score_out(complex_emb2[:, :, 2, :]))  * mask2

        # use symetrical attention
        
        # a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = torch.matmul(query2, key1.transpose(1,2))
        
        # value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        # emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = emb2

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertEmbSimV3_WeightedCosineDirectional(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertEmbSimV3_WeightedCosineDirectional,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        # query2 = complex_emb2[:, :, 0, :]

        # key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        #act = nn.Sigmoid()

        # value1 = self.score_out(complex_emb1[:, :, 2, :])  * mask1
        value2 = self.score_out(complex_emb2[:, :, 2, :])  * mask2        
        
        # a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        # a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        # try first token cosine-sim

        cos_sim_fn = nn.CosineSimilarity(dim=-1, eps=1e-6)

        query1 = query1[:,0,:]
        key2 = key2[:,0,:]
        value2 = value2[:,0,:]

        cos12 = cos_sim_fn(query1, key2).unsqueeze(1)
        emb2 = cos12 * value2

        # query2 = query2[:,0,:]
        # key1 = key1[:,0,:]
        # value1 = value1[:,0,:]

        # cos21 = cos_sim_fn(query2, key1).unsqueeze(1)
        # emb1 = cos21 * value1


        # value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        # value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        # emb1 = value12[:, 0, :]
        # emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        # value_agg = (emb1 + emb1) / 2
        value_agg = emb2

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimV3_LinearEmb(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_LinearEmb,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        # print(complex_emb1.size())
        # print(mask1.size())

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]
        
        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]

        # zero out values of padding
        value1 = self.score_out(complex_emb1[:, :, 2, :])  * mask1
        value2 = self.score_out(complex_emb2[:, :, 2, :])  * mask2

        # use symetrical attention
        act = nn.Sigmoid()
        
        a12 = act(torch.matmul(query1, key2.transpose(1,2))) # [batch, compressed len1, compressed len2]
        a21 = act(torch.matmul(query2, key1.transpose(1,2)))
        
        value12 = torch.matmul(value1.transpose(1,2), a12).transpose(1,2)
        value21 = torch.matmul(value2.transpose(1,2), a21).transpose(1,2)

        emb1 = value12[:, 0, :]
        emb2 = value21[:, 0, :]

        # mean doesn't work well probably because attention mask is not properly implemented
        # emb1 = value12.mean(1)
        # emb2 = value21.mean(1)

        # TODO: try this out
        # emb1 = value12.sum(1) / mask1.sum(1)
        # emb2 = value21.sum(1) / mask2.sum(1)
        
        # output should be [B, emb/3]
        # value_agg = complex_emb1[:,2,:] * query_key_weight + 
        value_agg = (emb1 + emb2) / 2
        value_agg = value_agg

        # output should be [B, 1]
        logits = value_agg.squeeze(1)

        if labels is not None:
            #loss_fct = MSELoss()
            # try 2 class CrossEntropy?
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimDiff(RobertaForSequenceClassification):

    def __init__(self, config):
        super(RobertaEmbSimDiff, self).__init__(config)
        self.dot_encoder = RobertaForSequenceClassification.from_pretrained("distilroberta-base")
        self.embeddingHead1 = ClickEmbeddingHead(config, embeddingDim = 200)
        self.embeddingHead2 = ClickEmbeddingHead(config, embeddingDim = 200)

        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.dot_encoder.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

        # try separate embeddings
        emb_1 = self.embeddingHead1(outputs1[0][:, 0, :])

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)

        # try separate embeddings
        emb_2 = self.embeddingHead2(outputs2[0][:, 0, :])

        cosinesim_fct = nn.CosineSimilarity(dim=-1, eps=1e-7)
        logits = cosinesim_fct(emb_1, emb_2)

        if labels is not None:
            # loss_fct = torch.nn.MSELoss() # tested, doesn't seem to perform as well as the built in embedding loss
            loss_fct = torch.nn.CosineEmbeddingLoss(margin=0.2)
            
            # [0, 1] => [-1, 1]
            normalized_label = (labels * 2 - 1).float()

            loss = loss_fct(emb_1, emb_2, normalized_label)
            # loss = loss_fct(logits, normalized_label)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)



class RobertaEmbSimV3_DirectionalBoundExpo(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_DirectionalBoundExpo,self).__init__(config)
        # self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 100)
        #self.embeddingHead = ClickEmbeddingHead(config, embeddingDim = 600)
        # self.norm = nn.LayerNorm(200)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 600, 3, stride=2, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.score_out = ScoringHead(config, 200 , 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        # TODO: better compression experiment needed
        x = self.downsample1(x)
        # x = self.downsample1(x) # repeat same downsampler exp
        x = self.downsample2(x)
        x = self.downsample3(x) # group is 3

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 3, embeddingS//3))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 3, embeddingS//3))
        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] 
        query1 = query1 / torch.norm(query1, p=2, dim=-1, keepdim=True) # output should be [batch, compressed_len, emb/3]

        key2 = complex_emb2[:, :, 1, :] 
        key2 = key2 / torch.norm(key2, p=2, dim=-1, keepdim=True)

        # zero out values of padding
        value2 = self.score_out(complex_emb2[:, :, 2, :])  * mask2

        #value2 = torch.exp(value2)
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2], L2 normalized, becomes cosine sim
        
        value12 = torch.matmul(a12, value2)
        value_agg = value12[:, 0, :]

        # output should be [B, 2]
        logits = torch.exp(value_agg.squeeze(1))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)



# --------------------------------------------------------------------------
# Long sequence model
# --------------------------------------------------------------------------
def nms_per_batch(keys, key_strength_index, key_strength_value, threshold = 0.8, max_size = 10):
    # keys: [len, emb]
    # key_strength_index: [topk]
    # key_strength_value: [len, 1] L2 norm

    [_, embS] = keys.size()
    [topk] = key_strength_index.size()

    effective_vectors = keys[key_strength_index, :] # [topk, emb]
    effective_strength = key_strength_value[key_strength_index, :]

    cosine_sim_matrix = torch.matmul(effective_vectors, effective_vectors.transpose(0,1)) # [topk -> topk] cosine sim

    tensor_to_keep_idx = []

    for i in range(topk):
        keep = True
        for j in tensor_to_keep_idx:
            if cosine_sim_matrix[j, i] >= threshold:
                keep = False

        if keep:
            tensor_to_keep_idx.append(i)

        if len(tensor_to_keep_idx) >= max_size:
            break

    nms_vectors = effective_vectors[tensor_to_keep_idx, :]
    nms_values = effective_strength[tensor_to_keep_idx, :]

    return nms_vectors, nms_values


class RobertaEmbSimV3_PDD_MAX(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PDD_MAX,self).__init__(config)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

                            
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()


        # multiple chunks of bert
        [batchS, full_length] = input_ids_b.size()

        # TODO: tune this factor
        chunk_factor = 4 # chunk the model

        input_seq = input_ids_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs2 = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        [batch_expand, compressed_len, embeddingS] = compressed_output2.size()
        complex_emb2 = compressed_output2.reshape(batch_expand, compressed_len, 2, embeddingS//2).reshape(batchS, chunk_factor * compressed_len, 2, embeddingS//2)

        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, :, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :]

        key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]
        
        a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        # a21 = torch.matmul(query2, key1.transpose(1,2))

        emb1 = torch.max(a12[:, 0, :], dim = -1).values

        # emb2 = a21[:, 0, 0].unsqueeze(1)

        value_agg = emb1 #+ emb2
        # value_agg = value_agg / 2

        logits = value_agg

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_PDD_NMS_MAX(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PDD_NMS_MAX,self).__init__(config)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

                            
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()


        # multiple chunks of bert
        [batchS, full_length] = input_ids_b.size()

        # TODO: tune this factor
        chunk_factor = 4 # chunk the model

        input_seq = input_ids_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs2 = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        [batch_expand, compressed_len, embeddingS] = compressed_output2.size()
        complex_emb2 = compressed_output2.reshape(batch_expand, compressed_len, 2, embeddingS//2).reshape(batchS, chunk_factor * compressed_len, 2, embeddingS//2)

        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, 0, 0, :] # only use first token, output should be [batch, emb/3]
        # query2 = complex_emb2[:, :, 0, :]

        # key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]
        

        # simulate a NMS process on the key2
        query1_vector = query1 / torch.norm(query1, p=2, dim=-1, keepdim=True)

        key2_strength =  torch.norm(key2, p=2, dim=-1, keepdim=True) # [batch, compressed_len, 1]
        key2_vector = key2 / key2_strength

        #a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        
        # perform NMS - first, select at most 20 vectors (100 * 8 ~ =)
        # TODO: change to hyper param
        NMS_Limit_1 = 100 # purely for computational efficiency
        NMS_Cosine_Sim_Threshold = 0.8
        NMS_Limit_2 = 20

        _, key2_strength_topk_idx = torch.topk(key2_strength, k=NMS_Limit_1, dim=1, largest=True, sorted=True)
        key2_strength_topk_idx = key2_strength_topk_idx.squeeze(-1) #[B, topK]

        # just use it as a place holder, all values should be overwritten
        logits_list = []
        for b in range(batchS):
            per_record_key_strength_idx = key2_strength_topk_idx[b, :]
            per_record_key_strength_val = key2_strength[b, :, :]
            per_record_key_vector = key2_vector[b, :, :]

            nms_vectors, nms_values = nms_per_batch(per_record_key_vector, per_record_key_strength_idx, per_record_key_strength_val, threshold = NMS_Cosine_Sim_Threshold, max_size = NMS_Limit_2)

            # select only top N vectors at most
            nms_vectors = nms_vectors[:NMS_Limit_2, :] #[NMS_Limit_2, emb]
            nms_values = nms_values[:NMS_Limit_2, :] #[NMS_Limit_2, 1]

            # query term
            per_record_query = query1_vector[b, :].unsqueeze(0) #[1, emb]

            # try out different aggregation matrix here:
            nms_vectors_effective = nms_vectors * nms_values
            # print("nms_vectors_effective size", nms_vectors_effective.size())  #[NMS_Limit_2, emb]
            # print("per_record_query size", per_record_query.size())
            _scores = torch.matmul(per_record_query, nms_vectors_effective.transpose(0,1)) # [1, NMS_Limit_2] 
            logits_list.append(torch.max(_scores, dim = -1).values)

        logits = torch.cat(logits_list, 0)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)

class RobertaEmbSimV3_PDD_NMS_SUM(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PDD_NMS_SUM,self).__init__(config)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]


    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels=None):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

                            
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))
        mask1 = self._downsample_mask(attention_mask_a, 8).unsqueeze(2).float()


        # multiple chunks of bert
        [batchS, full_length] = input_ids_b.size()

        # TODO: tune this factor
        chunk_factor = 4 # chunk the model

        input_seq = input_ids_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs2 = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        [batch_expand, compressed_len, embeddingS] = compressed_output2.size()
        complex_emb2 = compressed_output2.reshape(batch_expand, compressed_len, 2, embeddingS//2).reshape(batchS, chunk_factor * compressed_len, 2, embeddingS//2)

        mask2 = self._downsample_mask(attention_mask_b, 8).unsqueeze(2).float()

        # query, key, value
        query1 = complex_emb1[:, 0, 0, :] # only use first token, output should be [batch, emb/3]
        # query2 = complex_emb2[:, :, 0, :]

        # key1 = complex_emb1[:, :, 1, :]
        key2 = complex_emb2[:, :, 1, :]
        

        # simulate a NMS process on the key2
        query1_vector = query1 / torch.norm(query1, p=2, dim=-1, keepdim=True)

        key2_strength =  torch.norm(key2, p=2, dim=-1, keepdim=True) # [batch, compressed_len, 1]
        key2_vector = key2 / torch.norm(key2, p=2, dim=-1, keepdim=True) 

        #a12 = torch.matmul(query1, key2.transpose(1,2)) # [batch, compressed len1, compressed len2]
        
        # perform NMS - first, select at most 20 vectors (100 * 8 ~ =)
        # TODO: change to hyper param
        NMS_Limit_1 = 100 # purely for computational efficiency
        NMS_Cosine_Sim_Threshold = 0.8
        NMS_Limit_2 = 20

        key2_strength_topk_val, key2_strength_topk_idx = torch.topk(key2_strength, k=NMS_Limit_1, dim=1, largest=True, sorted=True)
        key2_strength_topk_idx = key2_strength_topk_idx.squeeze(-1) #[B, topK]

        logits_list = []
        for b in range(batchS):
            per_record_key_strength_idx = key2_strength_topk_idx[b, :]
            per_record_key_strength_val = key2_strength[b, :, :]
            per_record_key_vector = key2_vector[b, :, :]

            nms_vectors, nms_values = nms_per_batch(per_record_key_vector, per_record_key_strength_idx, per_record_key_strength_val, threshold = NMS_Cosine_Sim_Threshold, max_size = NMS_Limit_2)

            # select only top N vectors at most
            nms_vectors = nms_vectors[:NMS_Limit_2, :] #[NMS_Limit_2, emb]
            nms_values = nms_values[:NMS_Limit_2, :] #[NMS_Limit_2, 1]

            # query term
            per_record_query = query1_vector[b, :].unsqueeze(0) #[1, emb]

            # try out different aggregation matrix here:
            nms_vectors_effective = nms_vectors * nms_values
            # print("nms_vectors_effective size", nms_vectors_effective.size())  #[NMS_Limit_2, emb]
            # print("per_record_query size", per_record_query.size())
            _scores = torch.matmul(per_record_query, nms_vectors_effective.transpose(0,1)) # [1, NMS_Limit_2] 
            logits_list.append(torch.sum(_scores, dim = -1))

        logits = torch.cat(logits_list, 0)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)

class EmbeddingGen(RobertaEmbSimV3_PureDotDirectional):
    def __init__(self, config):
        super(EmbeddingGen,self).__init__(config)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))

        # query, key, value
        query1 = complex_emb1[:, 0, 0, :] # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, 0, 0, :]

        key1 = complex_emb1[:, 0, 1, :]
        key2 = complex_emb2[:, 0, 1, :] # [B, E]

        return query1, key1, query2, key2

class EmbeddingGenDocs(RobertaEmbSimV3_PDD_MAX):
    def __init__(self, config):
        super(EmbeddingGenDocs,self).__init__(config)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        chunk_factor = 4 # chunk the model

        [batchS, full_length] = input_ids_a.size()
        input_seq_a = input_ids_a.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq_a = attention_mask_a.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        outputs1 = self.roberta(input_ids=input_seq_a,
                            attention_mask=attention_mask_seq_a)

        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batch_expand, compressed_len, embeddingS] = compressed_output1.size()
        # complex_emb1 = compressed_output1.reshape(batch_expand, compressed_len, 2, embeddingS//2).reshape(batchS, chunk_factor * compressed_len, 2, embeddingS//2)
        complex_emb1 = compressed_output1.reshape(batch_expand, compressed_len, 2, embeddingS//2)

        # multiple chunks of bert
        # [batchS, full_length] = input_ids_b.size()
        input_seq = input_ids_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask_b.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        outputs2 = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        complex_emb2 = compressed_output2.reshape(batch_expand, compressed_len, 2, embeddingS//2)

        # query, key, value
        query1 = complex_emb1[:, :, 0, :].reshape(batch_expand * compressed_len, embeddingS//2) # output should be [batch, compressed_len, emb/3]
        query2 = complex_emb2[:, :, 0, :].reshape(batch_expand * compressed_len, embeddingS//2)

        key1 = complex_emb1[:, :, 1, :].reshape(batch_expand * compressed_len, embeddingS//2)
        key2 = complex_emb2[:, :, 1, :].reshape(batch_expand * compressed_len, embeddingS//2) # [B, E]

        

        return query1, key1, query2, key2

class EmbeddingGenMean(RobertaForSequenceClassification):
    def __init__(self, config):
        super(EmbeddingGenMean,self).__init__(config)
    
    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        outputs1 = self.roberta(input_ids=input_ids_a,
                            attention_mask=attention_mask_a)

        outputs2 = self.roberta(input_ids=input_ids_b,
                            attention_mask=attention_mask_b)
                            
        query1 = self.masked_mean(outputs1[0], attention_mask_a)
        query1 = F.normalize(query1, p=2, dim=1)
        query2 = self.masked_mean(outputs2[0], attention_mask_b)
        query2 = F.normalize(query2, p=2, dim=1)

        return query1, query1, query2, query2

# -------------------------------- ANN ---------------------------------------

class RobertaEmbSimV3_PureDotDirectionalANN(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectionalANN,self).__init__(config)
        
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]

    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))

        query1 = complex_emb1[:, 0, 0, :]
        return query1

    def body_emb(self, input_ids, attention_mask):
        outputs2 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output2.size()
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))
        
        key2 = complex_emb2[:, 0, 1, :]
        return key2


    def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(input_ids_a, attention_mask_a)
        elif input_ids_b is None:
            return self.body_emb(input_ids_a, attention_mask_a)
        
        # query, key, value
        query1 = self.query_emb(input_ids_a, attention_mask_a) # output should be [batch, emb/3]
        key2 = self.body_emb(input_ids_b, attention_mask_b)

        a12 = torch.matmul(query1.unsqueeze(1), key2.unsqueeze(2)) # [batch, 1, 1]

        logits = a12[:, 0, 0]

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)

# -----------------------------------------------------------------------------------------------------------------
# Triplet loss only model -----------------------------------------------------------------------------------------

class RobertaMeanEmbedTripleLossANN(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaMeanEmbedTripleLossANN,self).__init__(config)
        self.triplet_margin = 1e-5
    
    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)

        mean_emb_q = self.masked_mean(outputs1[0], attention_mask)
        
        # perform L2 norm
        mean_emb_q = F.normalize(mean_emb_q, p=2, dim=1)
        return mean_emb_q

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


    def mean_embedding(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # outputs_q = self.roberta(input_ids=query_ids,
        #                     attention_mask=attention_mask_q)

        # outputs1 = self.roberta(input_ids=input_ids_a,
        #                     attention_mask=attention_mask_a)
        
        # outputs2 = self.roberta(input_ids=input_ids_b,
        #                     attention_mask=attention_mask_b)

        mean_emb_q = self.query_emb(query_ids, attention_mask_q)
        mean_emb1 = self.body_emb(input_ids_a, attention_mask_a)
        mean_emb2 = self.body_emb(input_ids_b, attention_mask_b)

        return mean_emb_q, mean_emb1, mean_emb2

    #def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        mean_emb_q, mean_emb1, mean_emb2 = self.mean_embedding(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        distance_pos = 1 - F.cosine_similarity(mean_emb_q, mean_emb1)
        distance_neg = 1 - F.cosine_similarity(mean_emb_q, mean_emb2)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return (losses.mean(),)


class RobertaMeanEmbedTripleLoss200ANN(RobertaMeanEmbedTripleLossANN):
    def __init__(self, config):
        super(RobertaMeanEmbedTripleLossANN,self).__init__(config)
        self.embeddingHead = nn.Linear(config.hidden_size, 200)

        self.triplet_margin = 1e-5
    
    
    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)

        mean_emb_q = self.masked_mean(outputs1[0], attention_mask)
        mean_emb_q = self.embeddingHead(mean_emb_q)
        # perform L2 norm
        mean_emb_q = F.normalize(mean_emb_q, p=2, dim=1)
        return mean_emb_q

class RobertaMeanEmbedTripleLoss200ANN_Inference(RobertaMeanEmbedTripleLoss200ANN):

    def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
    #def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(input_ids_a, attention_mask_a)
        elif input_ids_b is None:
            return self.body_emb(input_ids_a, attention_mask_a)

        return None

class SentenceBert200CE(RobertaMeanEmbedTripleLoss200ANN):
    def __init__(self, config):
        super(SentenceBert200CE,self).__init__(config)
        self.embeddingHead = nn.Linear(config.hidden_size, 200)

        self.triplet_margin = 1e-5
    
    
    def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(input_ids_a, attention_mask_a)
        elif input_ids_b is None:
            return self.body_emb(input_ids_a, attention_mask_a)
        
        # query, key, value
        query1 = self.query_emb(input_ids_a, attention_mask_a) # output should be [batch, emb/3]
        key2 = self.body_emb(input_ids_b, attention_mask_b)

        a12 = torch.matmul(query1.unsqueeze(1), key2.unsqueeze(2)) # [batch, 1, 1]

        logits = a12[:, 0, 0] * 5

        if labels is not None:
            # loss_fct = torch.nn.MSELoss()
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            # loss = loss_fct(logits, labels.float() * 2 - 1)

            # loss_fct = torch.nn.CosineEmbeddingLoss()
            # loss = loss_fct(query1, key2, labels.float())

            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_PureDotDirectional_Triple(RobertaEmbSimV3_PureDotDirectional):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_Triple,self).__init__(config)
        self.triplet_margin = 0.5

    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]

    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs_q = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)

        compressed_output_q = self._self_run_cnn(outputs_q[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output_q.size()
        complex_emb_q = torch.reshape(compressed_output_q, (batchS, compressed_len, 2, embeddingS//2))
        query = complex_emb_q[:, 0, 0, :] # output should be [batch, compressed_len, emb/3]
        return query

    def body_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs_k = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)

        compressed_output_k = self._self_run_cnn(outputs_k[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output_k.size()
        complex_emb_k = torch.reshape(compressed_output_k, (batchS, compressed_len, 2, embeddingS//2))
        
        key = complex_emb_k[:, 0, 1, :]
        return key
        
    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        query = self.query_emb(query_ids, attention_mask_q)
        key1 = self.body_emb(input_ids_a, attention_mask_a)
        key2 = self.body_emb(input_ids_b, attention_mask_b)

        r,c = query.shape[0], query.shape[1]
        sim_pos = torch.matmul(query.reshape(r, 1, c), key1.reshape(r, c, 1))
        sim_neg = torch.matmul(query.reshape(r, 1, c), key2.reshape(r, c, 1))

        losses = F.relu(sim_neg - sim_pos + self.triplet_margin)
        return (losses.mean(),)


class RobertaEmbSimV3_PureDotDirectional_Triple_Mean(RobertaEmbSimV3_PureDotDirectional_Triple):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_Triple_Mean,self).__init__(config)
        self.triplet_margin = 0.2

    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs_q = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)

        compressed_output_q = self._self_run_cnn(outputs_q[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output_q.size()
        complex_emb_q = torch.reshape(compressed_output_q, (batchS, compressed_len, 2, embeddingS//2))
        attention_mask = self._downsample_mask(attention_mask, 8)
        query = self.masked_mean(complex_emb_q[:, :, 0, :], attention_mask) # output should be [batch, compressed_len, emb/3]
        return query

    def body_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs_k = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)

        compressed_output_k = self._self_run_cnn(outputs_k[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output_k.size()
        complex_emb_k = torch.reshape(compressed_output_k, (batchS, compressed_len, 2, embeddingS//2))
        attention_mask = self._downsample_mask(attention_mask, 8)
        key = self.masked_mean(complex_emb_k[:, :, 1, :], attention_mask)
        return key

    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        query = self.query_emb(query_ids, attention_mask_q)
        key1 = self.body_emb(input_ids_a, attention_mask_a)
        key2 = self.body_emb(input_ids_b, attention_mask_b)

        r,c = query.shape[0], query.shape[1]

        loss_func = nn.LogSigmoid()

        dist_pos = -loss_func(torch.matmul(query.reshape(r, 1, c), key1.reshape(r, c, 1))) # higher score ->lower dist
        dist_neg = -loss_func(torch.matmul(query.reshape(r, 1, c), key2.reshape(r, c, 1)))
        
        losses = F.relu(dist_pos - dist_neg + self.triplet_margin)

        return (losses.mean(),)


class RobertaEmbSimV3_PureDotDirectional_Mean(RobertaEmbSimV3_PureDotDirectional_Triple_Mean):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_Mean,self).__init__(config)
        

    def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(input_ids_a, attention_mask_a)
        elif input_ids_b is None:
            return self.body_emb(input_ids_a, attention_mask_a)
        
        # query, key, value
        query1 = self.query_emb(input_ids_a, attention_mask_a) # output should be [batch, emb/3]
        key2 = self.body_emb(input_ids_b, attention_mask_b)

        a12 = torch.matmul(query1.unsqueeze(1), key2.unsqueeze(2)) # [batch, 1, 1]

        logits = a12[:, 0, 0]

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)



class RobertaEmbSimV3_PureDotDirectional_Mean_Chunk(RobertaEmbSimV3_PureDotDirectional_Mean):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_Mean,self).__init__(config)
        
    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = 4 # chunk the model

        input_seq = input_ids.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output_k = self._self_run_cnn(outputs_k[0]) # [batch, len/8, dim]

        [batch_expand, compressed_len, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(batch_expand, compressed_len, 2, embeddingS//2).reshape(batchS, chunk_factor * compressed_len, 2, embeddingS//2)

        attention_mask = self._downsample_mask(attention_mask, 8)
        key = self.masked_mean(complex_emb_k[:, :, 1, :], attention_mask)
        return key


class RobertaEmbSimV3_PureDotDirectional_CLF_Mean_Chunk(RobertaEmbSimV3_PureDotDirectionalANN):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_CLF_Mean_Chunk,self).__init__(config)
        
    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        
        base_len = 128 # chunk the model, 512 => 128 * 4
        chunk_factor = full_length // base_len

        input_seq = input_ids.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output_k = self._self_run_cnn(outputs_k[0]) # [batch, len/8, dim]

        [batch_expand, compressed_len, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(batch_expand, compressed_len, 2, embeddingS//2).reshape(batchS, chunk_factor, compressed_len, 2, embeddingS//2)
        attention_mask = self._downsample_mask(attention_mask, 8).reshape(batchS, chunk_factor, -1)

        complex_emb_k_CLF = complex_emb_k[:,:, 0, 1, :] # size [batchS, chunk_factor, embeddingS//2]
        attention_mask_CLF = attention_mask[:, :, 0] # size [batchS, chunk_factor]


        key = self.masked_mean(complex_emb_k_CLF, attention_mask_CLF)

        return key


class RobertaEmbSimV3_PureDotDirectional_CLF_MAX_Chunk(RobertaEmbSimV3_PureDotDirectionalANN):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_CLF_MAX_Chunk,self).__init__(config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output_k = self._self_run_cnn(outputs_k[0]) # [batch, len/8, dim]

        [batch_expand, compressed_len, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(batch_expand, compressed_len, 2, embeddingS//2).reshape(batchS, chunk_factor, compressed_len, 2, embeddingS//2)
        attention_mask = self._downsample_mask(attention_mask, 8).reshape(batchS, chunk_factor, -1)

        complex_emb_k_CLF = complex_emb_k[:, :, 0, 1, :] # size [batchS, chunk_factor, embeddingS//2]
        attention_mask_CLF = attention_mask[:, :, 0] # size [batchS, chunk_factor] => to recompute in forward

        return complex_emb_k_CLF # size [batchS, chunk_factor, embeddingS//2]

    def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(input_ids_a, attention_mask_a)
        elif input_ids_b is None:
            return self.body_emb(input_ids_a, attention_mask_a)
        
        # query, key, value
        query1 = self.query_emb(input_ids_a, attention_mask_a) # output should be [batch, emb/3]
        key2 = self.body_emb(input_ids_b, attention_mask_b) # size [batchS, chunk_factor, embeddingS//2]

        # special handle of attention mask -----
        [batchS, full_length] = input_ids_b.size()
        chunk_factor = full_length // self.base_len
        attention_mask_body = self._downsample_mask(attention_mask_b, 8).reshape(batchS, chunk_factor, -1)[:, :, 0] #[batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()
        # -------------------------------------

        a12 = torch.matmul(query1.unsqueeze(1), key2.transpose(1,2)) # [batch, 1, chunk_factor]
        # logits = (a12[:, 0, :] + inverted_bias).max(dim=-1)
        logits = (a12[:, 0, :] + inverted_bias).max(dim=-1, keepdim=False).values # [batch]

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)

        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaEmbSimV3_CLF_ANN(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEmbSimV3_CLF_ANN,self).__init__(config)
        self.downsample4 = nn.Conv1d(config.hidden_size, 400, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        x = self.downsample4(x)

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        #mask: [batch, len]
        [batchS, seq_len] = mask.size()

        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))

        return mask[:, :, 0]

    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        # try separate embeddings
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))

        query1 = complex_emb1[:, 0, 0, :]
        return query1

    def body_emb(self, input_ids, attention_mask):
        outputs2 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        compressed_output2 = self._self_run_cnn(outputs2[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output2.size()
        complex_emb2 = torch.reshape(compressed_output2, (batchS, compressed_len, 2, embeddingS//2))
        
        key2 = complex_emb2[:, 0, 1, :]
        return key2


    def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(input_ids_a, attention_mask_a)
        elif input_ids_b is None:
            return self.body_emb(input_ids_a, attention_mask_a)
        
        # query, key, value
        query1 = self.query_emb(input_ids_a, attention_mask_a) # output should be [batch, emb/3]
        key2 = self.body_emb(input_ids_b, attention_mask_b)

        a12 = torch.matmul(query1.unsqueeze(1), key2.unsqueeze(2)) # [batch, 1, 1]

        logits = a12[:, 0, 0]

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)


class RobertaDot_CLF_ANN(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaDot_CLF_ANN,self).__init__(config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)


    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        # try separate embeddings
        # compressed_output1 = self.embeddingHead(outputs1[1]) # [batch, len/8, dim]
        # query1 = compressed_output1[:, :]

        compressed_output1 = self.embeddingHead(outputs1[0]) # [batch, len/8, dim]
        query1 = self.norm(compressed_output1[:, 0, :])
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class RobertaDot_CLF_ANN_NLL(RobertaDot_CLF_ANN):
    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)

class RobertaDot_CLF_ANN_NLL_IB(RobertaDot_CLF_ANN):
    
    def gen_embed_labels(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # take q-pos-neg and convert to q-psg-label format for bce/nce/nll
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        labels = torch.tensor([1]*a_embs.shape[0]+[0]*b_embs.shape[0]).to(q_embs.device)
        q_embs = torch.cat([q_embs, q_embs], dim=0)
        a_embs = torch.cat([a_embs, b_embs], dim=0)
        return q_embs, a_embs, labels
    
    def nll_loss(self, q_embs, a_embs, labels=None):
        if labels is None:
            labels = torch.ones(a_embs.shape[0]).float().to(q_embs.device)
        logit_matrix = torch.matmul(q_embs, a_embs.t()) #[Q, Q]
        pos_indices = torch.where(labels==1.0)[0]
        pos_logit_matrix = logit_matrix[pos_indices]
        lsm = F.log_softmax(pos_logit_matrix, dim=1)
        loss = -1.0*lsm.gather(1, pos_indices.view(-1,1)).squeeze()
        return loss 

    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs, a_embs, labels = self.gen_embed_labels(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        nll_loss = self.nll_loss(q_embs, a_embs, labels)
        return (nll_loss.mean(),)


class RobertaDot_CLF_ANN_NLL_MultiChunk(RobertaDot_CLF_ANN):
    def __init__(self, config):
        super(RobertaDot_CLF_ANN_NLL_MultiChunk,self).__init__(config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(outputs_k[0]) # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(batchS, chunk_factor, embeddingS)

        return complex_emb_k # size [batchS, chunk_factor, embeddingS]


    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None, is_query = True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len
        
        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(batchS, chunk_factor, -1)[:, :, 0] #[batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()
        

        a12 = torch.matmul(q_embs.unsqueeze(1), a_embs.transpose(1,2)) # [batch, 1, chunk_factor]
        # logits = (a12[:, 0, :] + inverted_bias).max(dim=-1)
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-1, keepdim=False).values # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(batchS, chunk_factor, -1)[:, :, 0] #[batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()
        

        a12 = torch.matmul(q_embs.unsqueeze(1), b_embs.transpose(1,2)) # [batch, 1, chunk_factor]
        # logits = (a12[:, 0, :] + inverted_bias).max(dim=-1)
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-1, keepdim=False).values # [batch]
        # -------------------------------------

        logit_matrix = torch.cat([logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)


# prototype long doc models ----------------------------------------------
class RobertaEmbSimV3_PureDotDirectional_CLF_Softmax_Chunk(RobertaDot_CLF_ANN):
    def __init__(self, config):
        super(RobertaEmbSimV3_PureDotDirectional_CLF_Softmax_Chunk,self).__init__(config)
        # self.embeddingHead = nn.Linear(config.hidden_size, 200)
        self.score = nn.Linear(config.hidden_size, 1)

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        
        base_len = 128 # chunk the model, 512 => 128 * 4
        chunk_factor = full_length // base_len

        input_seq = input_ids.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)
        attention_mask_seq = attention_mask.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                            attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(outputs_k[0]) # [batch, len, dim]
        emb_scores = self.score(outputs_k[0]) # [batch, len, 1]

        [batch_expand, compressed_len, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(batchS, chunk_factor, compressed_len, embeddingS)
        emb_scores = emb_scores.reshape(batchS, chunk_factor, compressed_len, 1)

        attention_mask = attention_mask.reshape(batchS, chunk_factor, -1)

        complex_emb_k_CLF = complex_emb_k[:,:, 0, :] # size [batchS, chunk_factor, embeddingS]
        emb_scores_CLF = emb_scores[:,:, 0, :] # size [batchS, chunk_factor, embeddingS]
        attention_mask_CLF = attention_mask[:, :, 0] # size [batchS, chunk_factor]
        
        # this design is of HUGE unknown
        # perform softmax average? or perform temperature sampling?
        # a better way is 1 layer of transformers?
        inverted_attention_mask_CLF = attention_mask_CLF.unsqueeze(-1).float() * (-9999)
        sm = nn.Softmax(-1)
        attention_weight = sm(emb_scores_CLF + inverted_attention_mask_CLF) # [batchS, chunk_factor, 1]
        attentioned_pooling_emb = torch.matmul(complex_emb_k_CLF.transpose(1,2), attention_weight).squeeze(-1)  # [batchS, embeddingS]
        
        return attentioned_pooling_emb

# Longformer ==============================================================
from adapt_longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size

class Longformer_CLF_Dot_ANN(BertPreTrainedModel):
    def __init__(self, config):
        super(Longformer_CLF_Dot_ANN,self).__init__(config)

        self.roberta = Longformer(config) 
        self.embeddingHead = nn.Linear(config.hidden_size, 200)
        self.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        attention_mask[:, [0,]] =  2
        # input_ids, attention_mask = pad_to_window_size(
        #             input_ids, attention_mask, self.config.attention_window[0], 1) # pad id = 1, tokenizer.pad_token_id for roberta

        outputs1 = self.roberta(input_ids,
                                    attention_mask=attention_mask)
        # try separate embeddings
        compressed_output1 = self.embeddingHead(outputs1[0]) # [batch, len, dim]

        query1 = compressed_output1[:, 0, :] #CLF
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


    def forward(self, input_ids_a, attention_mask_a, input_ids_b = None, attention_mask_b = None, labels=None, is_query = True):
        #print("forward called!!!")
        
        if input_ids_b is None and is_query:
            return self.query_emb(input_ids_a, attention_mask_a)
        elif input_ids_b is None:
            return self.body_emb(input_ids_a, attention_mask_a)
        
        # query, key, value
        query1 = self.query_emb(input_ids_a, attention_mask_a) # output should be [batch, emb/3]
        key2 = self.body_emb(input_ids_b, attention_mask_b)

        a12 = torch.matmul(query1.unsqueeze(1), key2.unsqueeze(2)) # [batch, 1, 1]

        logits = a12[:, 0, 0]

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.float())
            return (loss, logits)


        return (logits)  # (loss), logits, (hidden_states), (attentions)


class Longformer_CLF_QK_ANN(Longformer_CLF_Dot_ANN):
    def __init__(self, config):
        super(Longformer_CLF_QK_ANN,self).__init__(config)
        self.KeyEmbeddingHead = nn.Linear(config.hidden_size, 200)
        
        self.apply(self._init_weights)
        self.config = config

    def body_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        attention_mask[:, [0,]] =  2
        # input_ids, attention_mask = pad_to_window_size(
        #             input_ids, attention_mask, self.config.attention_window[0], 1) # pad id = 1, tokenizer.pad_token_id for roberta

        outputs1 = self.roberta(input_ids,
                                    attention_mask=attention_mask)
        # try separate embeddings
        compressed_output1 = self.KeyEmbeddingHead(outputs1[0]) # [batch, len, dim]

        query1 = compressed_output1[:, 0, :] #CLF
        return query1

# --------------------------------------------------------------------------

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,

            LongformerConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    # "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    # "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    # "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    # "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "cosine_sim": (RobertaConfig, RobertaEmbSim, RobertaTokenizer),
    # "roberta_pair2": (RobertaConfig, RobertaEmbSimV2, RobertaTokenizer),
    "roberta_pair_lstm": (RobertaConfig, RobertaEmbSimLSTM, RobertaTokenizer),
    "r3": (RobertaConfig, RobertaEmbSimV3, RobertaTokenizer),
    # "r4": (RobertaConfig, RobertaEmbSimV4, RobertaTokenizer),

    # "roberta_pair3d": (RobertaConfig, RobertaEmbSimV3_d, RobertaTokenizer),
    # "r5" : (RobertaConfig, RobertaEmbSimV5, RobertaTokenizer), 
    "r6" : (RobertaConfig, RobertaEmbSimV6, RobertaTokenizer),
    "r3n": (RobertaConfig, RobertaEmbSimV3N, RobertaTokenizer),
    "r3l": (RobertaConfig, RobertaEmbSimV3_Linear, RobertaTokenizer), 
    "r3ld": (RobertaConfig, RobertaEmbSimV3_LinearDirectional, RobertaTokenizer),
    "r3le": (RobertaConfig, RobertaEmbSimV3_LinearEmb, RobertaTokenizer),
    "r3d": (RobertaConfig, RobertaEmbSimV3_Directional, RobertaTokenizer),
    "b3le": (BertConfig, BertEmbSimV3_LinearEmb, BertTokenizer),
    
    "r3db": (RobertaConfig, RobertaEmbSimV3_DirectionalBound, RobertaTokenizer),
    "r3dbe": (RobertaConfig, RobertaEmbSimV3_DirectionalBoundExpo, RobertaTokenizer),

    "r3_nwpd": (RobertaConfig, RobertaEmbSimV3_PureDot_NoWindow, RobertaTokenizer),
    "r3pure_dot": (RobertaConfig, RobertaEmbSimV3_PureDot, RobertaTokenizer),
    "r3pure_qq": (RobertaConfig, RobertaEmbSimV3_QQ, RobertaTokenizer),
    "dot_sim": (RobertaConfig, RobertaEmbDotSim, RobertaTokenizer),   
    "cw": (RobertaConfig, RobertEmbSimV3_WeightedCosine, RobertaTokenizer),   
    "r3pdot_bow": (RobertaConfig, RobertaEmbSimV3_PureDotBOW, RobertaTokenizer),

    "r3pdd" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional, RobertaTokenizer), 
    "r3pdd_bow" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_BOW, RobertaTokenizer), 

    "dot_sim_diff": (RobertaConfig, RobertaEmbDotSim_Diff_Encoder, RobertaTokenizer),
    "dot_sim_diff_full": (RobertaConfig, RobertaEmbDotSim_Diff_FullEncoder, RobertaTokenizer), 
    "cosine_sim_diff" : (RobertaConfig, RobertaEmbSimDiff, RobertaTokenizer),
    "cosine_sim_mean_triple" : (RobertaConfig, RobertaMeanEmbedTripleLoss, RobertaTokenizer),
    "cosine_sim_mean_triple_200" : (RobertaConfig, RobertaMeanEmbedTripleLoss200, RobertaTokenizer),  

    "r3d2" : (RobertaConfig, RobertaEmbSimV3_Directional2, RobertaTokenizer) ,
    "cwd": (RobertaConfig, RobertEmbSimV3_WeightedCosineDirectional, RobertaTokenizer),  
    "r3ds": (RobertaConfig, RobertaEmbSimV3_DirectionalSym, RobertaTokenizer), 

    "r3pdd_max" : (RobertaConfig, RobertaEmbSimV3_PDD_MAX, RobertaTokenizer), 
    "r3pdd_nms_max" : (RobertaConfig, RobertaEmbSimV3_PDD_NMS_MAX, RobertaTokenizer), 
    "r3pdd_nms_sum" : (RobertaConfig, RobertaEmbSimV3_PDD_NMS_SUM, RobertaTokenizer),

    # ------------ Embedding output models -----------------------------
    "r3pdd_emb_qp": (RobertaConfig, EmbeddingGen, RobertaTokenizer),
    "r3pdd_emb_mean": (RobertaConfig, EmbeddingGenMean, RobertaTokenizer),
    "r3pdd_max_emb_docs": (RobertaConfig, EmbeddingGenDocs, RobertaTokenizer),
    
    # ------------ ANN supported models -----------------------------
    "r3pdd_ann" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectionalANN, RobertaTokenizer),
    "r3pdd_ann_max_multichunk": (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_CLF_MAX_Chunk, RobertaTokenizer),

    "triple_ann" : (RobertaConfig, RobertaMeanEmbedTripleLossANN, RobertaTokenizer),

    "r3pdd_ann_max_long" : (RobertaConfig, RobertaEmbSimV3_CLF_ANN, RobertaTokenizer),

    "triple_ann_200" : (RobertaConfig, RobertaMeanEmbedTripleLoss200ANN, RobertaTokenizer),  
    "triple_ann_200_inf" : (RobertaConfig, RobertaMeanEmbedTripleLoss200ANN_Inference, RobertaTokenizer),  
    "sentencebert200ce" : (RobertaConfig, SentenceBert200CE, RobertaTokenizer),

    "r3pdd_triple1_ann" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_Triple, RobertaTokenizer),
    # "r3pdd_triple2_ann" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_Triple2, RobertaTokenizer),
    # "r3pdd_triple2_mean_ann" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_Triple_Mean, RobertaTokenizer),

    "r3pdd_ann_mean" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_Mean, RobertaTokenizer),
    "r3pdd_ann_mean_long" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_Mean_Chunk, RobertaTokenizer),
    "r3pdd_ann_cls_mean_long" : (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_CLF_Mean_Chunk, RobertaTokenizer),

    "r3pdd_ann_clf" : (RobertaConfig, RobertaEmbSimV3_CLF_ANN, RobertaTokenizer),
    

    "rdot": ( RobertaConfig, RobertaDot_CLF_ANN , RobertaTokenizer ),
    "rdot_softmax": (RobertaConfig, RobertaEmbSimV3_PureDotDirectional_CLF_Softmax_Chunk , RobertaTokenizer),


    "rdot_nll": ( RobertaConfig, RobertaDot_CLF_ANN_NLL , RobertaTokenizer ),
    "rdot_nll_multi_chunk": ( RobertaConfig, RobertaDot_CLF_ANN_NLL_MultiChunk , RobertaTokenizer ),

    "rdot_nll_ib": ( RobertaConfig, RobertaDot_CLF_ANN_NLL_IB , RobertaTokenizer ),

    # ---------------------------------- Longformer ----------------------------------------------
    "l_ann_clf" : (LongformerConfig, Longformer_CLF_Dot_ANN, RobertaTokenizer),
    
}