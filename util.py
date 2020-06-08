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
import os, shutil
import random
import pytrec_eval

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm, trange
import torch.distributed as dist
from torch import nn

import random 
import copy
import csv
from torch import nn
import pickle

from sklearn.metrics import roc_curve, auc
import pandas as pd

logger = logging.getLogger(__name__)

#note that theres actually a typo in the path - mmr instead mrr
class InputFeaturesPair(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids_a, attention_mask_a=None, token_type_ids_a=None, 
                    input_ids_b = None, attention_mask_b=None, token_type_ids_b=None,
                    label=None):

        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a

        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj

import faiss
import gzip

def PadListTillSameLen(args, list_to_pad, prefix = ""):
    len_list = len(list_to_pad)
    all_len_list = barrier_list_merge(args, [len_list], prefix = prefix)

    max_train_samples = max(all_len_list)
    
    if len_list < max_train_samples:
        num_pad = max_train_samples - len_list

        for _ in range(num_pad):
            list_to_pad.append(list_to_pad[-1])

    return list_to_pad

def TrimListTillSameLen(args, list_to_pad, prefix = ""):
    len_list = len(list_to_pad)
    all_len_list = barrier_list_merge(args, [len_list], prefix = prefix)

    min_train_samples = min(all_len_list)
    return list_to_pad[:min_train_samples]


def barrier_array_merge(args, data_array, merge_axis = 0, prefix = "", load_cache = False, only_load_in_master = False):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    import pickle
    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        dist.barrier() # directory created
        pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(rank), prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process collects it
        dist.barrier()

    data_array = None

    data_list = []

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(args.world_size): # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list.append(b)
        except:
            continue
            
    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg


def barrier_list_merge(args, data_list, prefix = "", load_cache = False, only_load_in_master = False):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_list

    import pickle

    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                try:
                    os.makedirs(args.output_dir)
                except:
                    print("directory already made")

        dist.barrier() # directory created
        pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(rank), prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_list, handle, protocol = 4)

        # make sure all processes wrote their data before first process collects it
    
    data_list = []
    dist.barrier() # all files written
    
    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return data_list

    for i in range(args.world_size): # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list += b
        except:
            continue
            
    dist.barrier()
    return data_list


# much more robust than torch.save
def pickle_save(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)


def pickle_load(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)

    return b


def should_skip_rank(idx, args):
    if args.local_rank == -1:
        return False

    world_size = args.world_size
    return (idx % world_size) != args.rank


def pad_input_ids(input_ids, max_length, 
            pad_on_left=False,
            pad_token=0,):
    padding_length = max_length - len(input_ids)

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)

    return input_ids


def pad_ids(input_ids, attention_mask, token_type_ids, max_length, 
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True,):
    padding_length = max_length - len(input_ids)

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    return input_ids, attention_mask, token_type_ids


# to reuse pytrec_eval, id must be string
def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return roc_t.iloc[0]['threshold']

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def concat_key(all_list, key, axis=0):
    return np.concatenate([ele[key] for ele in all_list], axis=axis)

def gen_offset_map(f):
    c=0
    offset_map = []   
    offset = f.tell()
    line = f.readline()
    c+=1
    if line:
        offset_map.append(offset)
    while line:
        offset = f.tell()
        line = f.readline()
        if line:
            offset_map.append(offset)
        c+=1
        if c%10000000==0:
            print(c)
    return offset_map

def line_shuffle_generator(f, off_map, seed, encoding='utf-8'):
    if seed<0:
        f.seek(0)
        for line in f:
            yield line.decode(encoding)
    else:
        ix_array = np.random.RandomState(seed).permutation(len(off_map))
        for ix in ix_array:
            f.seek(off_map[ix])
            line = f.readline()
            yield line.decode(encoding)
