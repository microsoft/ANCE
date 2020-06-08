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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch import nn

# removed unnecessary clutter - import everything from lee
from pairwise_model import MODEL_CLASSES, ALL_MODELS
from iterable_dataset import StreamingDataLoader

from lamb import Lamb
import random 

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
)

from transformers import glue_compute_metrics as compute_metrics
#from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
import copy
import csv
from torch import nn
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


from sklearn.metrics import roc_curve, auc
import pandas as pd

logger = logging.getLogger(__name__)

#note that theres actually a typo in the path - mmr instead mrr
eval_configs = {
                "small":["../../../../glue_data/MSMarco/mrr_dev_small", "qrels.dev.filtered.tsv"],
                "full":["../../../../glue_data/MSMarco/mmr_dev", "qrels.dev.small.tsv"]
                }

def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj



def InferenceEmbeddingFromStreamDataLoader(args, model, train_dataloader, is_query_inference = True, prefix ="", emb_chunk_size = -1):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_task = args.task_name
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    dist.barrier()
    model.eval()

    chunk_id = 0
    with model.no_sync():
        for batch in tqdm(train_dataloader, desc="Inferencing", disable=args.local_rank not in [-1, 0], position=0, leave=True):
            
            idxs = batch[3].detach().numpy() #[#B]
            embedding2id.append(idxs)

            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids_a": batch[0].long(), "attention_mask_a": batch[1].long(), "is_query" : is_query_inference}
                
                outputs = model(**inputs)
                emb = outputs
            
            embs = emb.detach().cpu().numpy() #[#B, emb_dim]        
            embedding.append(embs)

            if emb_chunk_size > 0 and len(embs) * eval_batch_size >= emb_chunk_size:
                embedding = np.concatenate(embedding, axis=0)
                embedding2id = np.concatenate(embedding2id, axis=0)

                pickle_path = os.path.join(args.output_dir, "{1}_embedding_{0}_{2}.pb".format(str(args.rank), prefix, str(chunk_id)))
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(embedding, handle)


                pickle_path = os.path.join(args.output_dir, "{1}_embedding_id_{0}_{2}.pb".format(str(args.rank), prefix, str(chunk_id)))
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(embedding2id, handle)

                chunk_id += 1
                embedding = []
                embedding2id = []


    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


def GetDocProcessingFunct(args, tokenizer):
    def fn(line, i):
        passage_collection = []
        
        line_arr = line.split('\t')
        p_id = line_arr[0]
        p_id = int(p_id[1:]) # remove "D"

        url = line_arr[1].rstrip()
        title = line_arr[2].rstrip()
        p_text = line_arr[3].rstrip()

        full_text = url + "<sep>" + title + "<sep>" + p_text

        full_text = full_text[:args.max_doc_character] # keep only first 10000 characters, should be sufficient for any experiment that uses less than 500 - 1k tokens

        passage = tokenizer.encode(full_text, add_special_tokens=True, max_length= args.max_seq_length ,)
        token_type_ids_b = [1] * len(passage)
        attention_mask_b = [1] * len(passage)

        input_id_b, attention_mask_b, token_type_ids_b = pad_ids(passage, attention_mask_b, token_type_ids_b, args.max_seq_length)
        passage_collection.append((int(p_id), input_id_b, attention_mask_b, token_type_ids_b))
        
        query2id_tensor = torch.tensor([f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor([f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor([f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor([f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor)

        return [ts for ts in dataset]
    
    return fn


def GetQueryProcessingFunct(args, tokenizer):
    def fn(line, i):
        passage_collection = []
        
        line_arr = line.split('\t')
        q_id = int(line_arr[0])

        passage = tokenizer.encode(line_arr[1].rstrip(), add_special_tokens=True, max_length=args.max_query_length)

        token_type_ids_b = [1] * len(passage)
        attention_mask_b = [1] * len(passage)

        input_id_b, attention_mask_b, token_type_ids_b = pad_ids(passage, attention_mask_b, token_type_ids_b, args.max_query_length)
        passage_collection.append((q_id, input_id_b, attention_mask_b, token_type_ids_b))
        
        query2id_tensor = torch.tensor([f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor([f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor([f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor([f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor)

        return [ts for ts in dataset]
    
    return fn

# streaming inference
def StreamInferenceDoc(args, data_path, model, fn, prefix, file, is_query_inference = True):
    f = file
    logger.info("Start stream inference:" + data_path)
    
    inference_batch_size = args.per_gpu_eval_batch_size #* max(1, args.n_gpu)
    inference_dataloader = StreamingDataLoader(f, fn, batch_size=inference_batch_size, num_workers=1)

    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    dist.barrier() # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(args, model, inference_dataloader, is_query_inference = is_query_inference, prefix = prefix)
    
    # preserve to memory
    full_embedding = barrier_array_merge(args, _embedding, prefix = prefix + "_emb_p_", only_load_in_master = True) 
    full_embedding2id = barrier_array_merge(args, _embedding2id, prefix = prefix + "_embid_p_", only_load_in_master = True)

    if is_first_worker():
        # save to disk
        pickle_path = os.path.join(args.output_dir, "{0}_full.pb".format(prefix))
        logger.info("Saving: " + pickle_path)
        pickle_save(full_embedding, pickle_path)

        pickle_path = os.path.join(args.output_dir, "{0}_id_full.pb".format(prefix))
        logger.info("Saving: " + pickle_path)
        pickle_save(full_embedding2id, pickle_path)

    dist.barrier()

    logger.info("Finished stream inference")
    return True



# ANN - active learning ------------------------------------------------------
import faiss
import gzip

# more robust than torch.save
def pickle_save(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=-1)

def pickle_load(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)

    return b

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
            pickle.dump(data_array, handle)

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
            pickle.dump(data_list, handle)

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


def should_skip_rank(idx, args):
    if args.local_rank == -1:
        return False

    world_size = args.world_size
    return (idx % world_size) != args.rank


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


def EvalDevQuery(args, query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    for query_idx in range(I_nearest_neighbor.shape[0]): 
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
        selected_ann_idx = top_ann_pid[:50]
        rank = 0
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]
            rank += 1
            prediction[query_id][pred_pid] = -rank

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    ndcg = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]

    final_ndcg = ndcg / eval_query_cnt
    print("Rank:" + str(args.rank) + " --- ANN NDCG@10:" + str(final_ndcg))

    return final_ndcg, eval_query_cnt

def GenerateNegativePassaageID(args, query_embedding2id, passage_embedding2id, training_query_positive_id, I_nearest_neighbor, effective_q_id):
    query_negative_passage = {}
    all_neg_pid = set()
    all_positive_pid = set()
    SelectTopK = args.ann_measure_topk_mrr
    mrr = 0 # only meaningful if it is SelectTopK = True
    num_queries = 0

    for query_idx in range(I_nearest_neighbor.shape[0]): 

        query_id = query_embedding2id[query_idx]

        if not query_id in effective_q_id:
            continue

        num_queries += 1

        pos_pid = training_query_positive_id[query_id]
        all_positive_pid.add(pos_pid)
        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()

        if SelectTopK:
            selected_ann_idx = top_ann_pid[:args.negative_sample + 1]
        else:
            negative_sample_I_idx = np.random.choice(I_nearest_neighbor.shape[1], args.negative_sample + 1)
            selected_ann_idx = top_ann_pid[negative_sample_I_idx] #[negative_sample_ratio + 1]

        query_negative_passage[query_id] = []

        neg_cnt = 0
        rank = 0
        
        for idx in selected_ann_idx:
            neg_pid = passage_embedding2id[idx]
            rank += 1
            if neg_pid == pos_pid:
                if rank <= 10:
                    mrr += 1 / rank
                continue

            if neg_cnt >= args.negative_sample:
                continue
            
            all_neg_pid.add(neg_pid)
            query_negative_passage[query_id].append(neg_pid)
            neg_cnt+=1

    if SelectTopK:
        print("Rank:" + str(args.rank) + " --- ANN MRR:" + str(mrr / num_queries))

    return query_negative_passage, all_neg_pid, all_positive_pid

def LoadAdditionalPassages(args, all_pid, pid2qidx):
    #load data

    task = args.task_name
    # loading cache

    model_name = args.model_name_or_path if args.cache_model_name is None else args.cache_model_name

    cached_file_pattern = "cached_{}_{}_{}_{}".format("ann_doc_collection",
            list(filter(None, model_name.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        )

    cached_features_file = os.path.join(
        args.data_dir,
        cached_file_pattern ,
    )

    all_files_in_data_dir = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

    for filename in all_files_in_data_dir :
        if cached_file_pattern in filename:
            cache_file = torch.load(cached_features_file)
            passage_collection = cache_file["passage"]

    pid2index = {}
    new_passage_collection = []

    cnt = 0
    for (pid, p_input_id) in passage_collection:

        if pid in pid2qidx:
            continue

        if not (pid in all_pid):
            continue
        
        passage = p_input_id
        token_type_ids_b = [1] * len(passage)
        attention_mask_b = [1] * len(passage)
        input_id_b, attention_mask_b, token_type_ids_b = pad_ids(passage, attention_mask_b, token_type_ids_b, args.max_seq_length)


        new_passage_collection.append((pid, input_id_b, attention_mask_b, token_type_ids_b))
        pid2index[pid] = cnt
        cnt += 1

    logger.info("***** Loaded " + str(cnt) + " Additional Negative Passage *****")

    return new_passage_collection, pid2index

# ----------------------------------------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )
    parser.add_argument(
        "--eval_type",
        default="full",
        type=str,
        help="MSMarco eval type - dev full or small",
    )

    parser.add_argument(
        "--optimizer",
        default="lamb",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--logging_steps_per_eval", type=int, default=10, help="Eval every X logging steps.")
    
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    # parser.add_argument(
    #     "--expected_train_size",
    #     default= 100000,
    #     type=int,
    #     help="Expected train dataset size",
    # )

    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--cache_model_name",
        default= None,
        type=str,
        help="cacehed model name",
    )

    parser.add_argument(
        "--negative_sample",
        default= 5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )
    parser.add_argument(
        "--topk_training",
        default= 500,
        type=int,
        help="top k from which negative samples are collected",
    )
    
    parser.add_argument(
        "--max_record_ann",
        default= -1,
        type=int,
        help="for debugging, only generate ANN search index up to N",
    )

    parser.add_argument(
        "--ann_chunk_factor",
        default= 5, # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--passage_skip_factor",
        default= -1, 
        type=int,
        help="used to skip data for quick prototype and test",
    )

    parser.add_argument(
        "--load_optimizer_scheduler",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--load_doc_emb",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--starting_chunk",
        default= 0, 
        type=int,
        help="",
    )

    # ----------------- End of ANN HyperParam ------------------
    # ----------------- Doc Ranking HyperParam ------------------
    parser.add_argument(
        "--max_doc_character",
        default= 20000, 
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )


    parser.add_argument(
        "--additional_partition_factor",
        default= 1, 
        type=int,
        help="further chunk the passages",
    )

    parser.add_argument(
        "--parallel_loading_process",
        default= 8, 
        type=int,
        help="further chunk the passages",
    )

    parser.add_argument(
        "--emb_chunk_size",
        default= -1, 
        type=int,
        help="number of examples to process before dumping intermediate data",
    )
    
    

    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    #args.output_mode = "classification"
    label_list = processor.get_labels()
    #label_list = ["0", "1"]
    num_labels = len(label_list)


    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    

    query_collection_path = os.path.join(
        args.data_dir,
        "msmarco-docdev-queries.tsv.gz",
    )
    logger.info("***** inference of dev query *****")
    with gzip.open(query_collection_path, 'rt', encoding='utf8') as f:
        StreamInferenceDoc(args, query_collection_path, model, GetQueryProcessingFunct(args, tokenizer), "dev_query_", f, is_query_inference = True)


    query_collection_path = os.path.join(
        args.data_dir,
        "msmarco-test2019-queries.tsv.gz",
    )
    logger.info("***** inference of test query *****")
    with gzip.open(query_collection_path, 'rt', encoding='utf8') as f:
        StreamInferenceDoc(args, query_collection_path, model, GetQueryProcessingFunct(args, tokenizer), "test_query_", f, is_query_inference = True)


    query_collection_path = os.path.join(
        args.data_dir,
        "msmarco-doctrain-queries.tsv.gz" ,
    )

    logger.info("***** inference of query *****")
    with gzip.open(query_collection_path, 'rt', encoding='utf8') as f:
        StreamInferenceDoc(args, query_collection_path, model, GetQueryProcessingFunct(args, tokenizer), "query_", f, is_query_inference = True)

    passage_collection_path = os.path.join(
        args.data_dir,
        "msmarco-docs.tsv" ,
    )
    logger.info("***** inference of document *****")
    with open(passage_collection_path, "r", encoding="utf-8") as f:
        StreamInferenceDoc(args, passage_collection_path, model, GetDocProcessingFunct(args, tokenizer), "document_", f, is_query_inference = False)

    return



if __name__ == "__main__":
    main()