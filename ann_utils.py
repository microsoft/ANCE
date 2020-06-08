import json
import numpy as np
import gzip
import os
from multiprocessing import Process

from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import torch.distributed as dist
import re

from pairwise_model import MODEL_CLASSES, ALL_MODELS


def get_checkpoint_no(checkpoint_path):
    return int(re.findall(r'\d+', checkpoint_path)[-1])


def get_latest_ann_data(ann_data_path):
    if not os.path.exists(ann_data_path):
        return -1, None, None
    files = list(next(os.walk(ann_data_path))[2])
    num_start_pos = len("ann_ndcg_")
    data_no_list = [int(s[num_start_pos:]) for s in files if s[:num_start_pos] == "ann_ndcg_"]
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(ann_data_path, "ann_ndcg_" + str(data_no)), 'r') as f:
            ndcg_json = json.load(f)
        return data_no, os.path.join(ann_data_path, "ann_training_data_" + str(data_no)), ndcg_json
    return -1, None, None


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(meta['embedding_size']) * self.dtype.itemsize + 4
        if seed>=0:
            self.ix_array = np.random.RandomState(seed).permutation(self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError("Index {} is out of bound for cached embeddings of size {}".format(key, self.total_number))
        self.f.seek(key*self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas=-1 
    
    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                yield rec


def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            out_f.write(line_fn(args, line, tokenizer))


def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file, args=(args, i, num_process, in_path, out_path, line_fn,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
