import sys
import os
import torch
sys.path += ['../']
import gzip
import pickle
from utils.util import pad_input_ids, multi_file_process, numbered_byte_file_generator, EmbeddingCache
import csv
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset, get_worker_info
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import json


def write_query_rel(args, pid2offset, query_file, positive_id_file, out_query_file, out_id_file):

    print(
        "Writing query files " +
        str(out_query_file) +
        " and " +
        str(out_id_file))
    query_positive_id = set()

    query_positive_id_path = os.path.join(
        args.data_dir,
        positive_id_file,
    )

    print("Loading query_2_pos_docid")
    with gzip.open(query_positive_id_path, 'rt', encoding='utf8') if positive_id_file[-2:] == "gz" else open(query_positive_id_path, 'r', encoding='utf8') as f:
        if args.data_type == 0:
            tsvreader = csv.reader(f, delimiter=" ")
        else:
            tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            query_positive_id.add(int(topicid))

    query_collection_path = os.path.join(
        args.data_dir,
        query_file,
    )

    out_query_path = os.path.join(
        args.out_data_dir,
        out_query_file,
    )

    qid2offset = {}

    print('start query file split processing')
    multi_file_process(
        args,
        32,
        query_collection_path,
        out_query_path,
        QueryPreprocessingFn)

    print('start merging splits')

    idx = 0
    with open(out_query_path, 'wb') as f:
        for record in numbered_byte_file_generator(
                out_query_path, 32, 8 + 4 + args.max_query_length * 4):
            q_id = int.from_bytes(record[:8], 'big')
            if q_id not in query_positive_id:
                # exclude the query as it is not in label set
                continue
            f.write(record[8:])
            qid2offset[q_id] = idx
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(q_id))

    qid2offset_path = os.path.join(
        args.out_data_dir,
        "qid2offset.pickle",
    )
    with open(qid2offset_path, 'wb') as handle:
        pickle.dump(qid2offset, handle, protocol=4)
    print("done saving qid2offset")

    print("Total lines written: " + str(idx))
    meta = {'type': 'int32', 'total_number': idx,
            'embedding_size': args.max_query_length}
    with open(out_query_path + "_meta", 'w') as f:
        json.dump(meta, f)

    embedding_cache = EmbeddingCache(out_query_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[0])

    out_id_path = os.path.join(
        args.out_data_dir,
        out_id_file,
    )

    print("Writing qrels")
    with gzip.open(query_positive_id_path, 'rt', encoding='utf8') if positive_id_file[-2:] == "gz" else open(query_positive_id_path, 'r', encoding='utf8') as f, \
            open(out_id_path, "w", encoding='utf-8') as out_id:

        if args.data_type == 0:
            tsvreader = csv.reader(f, delimiter=" ")
        else:
            tsvreader = csv.reader(f, delimiter="\t")
        out_line_count = 0
        for [topicid, _, docid, rel] in tsvreader:
            topicid = int(topicid)
            if args.data_type == 0:
                docid = int(docid[1:])
            else:
                docid = int(docid)
            out_id.write(str(qid2offset[topicid]) +
                         "\t" +
                         str(pid2offset[docid]) +
                         "\t" +
                         rel +
                         "\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))


def preprocess(args):

    pid2offset = {}
    if args.data_type == 0:
        in_passage_path = os.path.join(
            args.data_dir,
            "msmarco-docs.tsv",
        )
    else:
        in_passage_path = os.path.join(
            args.data_dir,
            "collection.tsv",
        )

    out_passage_path = os.path.join(
        args.out_data_dir,
        "passages",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    out_line_count = 0

    print('start passage file split processing')
    multi_file_process(
        args,
        32,
        in_passage_path,
        out_passage_path,
        PassagePreprocessingFn)

    print('start merging splits')
    with open(out_passage_path, 'wb') as f:
        for idx, record in enumerate(numbered_byte_file_generator(
                out_passage_path, 32, 8 + 4 + args.max_seq_length * 4)):
            p_id = int.from_bytes(record[:8], 'big')
            f.write(record[8:])
            pid2offset[p_id] = idx
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1

    print("Total lines written: " + str(out_line_count))
    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length}
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)
    embedding_cache = EmbeddingCache(out_passage_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[0])

    pid2offset_path = os.path.join(
        args.out_data_dir,
        "pid2offset.pickle",
    )
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    print("done saving pid2offset")

    if args.data_type == 0:
        write_query_rel(
            args,
            pid2offset,
            "msmarco-doctrain-queries.tsv",
            "msmarco-doctrain-qrels.tsv",
            "train-query",
            "train-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "msmarco-test2019-queries.tsv",
            "2019qrels-docs.txt",
            "dev-query",
            "dev-qrel.tsv")
    else:
        write_query_rel(
            args,
            pid2offset,
            "queries.train.tsv",
            "qrels.train.tsv",
            "train-query",
            "train-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "queries.dev.small.tsv",
            "qrels.dev.small.tsv",
            "dev-query",
            "dev-qrel.tsv")


def PassagePreprocessingFn(args, line, tokenizer):
    if args.data_type == 0:
        line_arr = line.split('\t')
        p_id = int(line_arr[0][1:])  # remove "D"

        url = line_arr[1].rstrip()
        title = line_arr[2].rstrip()
        p_text = line_arr[3].rstrip()

        #full_text = url + "<sep>" + title + "<sep>" + p_text
        full_text = url + " "+tokenizer.sep_token+" " + title + " "+tokenizer.sep_token+" " + p_text
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = full_text[:args.max_doc_character]
    else:
        line = line.strip()
        line_arr = line.split('\t')
        p_id = int(line_arr[0])

        p_text = line_arr[1].rstrip()

        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = p_text[:args.max_doc_character]

    passage = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=args.max_seq_length,
    )
    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length,pad_token=tokenizer.pad_token_id)

    
    

    return p_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        max_length=args.max_query_length)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length,pad_token=tokenizer.pad_token_id)

    return q_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_query_length if query else args.max_seq_length

        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len

        passage_collection = [(i, passage, attention_mask, token_type_ids)]

        query2id_tensor = torch.tensor(
            [f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor(
            [f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor(
            [f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor(
            [f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(
            all_input_ids_a,
            all_attention_mask_a,
            all_token_type_ids_a,
            query2id_tensor)

        return [ts for ts in dataset]

    return fn


def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
                   neg_data[0], neg_data[1], neg_data[2])

    return fn


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir",
    )
    parser.add_argument(
        "--out_data_dir",
        default=None,
        type=str,
        required=True,
        help="The output data dir",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(
            MSMarcoConfigDict.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
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
    parser.add_argument(
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )
    parser.add_argument(
        "--data_type",
        default=0,
        type=int,
        help="0 for doc, 1 for passage",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()

    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    preprocess(args)


if __name__ == '__main__':
    main()
