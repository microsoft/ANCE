import sys
sys.path += ['../']
import torch
import os
import faiss
from utils.util import (
    barrier_array_merge,
    convert_to_string_id,
    is_first_worker,
    StreamingDataset,
    EmbeddingCache,
    get_checkpoint_no,
    get_latest_ann_data
)
import csv
import copy
import transformers
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
from data.msmarco_data import GetProcessingFn  
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from os.path import isfile, join
import argparse
import json
import logging
import random
import time
import pytrec_eval

torch.multiprocessing.set_sharing_strategy('file_system')


logger = logging.getLogger(__name__)


# ANN - active learning ------------------------------------------------------

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir):
        return args.init_model_dir, 0
    subdirectories = list(next(os.walk(args.training_dir))[1])

    def valid_checkpoint(checkpoint):
        chk_path = os.path.join(args.training_dir, checkpoint)
        scheduler_path = os.path.join(chk_path, "scheduler.pt")
        return os.path.exists(scheduler_path)

    checkpoint_nums = [get_checkpoint_no(
        s) for s in subdirectories if valid_checkpoint(s)]

    if len(checkpoint_nums) > 0:
        return os.path.join(args.training_dir, "checkpoint-" +
                            str(max(checkpoint_nums))) + "/", max(checkpoint_nums)
    return args.init_model_dir, 0


def load_positive_ids(args):

    logger.info("Loading query_2_pos_docid")
    training_query_positive_id = {}
    query_positive_id_path = os.path.join(args.data_dir, "train-qrel.tsv")
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            assert rel == "1"
            topicid = int(topicid)
            docid = int(docid)
            training_query_positive_id[topicid] = docid

    logger.info("Loading dev query_2_pos_docid")
    dev_query_positive_id = {}
    query_positive_id_path = os.path.join(args.data_dir, "dev-qrel.tsv")

    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            topicid = int(topicid)
            docid = int(docid)
            if topicid not in dev_query_positive_id:
                dev_query_positive_id[topicid] = {}
            dev_query_positive_id[topicid][docid] = int(rel)

    return training_query_positive_id, dev_query_positive_id


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path
    config = configObj.config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="MSMarco",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = configObj.model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    return config, tokenizer, model


def InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        train_dataloader,
        is_query_inference=True,
        prefix=""):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    for batch in tqdm(train_dataloader,
                      desc="Inferencing",
                      disable=args.local_rank not in [-1,
                                                      0],
                      position=0,
                      leave=True):

        idxs = batch[3].detach().numpy()  # [#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs)
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args, model, fn, prefix, f, is_query_inference=True):
    inference_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier()  # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(
        args, model, inference_dataloader, is_query_inference=is_query_inference, prefix=prefix)

    logger.info("merging embeddings")

    # preserve to memory
    full_embedding = barrier_array_merge(
        args,
        _embedding,
        prefix=prefix +
        "_emb_p_",
        load_cache=False,
        only_load_in_master=True)
    full_embedding2id = barrier_array_merge(
        args,
        _embedding2id,
        prefix=prefix +
        "_embid_p_",
        load_cache=False,
        only_load_in_master=True)

    return full_embedding, full_embedding2id


def generate_new_ann(
        args,
        output_num,
        checkpoint_path,
        training_query_positive_id,
        dev_query_positive_id,
        latest_step_num):
    config, tokenizer, model = load_model(args, checkpoint_path)

    logger.info("***** inference of dev query *****")
    dev_query_collection_path = os.path.join(args.data_dir, "dev-query")
    dev_query_cache = EmbeddingCache(dev_query_collection_path)
    with dev_query_cache as emb:
        dev_query_embedding, dev_query_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(
            args, query=True), "dev_query_" + str(latest_step_num) + "_", emb, is_query_inference=True)

    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        passage_embedding, passage_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(
            args, query=False), "passage_" + str(latest_step_num) + "_", emb, is_query_inference=False)
    logger.info("***** Done passage inference *****")

    if args.inference:
        return

    logger.info("***** inference of train query *****")
    train_query_collection_path = os.path.join(args.data_dir, "train-query")
    train_query_cache = EmbeddingCache(train_query_collection_path)
    with train_query_cache as emb:
        query_embedding, query_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(
            args, query=True), "query_" + str(latest_step_num) + "_", emb, is_query_inference=True)

    if is_first_worker():
        dim = passage_embedding.shape[1]
        print('passage embedding shape: ' + str(passage_embedding.shape))
        top_k = args.topk_training
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(passage_embedding)
        logger.info("***** Done ANN Index *****")

        # measure ANN mrr
        # I: [number of queries, topk]
        _, dev_I = cpu_index.search(dev_query_embedding, 100)
        dev_ndcg, num_queries_dev = EvalDevQuery(
            args, dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I)

        # Construct new traing set ==================================
        chunk_factor = args.ann_chunk_factor
        effective_idx = output_num % chunk_factor

        if chunk_factor <= 0:
            chunk_factor = 1
        num_queries = len(query_embedding)
        queries_per_chunk = num_queries // chunk_factor
        q_start_idx = queries_per_chunk * effective_idx
        q_end_idx = num_queries if (
            effective_idx == (
                chunk_factor -
                1)) else (
            q_start_idx +
            queries_per_chunk)
        query_embedding = query_embedding[q_start_idx:q_end_idx]
        query_embedding2id = query_embedding2id[q_start_idx:q_end_idx]

        logger.info(
            "Chunked {} query from {}".format(
                len(query_embedding),
                num_queries))
        # I: [number of queries, topk]
        _, I = cpu_index.search(query_embedding, top_k)

        effective_q_id = set(query_embedding2id.flatten())
        query_negative_passage = GenerateNegativePassaageID(
            args,
            query_embedding2id,
            passage_embedding2id,
            training_query_positive_id,
            I,
            effective_q_id)

        logger.info("***** Construct ANN Triplet *****")
        train_data_output_path = os.path.join(
            args.output_dir, "ann_training_data_" + str(output_num))

        with open(train_data_output_path, 'w') as f:
            query_range = list(range(I.shape[0]))
            random.shuffle(query_range)
            for query_idx in query_range:
                query_id = query_embedding2id[query_idx]
                if query_id not in effective_q_id or query_id not in training_query_positive_id:
                    continue
                pos_pid = training_query_positive_id[query_id]
                f.write(
                    "{}\t{}\t{}\n".format(
                        query_id, pos_pid, ','.join(
                            str(neg_pid) for neg_pid in query_negative_passage[query_id])))

        ndcg_output_path = os.path.join(
            args.output_dir, "ann_ndcg_" + str(output_num))
        with open(ndcg_output_path, 'w') as f:
            json.dump({'ndcg': dev_ndcg, 'checkpoint': checkpoint_path}, f)

        return dev_ndcg, num_queries_dev


def GenerateNegativePassaageID(
        args,
        query_embedding2id,
        passage_embedding2id,
        training_query_positive_id,
        I_nearest_neighbor,
        effective_q_id):
    query_negative_passage = {}
    SelectTopK = args.ann_measure_topk_mrr
    mrr = 0  # only meaningful if it is SelectTopK = True
    num_queries = 0

    for query_idx in range(I_nearest_neighbor.shape[0]):

        query_id = query_embedding2id[query_idx]

        if query_id not in effective_q_id:
            continue

        num_queries += 1

        pos_pid = training_query_positive_id[query_id]
        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()

        if SelectTopK:
            selected_ann_idx = top_ann_pid[:args.negative_sample + 1]
        else:
            negative_sample_I_idx = list(range(I_nearest_neighbor.shape[1]))
            random.shuffle(negative_sample_I_idx)
            selected_ann_idx = top_ann_pid[negative_sample_I_idx]

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

            if neg_pid in query_negative_passage[query_id]:
                continue

            if neg_cnt >= args.negative_sample:
                break

            query_negative_passage[query_id].append(neg_pid)
            neg_cnt += 1

    if SelectTopK:
        print("Rank:" + str(args.rank) +
              " --- ANN MRR:" + str(mrr / num_queries))

    return query_negative_passage


def EvalDevQuery(
        args,
        query_embedding2id,
        passage_embedding2id,
        dev_query_positive_id,
        I_nearest_neighbor):
    # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
    prediction = {}

    for query_idx in range(I_nearest_neighbor.shape[0]):
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
        selected_ann_idx = top_ann_pid[:50]
        rank = 0
        seen_pid = set()
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]

            if pred_pid not in seen_pid:
                # this check handles multiple vector per document
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

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


def get_arguments():
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
        "--training_dir",
        default=None,
        type=str,
        required=True,
        help="Training dir, will look for latest checkpoint dir in here",
    )

    parser.add_argument(
        "--init_model_dir",
        default=None,
        type=str,
        required=True,
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )

    parser.add_argument(
        "--last_checkpoint_dir",
        default="",
        type=str,
        help="Last checkpoint used, this is for rerunning this script when some ann data is already generated",
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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where cached data will be written",
    )

    parser.add_argument(
        "--end_output_num",
        default=-
        1,
        type=int,
        help="Stop after this number of data versions has been generated, default run forever",
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
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--ann_chunk_factor",
        default=5,  # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--topk_training",
        default=500,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--negative_sample",
        default=5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--only_keep_latest_embedding_file",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
    )
    
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank",
    )
    
    parser.add_argument(
        "--server_ip",
        type=str,
        default="",
        help="For distant debugging.",
    )

    parser.add_argument(
        "--server_port",
        type=str,
        default="",
        help="For distant debugging.",
    )
    
    parser.add_argument(
        "--inference",
        default=False,
        action="store_true",
        help="only do inference if specify",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    args = parser.parse_args()

    return args


def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )


def ann_data_gen(args):
    last_checkpoint = args.last_checkpoint_dir
    ann_no, ann_path, ndcg_json = get_latest_ann_data(args.output_dir)
    output_num = ann_no + 1

    logger.info("starting output number %d", output_num)

    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

    training_positive_id, dev_positive_id = load_positive_ids(args)

    while args.end_output_num == -1 or output_num <= args.end_output_num:
        next_checkpoint, latest_step_num = get_latest_checkpoint(args)

        if args.only_keep_latest_embedding_file:
            latest_step_num = 0

        if next_checkpoint == last_checkpoint:
            time.sleep(60)
        else:
            logger.info("start generate ann data number %d", output_num)
            logger.info("next checkpoint at " + next_checkpoint)
            generate_new_ann(
                args,
                output_num,
                next_checkpoint,
                training_positive_id,
                dev_positive_id,
                latest_step_num)
            if args.inference:
                break
            logger.info("finished generating ann data number %d", output_num)
            output_num += 1
            last_checkpoint = next_checkpoint
        if args.local_rank != -1:
            dist.barrier()


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    main()
