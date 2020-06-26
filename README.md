# Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval

## Requirements

To install requirements, run the following commands:

```setup
pip install transformers==2.3.0
pip install pytrec-eval
pip install faiss-cpu --no-cache
```

## Data Download
Relevant datasets for passage and documents are listed in the tables below with download links.

### Document ranking dataset

| Type   | Filename                                                                                                              | File size |              Num Records | Format                                                         |
|--------|-----------------------------------------------------------------------------------------------------------------------|----------:|-------------------------:|----------------------------------------------------------------|
| Corpus | [msmarco-docs.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz)                          |     22 GB |               3,213,835  | tsv: docid, url, title, body                                   |
| Corpus | [msmarco-docs.trec](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz)                        |     22 GB |               3,213,835  | TREC DOC format (same content as msmarco-docs.tsv)                                               |
| Corpus | [msmarco-docs-lookup.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz)            |    101 MB |               3,213,835  | tsv: docid, offset_trec, offset_tsv                            |
| Train  | [msmarco-doctrain-queries.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz)  |     15 MB |                 367,013  | tsv: qid, query                                                |
| Train  | [msmarco-doctrain-top100](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz)            |    1.8 GB |              36,701,116  | TREC submission: qid, "Q0", docid, rank, score, runstring      |
| Train  | [msmarco-doctrain-qrels.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz)      |    7.6 MB |                 384,597  | TREC qrels format                                              |
| Train  | [msmarco-doctriples.py](https://github.com/microsoft/TREC-2019-Deep-Learning/blob/master/utils/msmarco-doctriples.py) |         - |                       -  | Python script generates training triples |
| Dev    | [msmarco-docdev-queries.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz)      |    216 KB |                   5,193  | tsv: qid, query                                                |
| Dev    | [msmarco-docdev-top100](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz)        |       27 MB |                     519,300  | TREC submission: qid, "Q0", docid, rank, score, runstring      |
| Dev    | [msmarco-docdev-qrels.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz)          |    112 KB |                   5,478  | TREC qrels format                                              |
| Test    | [msmarco-test2019-queries.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz)          |     12K |                   200  | tsv: qid, query                                              |
| Test    | [msmarco-doctest2019-top100](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz)          |   1.1M |                  20,000  | TREC submission: qid, "Q0", docid, rank, score, runstring       |
| Test    | [2019qrels-docs](https://trec.nist.gov/data/deep/2019qrels-docs.txt)          |   331K |                  16,258  | qid, "Q0", docid, rating       |

### Passage ranking dataset

| Description                                           | Filename                                                                                                                | File size |                        Num Records | Format                                                         |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------:|-----------------------------------:|----------------------------------------------------------------|
| Collection                                | [collection.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz)                             |    2.9 GB |                         8,841,823  | tsv: pid, passage |
| Queries                                   | [queries.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz)                                   |   42.0 MB |                         1,010,916  | tsv: qid, query |
| Qrels Dev                                 | [qrels.dev.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv)                                     |    1.1 MB |                            59,273  | TREC qrels format |
| Qrels Train                               | [qrels.train.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv)                                 |   10.1 MB |                           532,761  | TREC qrels format |
| Queries, Passages, and Relevance   Labels | [collectionandqueries.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz)         |    2.9 GB |                        10,406,754  | |
| Train Triples Small                       | [triples.train.small.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz)           |   27.1 GB |                        39,780,811  | tsv: query, positive passage, negative passage |
| Train Triples Large                      | [triples.train.full.tsv.gz](https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.full.tsv.gz)             |  272.2 GB |                       397,756,691  | tsv: query, positive passage, negative passage |
| Train Triples QID PID Format               | [qidpidtriples.train.full.2.tsv.gz](https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz) |    5.7 GB |                       397,768,673  | tsv: qid, positive pid, negative pid |
| Top 1000 Train                            | [top1000.train.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/top1000.train.tar.gz)                       |  175.0 GB |                       478,002,393  | tsv: qid, pid, query, passage |
| Top 1000 Dev                              | [top1000.dev.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz)                           |    2.5 GB |                         6,668,967  | tsv: qid, pid, query, passage |
| Test    | [msmarco-test2019-queries.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz)          |     12K |                   200  | tsv: qid, query                                              |
| Test    | [msmarco-passagetest2019-top1000.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz)          |     71M |                  189,877  | tsv: qid, pid, query, passage                                              |
| Test    | [2019qrels-pass.txt](https://trec.nist.gov/data/deep/2019qrels-pass.txt)          |     182K |                  9,260  | qid, "Q0", docid, rating   

For passage, we used the official small dev set on MSMarco. The corresponding files in one zipped file can be downloaded [here](https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz). 

For detailed information on the datasets, refer to [TREC-2019-Deep-Learning](https://github.com/microsoft/TREC-2019-Deep-Learning) for document datasets and [MS MARCO dataset](https://github.com/microsoft/MSMARCO-Passage-Ranking) for passage datasets.

## Data Preprocessing
The command to preprocess passage and document data is listed below:

```
python data/msmarco_data.py 
--data_dir $raw_data_dir \
--out_data_dir $preprocessed_data_dir \ 
--model_type {use rdot_nll for ANCE FirstP, rdot_nll_multi_chunk for ANCE MaxP} \ 
--model_name_or_path roberta-base \ 
--max_seq_length {use 512 for ANCE FirstP, 2048 for ANCE MaxP} \ 
--data_type {use 1 for passage, 0 for document}
```

The data preprocessing command is included as the first step in the training command file commands/run_train.sh

## Training

To train the model(s) in the paper, you need to start two commands in the following order:

1. run commands/run_train.sh which does three things in a sequence:

	a. Data preprocessing: this is explained in the previous data preprocessing section. This step will check if the preprocess data folder exists, and will be skipped if the checking is positive.

	b. Initial ANN data generation: this step will use the pretrained BM25 warmup checkpoint to generate the initial training data. The command is as follow:

        python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py 
        --training_dir {# checkpoint location, not used for initial data generation} \ 
        --init_model_dir {pretrained BM25 warmup checkpoint location} \ 
        --model_type rdot_nll \
        --output_dir $model_ann_data_dir \
        --cache_dir $model_ann_data_dir_cache \
        --data_dir $preprocessed_data_dir \
        --max_seq_length 512 \
        --per_gpu_eval_batch_size 16 \
        --topk_training {top k candidates for ANN search(ie:200)} \ 
        --negative_sample {negative samples per query(20)} \ 
        --end_output_num 0 # only set as 0 for initial data generation, do not set this otherwise

	c. Training: ANCE training with the most recently generated ANN data, the command is as follow:

        python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann.py 
        --model_type rdot_nll \
        --model_name_or_path $pretrained_checkpoint_dir \
        --task_name MSMarco \
        --triplet {# default = False, action="store_true", help="Whether to run training}\ 
        --data_dir $preprocessed_data_dir \
        --ann_dir {location of the ANN generated training data} \ 
        --max_seq_length 512 \
        --per_gpu_train_batch_size=8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-6 \
        --output_dir $model_dir \
        --warmup_steps 5000 \
        --logging_steps 100 \
        --save_steps 10000 \
        --optimizer lamb 
		
2. Once training starts, start another job in parallel to fetch the latest checkpoint from the ongoing training and update the training data. To do that, run

        bash commands/run_ann_data_gen.sh

    The command is similar to the initial ANN data generation command explained previously

## Warmup for Training
ANCE training starts from a pretrained BM25 warmup checkpoint. The command with our used parameters to train this warmup checkpoint is in commands/run_train_warmup.py and is shown below:

        python3 -m torch.distributed.launch --nproc_per_node=1 ../drivers/run_warmup.py \
        --train_model_type rdot_nll \
        --model_name_or_path roberta-base \
        --task_name MSMarco \
        --do_train \
        --evaluate_during_training \
        --data_dir ${location of your raw data}  
        --max_seq_length 128 
        --per_gpu_eval_batch_size=256 \
        --per_gpu_train_batch_size=32 \
        --learning_rate 2e-4  \
        --logging_steps 100   \
        --num_train_epochs 2.0  \
        --output_dir ${location for checkpoint saving} \
        --warmup_steps 1000  \
        --overwrite_output_dir \
        --save_steps 30000 \
        --gradient_accumulation_steps 1 \
        --expected_train_size 35000000 \
        --logging_steps_per_eval 1 \
        --fp16 \
        --optimizer lamb \
        --log_dir ~/tensorboard/${DLWS_JOB_ID}/logs/OSpass


## Inference
The command for inferencing query and passage/doc embeddings is the same as that for Initial ANN data generation described above as the first step in ANN data generation is inference. However you need to add --inference to the command to have the program to stop after the initial inference step. commands/run_inference.sh provides a sample command.

## Evaluation

The evaluation is done through "Calculate Metrics.ipynb". This notebook calculates full ranking and reranking metrics used in the paper including NDCG, MRR, hole rate, recall for passage/document, dev/eval set specified by user. In order to run it, you need to define the following parameters at the beginning of the Jupyter notebook.
        
        checkpoint_path = {location for dumpped query and passage/document embeddings which is output_dir from run_ann_data_gen.py}
        checkpoint =  {embedding from which checkpoint(ie: 200000)}
        data_type =  {0 for document, 1 for passage}
        test_set =  {0 for MSMARCO dev_set, 1 for TREC eval_set}
        raw_data_dir = 
        processed_data_dir = 

## Results
The run_train.sh and run_ann_data_gen.sh files contain the commands with the parameters we used for passage ANCE(FirstP), document ANCE(FirstP) and document ANCE(MaxP)
Our model achieves the following performance on MSMARCO dev set and TREC eval set :


|   MSMARCO Dev Passage Retrieval    | MRR@10  | Recall@1k | Steps |
|---------------- | -------------- |-------------- | -------------- |
| ANCE(FirstP)   |     0.330         |      0.959       |      600K       |
| ANCE(MaxP)   |     -         |      -       |      -       |

|   TREC DL Passage NDCG@10    | Rerank  | Retrieval | Steps |
|---------------- | -------------- |-------------- | -------------- |
| ANCE(FirstP)   |     0.677   |     0.648       |      600K      |
| ANCE(MaxP)   |      -         |     -     |      -       |

|   TREC DL Document NDCG@10    | Rerank  | Retrieval | Steps |
|---------------- | -------------- |-------------- | -------------- |
| ANCE(FirstP)   |     0.641       |      0.615       |      210K       |
| ANCE(MaxP)   |      0.671       |      0.628    |      139K       |

|   MSMARCO Dev Passage Retrieval    | MRR@10  |  Steps |
|---------------- | -------------- | -------------- |
| pretrained BM25 warmup checkpoint   |     0.311       |       60K       |



The pretrained BM25 warmup checkpoint used to train our ANCE models could be downloaded [here](https://drive.google.com/open?id=1m5YI_11wV354CR3sGmJl69eV1sdJkXva)

You can download our best trained models here:
[Passage ANCE(FirstP)](https://drive.google.com/open?id=11n4TaqIxFN44h-AqRYEuU8JcSVbSkAQ2)
[Document ANCE(FirstP)](https://drive.google.com/open?id=17FNKsP54JwXRRpAtfU0oWGOkrocNSMez)
[Document 2048 ANCE(MaxP)](https://drive.google.com/open?id=1hINs2-LPI5NUc16VO_1mYYl3U7dWYRIo)

Our result for document ANCE(FirstP) TREC eval set top 100 retrieved document per query could be downloaded [here](https://drive.google.com/open?id=12ww9BAe6jjtwJG7-baqsJTbwXGjNxzUy)
Our result for document ANCE(MaxP) TREC eval set top 100 retrieved document per query could be downloaded [here](https://drive.google.com/open?id=170uwM-rcBpc268QGfP0zwlUwLXfOe9de)

The TREC eval set query embedding and their ids for our passage ANCE(FirstP) experiment could be downloaded [here](https://drive.google.com/open?id=1al6JrtpZPSFL8JEEeFtWPHcSur522-ty) 
The TREC eval set query embedding and their ids for our document ANCE(FirstP) experiment could be downloaded [here](https://drive.google.com/open?id=1lgJek1of0T1X_5IJgqmgvN7VBnNDq0JL) 
The TREC eval set query embedding and their ids for our document 2048 ANCE(MaxP) experiment could be downloaded [here](https://drive.google.com/open?id=1Q3ltIJ0psGTafESnTE27Jxpq0X7anS-X)

The t-SNE plots for all the queries in the TREC document eval set for ANCE(FirstP) could be viewed [here](https://drive.google.com/drive/folders/1-3kpPyTWC15sEZVv9uvBmDpEOkanrx8W)

run_train.sh and run_ann_data_gen.sh files contain the commands with the parameters we used for passage ANCE(FirstP), document ANCE(FirstP) and document 2048 ANCE(MaxP) to reproduce the results in this section.
run_train_warmup.sh contains the commands to reproduce the results for the pretrained BM25 warmup checkpoint in this section

Note the steps to reproduce similar results as shown in the table might be a little different due to different synchronizing between training and ann data generation processes and other possible environment differences of the user experiments.


