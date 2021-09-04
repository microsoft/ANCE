This repository provides the fine-tuning stage on Marco ranking task for [SEED-Encoder](https://arxiv.org/abs/2102.09206) and is based on ANCE (https://github.com/microsoft/ANCE).

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6

## Requirements

To install requirements, run the following commands:

```setup
cd SEED_Encoder
python setup.py install
```



# Fine-tuning for SEED-Encoder
* We follow the ranking experiments in ANCE ([Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf) ) as our downstream tasks.




## Our Checkpoints
[Pretrained SEED-Encoder with 3-layer decoder, attention span = 2 ](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-3-decoder-layers.tar)

[Pretrained SEED-Encoder with 1-layer decoder, attention span = 8 ](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-1-decoder-layer.tar)

[SEED-Encoder warmup checkpoint](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-warmup-90000.tar)

[ANCE finetuned SEED-Encoder checkpoint on passage ranking task](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-pass-440000.tar)

[ANCE finetuned SEED-Encoder checkpoint on document ranking task](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-doc-800000.tar)





## Data Preprocessing

        seq_length=512
        tokenizer_type="seed-encoder"
        base_data_dir={}
        data_type={}
        model_path={}
        preprocessed_data_dir=${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/

        preprocess_cmd="\
        python ../data/msmarco_data.py --data_dir $base_data_dir --out_data_dir $preprocessed_data_dir --model_name_or_path $model_path --model_type seeddot_nll --max_seq_length $seq_length --data_type $data_type "

        echo $preprocess_cmd
        eval $preprocess_cmd


## Warmup for Training


    DATA_DIR=../../data/raw_data
    SAVE_DIR=../../temp/
    LOAD_DIR=$your_path/SEED-Encoder-1-decoder-layer/

    python3 -m torch.distributed.launch --nproc_per_node=8 ../drivers/run_warmup.py \
    --train_model_type seeddot_nll --model_name_or_path $LOAD_DIR --task_name MSMarco --do_train \
    --evaluate_during_training --data_dir $DATA_DIR \
    --max_seq_length 128 --per_gpu_eval_batch_size=256  --per_gpu_train_batch_size=32 --learning_rate 2e-4 --logging_steps 100 --num_train_epochs 2.0 \
    --output_dir $SAVE_DIR --warmup_steps 1000 --overwrite_output_dir --save_steps 20000 --gradient_accumulation_steps 1 --expected_train_size 35000000 \
    --logging_steps_per_eval 100 --fp16 --optimizer lamb --log_dir $SAVE_DIR/log --do_lower_case --fp16



    DATA_DIR=../../data/raw_data
    SAVE_DIR=../../temp/
    LOAD_DIR=$your_path/SEED-Encoder-3-decoder-layers/

    python3 -m torch.distributed.launch --nproc_per_node=8 ../drivers/run_warmup.py \
    --train_model_type seeddot_nll --model_name_or_path $LOAD_DIR --task_name MSMarco --do_train \
    --evaluate_during_training --data_dir $DATA_DIR \
    --max_seq_length 128 --per_gpu_eval_batch_size=256  --per_gpu_train_batch_size=32 --learning_rate 2e-4 --logging_steps 100 --num_train_epochs 2.0 \
    --output_dir $SAVE_DIR --warmup_steps 1000 --overwrite_output_dir --save_steps 20000 --gradient_accumulation_steps 1 --expected_train_size 35000000 \
    --logging_steps_per_eval 100 --fp16 --optimizer lamb --log_dir $SAVE_DIR/log --do_lower_case --fp16


    

## ANCE Training (passage, you may first use the second command to generate the initial data)

        gpu_no=4
        seq_length=512
        tokenizer_type={}
        model_type=seeddot_nll
        base_data_dir={}
        preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
        job_name=21_09_04_try
        pretrained_checkpoint_dir=${you_model_dir}/SEED-Encoder-pass-440000/
        data_type=1
        warmup_steps=5000
        per_gpu_train_batch_size=16
        gradient_accumulation_steps=1
        learning_rate=1e-6

        model_dir="${base_data_dir}${job_name}/"
        model_ann_data_dir="${model_dir}ann_data/"


        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
        --init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
        --cache_dir "${model_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
        --per_gpu_eval_batch_size 64 --topk_training 200 --negative_sample 20


        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_addr 127.0.0.2 --master_port 35000 ../drivers/run_ann.py --model_type $model_type \
        --model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco --triplet --data_dir $preprocessed_data_dir \
        --ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$per_gpu_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
        --warmup_steps $warmup_steps --logging_steps 100 --save_steps 100000000 --optimizer lamb --single_warmup --cache_dir "${model_dir}cache/" --do_lower_case

        
       
## ANCE Training (document)

        gpu_no=4
        seq_length=512
        tokenizer_type={}
        model_type=seeddot_nll
        base_data_dir={}
        preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
        job_name=21_09_04_try2
        pretrained_checkpoint_dir=${you_model_dir}/SEED-Encoder-doc-800000/
        data_type=0
        warmup_steps=3000
        per_gpu_train_batch_size=4
        gradient_accumulation_steps=4
        learning_rate=5e-6

        model_dir="${base_data_dir}${job_name}/"
        model_ann_data_dir="${model_dir}ann_data/"



        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
        --init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
        --cache_dir "${model_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
        --per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20



        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_addr 127.0.0.2 --master_port 35000 ../drivers/run_ann.py --model_type $model_type \
        --model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco --triplet --data_dir $preprocessed_data_dir \
        --ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$per_gpu_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
        --warmup_steps $warmup_steps --logging_steps 100 --save_steps 100000000 --optimizer lamb --single_warmup --cache_dir "${model_dir}cache/" --do_lower_case

        




