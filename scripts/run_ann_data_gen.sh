#!/bin/bash
#
# This script is for generate ann data for a model in training
#
# For the overall design of the ann driver, check run_train.sh
#
# This script continuously generate ann data using latest model from model_dir
# For training, run this script after initial ann data is created from run_train.sh
# Make sure parameter used here is consistent with the training script
#


gpu_no=4

# model type
model_type="triple_ann_200"
tokenizer_type="roberta-base"
seq_length=128

# ann parameters
batch_size=16
ann_topk=200
ann_negative_sample=20

# input/output directories
base_data_dir="/webdata-nfs/yeli1/msmarco_doc_5M_30k/"
preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
job_name="debug6"
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
pretrained_checkpoint_dir="/webdata-nfs/yeli1/msmarco_doc_5M_30k/triplet_ann_128_checkpoint_6/checkpoint-16500/"


data_gen_cmd="\
sudo python -m torch.distributed.launch --nproc_per_node=$gpu_no run_ann_data_gen.py --training_dir $model_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size $batch_size --topk_training $ann_topk --negative_sample $ann_negative_sample \
"

echo $data_gen_cmd
eval $data_gen_cmd