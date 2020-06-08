#!/bin/bash
#
# This script is for training with updated ann driver
#
# The design for this ann driver is to have 2 separate processes for training: one for passage/query 
# inference using trained checkpoint to generate ann data and calcuate ndcg, another for training the model 
# using the ann data generated. Data between processes is shared on common directory, model_dir for checkpoints
# and model_ann_data_dir for ann data.
#
# This script initialize the training and start the model training process
# It first preprocess the msmarco data into indexable cache, then generate a single initial ann data
# version to train on, after which it start training on the generated ann data, continously looking for
# newest ann data generated in model_ann_data_dir
#
# To start training, you'll need to run this script first
# after intial ann data is created (you can tell by either finding "successfully created 
# initial ann training data" in console output or if you start seeing new model on tensorboard),
# start run_ann_data_gen.sh in another dlts job (or same dlts job using split GPU)
#
# Note if preprocess directory or ann data directory already exist, those steps will be skipped
# and training will start immediately

gpu_no=3

data_type=1 # 1 for passage, 0 for doc
# model type
model_type="triple_ann_200"
tokenizer_type="roberta-base"
seq_length=131
triplet="--triplet" # set this to empty for non triplet model

# hyper parameters
batch_size=16
gradient_accumulation_steps=2
ann_topk=200
ann_negative_sample=20
learning_rate=1e-5
warmup_steps=300

# input/output directories
base_data_dir="/mnt/azureblob/lexion/glue_data/MSMarcoANN/"
preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
job_name="triple_ann_200_test"
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
pretrained_checkpoint_dir="/mnt/azureblob/lexion/glue_data/MSMarco20M/8gpu_meanemb_triple/checkpoint-60000/"

preprocess_cmd="\
sudo python msmarco_doc_data.py --data_dir $base_data_dir --out_data_dir $preprocessed_data_dir --model_type $model_type \
--model_name_or_path $tokenizer_type --max_seq_length $seq_length --data_type $data_type\
"

echo $preprocess_cmd
eval $preprocess_cmd

if [[ $? = 0 ]]; then
    echo "successfully created preprocessed data"
else
	echo "preprocessing failed"
    echo "failure: $?"
    exit 1
fi

initial_data_gen_cmd="\
sudo python -m torch.distributed.launch --nproc_per_node=$gpu_no run_ann_data_gen.py --training_dir $model_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size $batch_size --topk_training $ann_topk --negative_sample $ann_negative_sample --end_output_num 0 \
"

echo $initial_data_gen_cmd
eval $initial_data_gen_cmd

if [[ $? = 0 ]]; then
    echo "successfully created initial ann training data"
else
	echo "initial data generation failed"
    echo "failure: $?"
    exit 1
fi


train_cmd="\
sudo python -m torch.distributed.launch --nproc_per_node=$gpu_no run_ann_doc.py --model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco $triplet --data_dir $preprocessed_data_dir \
--ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
--warmup_steps $warmup_steps --logging_steps 100 --save_steps 10000 --log_dir "~/tensorboard/${DLWS_JOB_ID}/logs/${job_name}"\
"

echo $train_cmd
eval $train_cmd
