ulimit -n 100000
export NCCL_SOCKET_IFNAME=eth0

export BASE_DIR=/home/kwtang/bling/Embedding
NOW=$(date +"%Y%m%d_%H%M")
export UUID=$(cat /proc/sys/kernel/random/uuid)
export model_type=nce_first_ln
export JOBID=${NOW}_${model_type}_${UUID}
export JOB_CODE_DIR=/job/${JOBID}/$(basename ${BASE_DIR})
rm -rf ${JOB_CODE_DIR}
sudo mkdir -p ${JOB_CODE_DIR}
sudo chown `whoami` ${JOB_CODE_DIR}

scriptname=$(readlink -f "$0")
cp $scriptname ${JOB_CODE_DIR}
echo "Copying to ${JOB_CODE_DIR}"
cd ${BASE_DIR}
find . -type f -name "*.py" -exec cp --parents {} ${JOB_CODE_DIR} \;

echo "Job id: ${JOBID}"

cd ${JOB_CODE_DIR}/MSMarco

cmd="python3 -m torch.distributed.launch --nproc_per_node=4 driver.py --train_model_type ${model_type}  --model_name_or_path /mnt/azureblob/kwtang/msmarco/20200505_2337_bm25_nll_first_ln_717533c8-74e0-4bbc-8747-d69d341c62f0/checkpoint-60000/ --task_name MSMarco --do_train --evaluate_during_training --data_dir /home/kwtang/msmarco  --max_seq_length 128     --per_gpu_eval_batch_size=128       --per_gpu_train_batch_size=32       --learning_rate 1e-4  --logging_steps 1000   --num_train_epochs 3.0   --output_dir /mnt/azureblob/kwtang/msmarco/${JOBID}/ --warmup_steps 1000  --overwrite_output_dir --save_steps 15000 --gradient_accumulation_steps 1  --expected_train_size 20000000 --logging_steps_per_eval 5 --log_dir /home/kwtang/tensorboard/${DLWS_JOB_ID}/logs/${JOBID} --fp16 --optimizer lamb"

echo $cmd
eval $cmd