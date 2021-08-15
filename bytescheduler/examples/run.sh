#! /bin/bash

NUM_PROC=4
HOSTS="localhost:4"


ADD_LIB_PATH="/opt/amazon/openmpi/lib:/home/ubuntu/anaconda3/envs/horovod_dev/lib"
PY_BIN="/home/ubuntu/anaconda3/envs/horovod_dev/bin/python"
SCRIPT_PATH="/home/ubuntu/byteps/bytescheduler/examples/pytorch_horovod_benchmark.py"
ADD_PY_PATH="/home/ubuntu/anaconda3/envs/horovod_dev/lib/python3.7/site-packages/"
# FP16_ALLREDUCE="--fp16-allreduce"
FP16_ALLREDUCE=""
N_ITER=100
N_CLASSES=1000
HOROVOD_CYCLE_TIME="1"
HOROVOD_FUSION_THRESHOLD="134217728"
USE_BYTESCHEDULER="1"
BYTESCHEDULER_PARTITION_TUNING="0"
BYTESCHEDULER_PARTITION=4000000
BYTESCHEDULER_CREDIT_TUNING="1"
BYTESCHEDULER_CREDIT=40000000

TEST_MODEL="vgg16"

cmd1="/opt/amazon/openmpi/bin/mpirun -np ${NUM_PROC} -H ${HOSTS} \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH=${ADD_LIB_PATH}:$LD_LIBRARY_PATH \
    -x HOROVOD_CYCLE_TIME=${HOROVOD_CYCLE_TIME} \
    -x HOROVOD_FUSION_THRESHOLD=${HOROVOD_FUSION_THRESHOLD} \
    -x USE_BYTESCHEDULER=${USE_BYTESCHEDULER} \
    -x PYTHONPATH=${ADD_PY_PATH}:$PYTHONPATH \
    -x BYTESCHEDULER_PARTITION=${BYTESCHEDULER_PARTITION} \
    -x BYTESCHEDULER_PARTITION_TUNING=${BYTESCHEDULER_PARTITION_TUNING} \
    -x BYTESCHEDULER_CREDIT_TUNING=${BYTESCHEDULER_CREDIT_TUNING} \
    -x BYTESCHEDULER_CREDIT=${BYTESCHEDULER_CREDIT} \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    ${PY_BIN} \
    ${SCRIPT_PATH} \
      --model ${TEST_MODEL} --num-iters ${N_ITER} ${FP16_ALLREDUCE} --num-classes ${N_CLASSES}"

cmd_bert="/opt/amazon/openmpi/bin/mpirun -np ${NUM_PROC} -H ${HOSTS} \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH=${ADD_LIB_PATH}:$LD_LIBRARY_PATH \
    -x HOROVOD_CYCLE_TIME=${HOROVOD_CYCLE_TIME} \
    -x HOROVOD_FUSION_THRESHOLD=${HOROVOD_FUSION_THRESHOLD} \
    -x USE_BYTESCHEDULER=${USE_BYTESCHEDULER} \
    -x PYTHONPATH=${ADD_PY_PATH}:$PYTHONPATH \
    -x BYTESCHEDULER_PARTITION=${BYTESCHEDULER_PARTITION} \
    -x BYTESCHEDULER_PARTITION_TUNING=${BYTESCHEDULER_PARTITION_TUNING} \
    -x BYTESCHEDULER_CREDIT_TUNING=${BYTESCHEDULER_CREDIT_TUNING} \
    -x BYTESCHEDULER_CREDIT=${BYTESCHEDULER_CREDIT} \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    ${PY_BIN} \
    /home/ubuntu/test-transformer/examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name MNLI \
    --do_train \
    --data_dir ~/data/glue/MNLI \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/bert-eval/ \
    --overwrite_output_dir"


pkill python
# eval ${cmd1}
eval ${cmd_bert}