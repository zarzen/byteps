#! /bin/bash

NUM_PROC=4
HOSTS="localhost:4"


ADD_LIB_PATH="/opt/amazon/openmpi/lib:/home/ubuntu/anaconda3/envs/horovod_dev/lib"
PY_BIN="/home/ubuntu/anaconda3/envs/horovod_dev/bin/python"
SCRIPT_PATH="/home/ubuntu/byteps/bytescheduler/examples/pytorch_horovod_benchmark.py"
ADD_PY_PATH="/home/ubuntu/anaconda3/envs/horovod_dev/lib/python3.7/site-packages/"
# FP16_ALLREDUCE="--fp16-allreduce"
FP16_ALLREDUCE=""
N_ITER=10
HOROVOD_CYCLE_TIME="1"
HOROVOD_FUSION_THRESHOLD="134217728"
USE_BYTESCHEDULER="1"

cmd1="/opt/amazon/openmpi/bin/mpirun -np ${NUM_PROC} -H ${HOSTS} \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH=${ADD_LIB_PATH}:$LD_LIBRARY_PATH \
    -x HOROVOD_CYCLE_TIME=${HOROVOD_CYCLE_TIME} \
    -x HOROVOD_FUSION_THRESHOLD=${HOROVOD_FUSION_THRESHOLD} \
    -x USE_BYTESCHEDULER=${USE_BYTESCHEDULER} \
    -x PYTHONPATH=${ADD_PY_PATH}:$PYTHONPATH \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    ${PY_BIN} \
    ${SCRIPT_PATH} \
      --model resnet50 --num-iters ${N_ITER} ${FP16_ALLREDUCE}"

pkill python
eval ${cmd1}