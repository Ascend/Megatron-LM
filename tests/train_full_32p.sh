#!/bin/bash
source ./tests/env_npu.sh

# NPU
export HCCL_IF_IP=xxxx # 当前机器IP地址，需手动设置
epoxrt HCCL_CONNECT_TIMEOUT=3600
epoxrt HCCL_EXEC_TIMEOUT=3600

GPUS_PER_NODE=8
# Change for multinode config
export MASTER_ADDR=xxxx # 设置主机诶单，需手动设置
export MASTER_PORT=23333
NNODES=4
NODE_RANK=0 # 依次设置0,1,2,3，需手动设置
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATASET_ROOT_PATH=./dataset/en_wiki/preprocess
DATA_PATH=$DATASET_ROOT_PATH/my-t5_text_sentence
VOCAB_FILE=$DATASET_ROOT_PATH/bert-large-uncased-vocab.txt
CHECKPOINT_PATH=./checkpoint

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       runner.py \
       --tensor-model-parallel-size 2 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --kv-channels 64 \
       --ffn-hidden-size 3072 \
       --encoder-seq-length 512 \
       --decoder-seq-length 128 \
       --micro-batch-size 16 \
       --global-batch-size 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --lr-decay-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16  \
       --vocab-extra-ids 100 \
       --use-cpu-initialization \
       --no-bias-gelu-fusion

