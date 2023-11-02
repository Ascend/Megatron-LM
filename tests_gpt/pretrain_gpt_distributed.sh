#! /bin/bash
source env_npu.sh
# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=../dataset/enwiki/my-t5_text_sentence
CHECKPOINT_PATH=./checkpoint_dist

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../../pretrain_gpt.py \
       --num-layers 8 \
       --hidden-size 12288 \
       --num-attention-heads 96 \
       --micro-batch-size 4 \
       --global-batch-size 96 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tensor-model-parallel-size 8 \
       --use-distributed-optimizer \
       --pipeline-model-parallel-size 8 \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.375e-5 \
       --lr-decay-style cosine \
       --min-lr 0.375e-6 \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --init-method-std 0.006 \
       --recompute-granularity full \
       --recompute-method uniform \
       --no-gradient-accumulation-fusion \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
