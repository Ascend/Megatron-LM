#! /bin/bash
source env_npu.sh

TASK="LAMBADA"

VALID_DATA=lambada_test.json
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=./checkpoint_dist
COMMON_TASK_ARGS="--num-layers 8 \
                  --hidden-size 12288 \
                  --num-attention-heads 96 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --tensor-model-parallel-size 8 \
                  --use-distributed-optimizer \
                  --pipeline-model-parallel-size 8 \
                  --distributed-backend nccl \
                  --recompute-granularity full \
                  --recompute-method uniform \
                  --no-gradient-accumulation-fusion \
                  --vocab-file $VOCAB_FILE \
                  --fp16"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=60035
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../../main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng