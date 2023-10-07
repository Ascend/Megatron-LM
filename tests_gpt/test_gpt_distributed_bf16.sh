#! /bin/bash
source env_npu.sh

# 参数校验，VALID_DATA为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
pre_tockens=-1
next_tockens=-1
shape_order=""
for para in $*
do
    if [[ $para == --pre* ]];then
        pre_tockens=`echo ${para#*=}`
    elif [[ $para == --next* ]];then
        next_tockens=`echo ${para#*=}`
    elif [[ $para == --shape* ]];then
        shape_order=`echo ${para#*=}`
    fi
done
if [[ $pre_tockens == -1 ]];then
    echo "[Error] para \"pre_tockens\" must be confing"
    exit 1
fi
if [[ $next_tockens == -1 ]];then
    echo "[Error] para \"next_tockens\" must be confing"
    exit 1
fi
if [[ $shape_order == "" ]];then
    echo "[Error] para \"shape_order\" must be confing"
    exit 1
fi

TASK="LAMBADA"

VALID_DATA=lambada_test.json
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=./checkpoint_dist
COMMON_TASK_ARGS="--num-layers 8 \
                  --hidden-size 5120 \
                  --num-attention-heads 40 \
                  --seq-length 8192 \
                  --max-position-embeddings 8192 \
                  --tensor-model-parallel-size 8 \
                  --use-distributed-optimizer \
                  --pipeline-model-parallel-size 8 \
                  --distributed-backend nccl \
                  --no-gradient-accumulation-fusion \
                  --use-flash-attn \
                  --pre-tockens $pre_tockens \
                  --next-tockens $next_tockens \
                  --shape-order $shape_order \
                  --sequence-parallel \
                  --vocab-file $VOCAB_FILE \
                  --bf16"

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