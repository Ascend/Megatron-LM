#! /bin/bash
source env_npu.sh
# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=60035
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=../dataset/enwiki/gpt_text_sentence
CHECKPOINT_PATH=./checkpoint_dist_65B

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
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

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../../pretrain_gpt.py \
       --num-layers 80 \
       --hidden-size 8192 \
       --num-attention-heads 64 \
       --micro-batch-size 1 \
       --global-batch-size 48 \
       --seq-length 6144 \
       --max-position-embeddings 6144 \
       --train-iters 10000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tensor-model-parallel-size 8 \
       --use-distributed-optimizer \
       --pipeline-model-parallel-size 4 \
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
       --use-flash-attn \
       --pre-tockens $pre_tockens \
       --next-tockens $next_tockens \
       --shape-order $shape_order \
       --sequence-parallel \
       --log-interval 1 \
       --save-interval 150000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --bf16 \
       --ffn-hidden-size 43712 \
       --is-x-model
       
