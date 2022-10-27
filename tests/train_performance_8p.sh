#!/bin/bash
source ./tests/env_npu.sh

GPUS_PER_NODE=8
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATASET_ROOT_PATH=./dataset/en_wiki/preprocess
DATA_PATH=$DATASET_ROOT_PATH/my-t5_text_sentence
VOCAB_FILE=$DATASET_ROOT_PATH/bert-large-uncased-vocab.txt
CHECKPOINT_PATH=./checkpoint

GlobalBatchSize=128
ExitInterval=1001

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

nohup \
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
       --global-batch-size ${GlobalBatchSize} \
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
       --no-bias-gelu-fusion \
       --exit-interval ${ExitInterval} \
> ./training.log &

wait

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
IterTime=$(grep "elapsed time per iteration" ./training.log | tail -n 1000 | awk '{print $14}' | awk '{sum+=$1} END {print sum/NR}')
FPS=$(echo "${GlobalBatchSize} * 1000 / ${IterTime}"|bc)

#打印，不需要修改
echo "Final Performance samples/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "PPL" ./training.log | awk 'END {print $15}')

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


