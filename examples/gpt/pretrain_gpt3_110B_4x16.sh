# This is an example: training gpt3

source env_npu.sh

export JIT_COMPILE=False
export HCCL_OP_BASE_FFTS_MODE_ENABLE=True
export MULTI_STREAM_MEMORY_REUSE=1

GPUS_PER_NODE=16
# Change for multinode config
# IPs is a list of IP for all nodes, and the first node is the master node
IPs=('xxx.xxx.xxx.1' 'xxx.xxx.xxx.2' 'xxx.xxx.xxx.3' 'xxx.xxx.xxx.4')
LOCAL_HOST=`hostname -I | awk -F " " '{print$1}'`
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6000
NNODES=${#IPs[@]}
NODE_RANK=""
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

for i in "${!IPs[@]}";
do
    echo "${IPs[$i]}"
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
    then
        NODE_RANK=$i
        break
    fi
done
if [[ $NODE_RANK == "" ]]; then
    echo "[Error] para \"NODE_RANK\" must be config"
    exit 1
fi

echo "local host ip: $LOCAL_HOST"
echo "master addr: $MASTER_ADDR, master port: $MASTER_PORT"
echo "total nodes: $NNODES, node rank: $NODE_RANK, device per node: $GPUS_PER_NODE"

DATA_PATH=./megatron_npu_adaptor/dataset/enwiki/gpt_text_sentence
CHECKPOINT_PATH=./checkpoint

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --use-flash-attn \
       --vocab-size 64000 \
       --make-vocab-size-divisible-by 64 \
       --attention-dropout 0.0 \
       --num-layers 80 \
       --hidden-size 10240 \
       --num-attention-heads 80 \
       --micro-batch-size 1 \
       --global-batch-size 192 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --train-iters 500000 \
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
       --no-gradient-accumulation-fusion \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --bf16 \
       --pre-tockens 65536 \
       --next-tockens 0 \
       --shape-order SBH \
       --reset-attention-mask \
       --sequence-parallel | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
TrainingTime=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$3}' | awk -F '|' '{print$1}' | tail -n +21 | awk '{sum+=$1} END {print"",sum/NR}'`
echo "Elapsed Time Per iteration: $TrainingTime"