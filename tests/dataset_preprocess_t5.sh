# Step1: 生成wiki_all.json文件

# WikiExtractor生成text文件目录
WIKI_DIR=./dataset/text
# 生成数据集位置
OUTDIR=./dataset/enwiki

mkdir -p $OUTDIR
# 对数据集目录做权限控制，保障数据集文件处理过程中的安全
chmod 750 $OUTDIR

touch $OUTDIR/wiki_all.json

find "$WIKI_DIR" -type f  -print0 |
    while IFS= read -r -d '' line; do
            filename=$(echo "$line" | rev | cut -d'/' -f 1 | rev)
            subfilename=$(echo "$line" | rev | cut -d'/' -f 2 | rev)
            prefix="${subfilename}_${filename}"
            new_name=$(echo "$line")
            echo "Procesing $prefix, $filename, $new_name"
            cat $new_name >> $OUTDIR/wiki_all.json
    done

# Step2: 执行原生Megatron-LM的preprocess_data.py，实现数据集预处理
VOCAB=./dataset/bert-large-uncased-vocab.txt
python3 ../tools/preprocess_data.py \
       --input $OUTDIR/wiki_all.json \
       --output-prefix $OUTDIR/my-t5 \
       --vocab-file $VOCAB \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --chunk-size 32 \
       --workers $(nproc)

# 对数据集目录及文件做权限控制
chmod -R 640 $OUTDIR
chmod 750 $OUTDIR