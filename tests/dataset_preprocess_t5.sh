# Step1: Concat json
WIKI_DIR=./text
OUTDIR=./output

mkdir -p $OUTDIR
rm $OUTDIR/wiki_all.json
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

# Step2: Do Preprocess
VOCAB=./bert-large-uncased-vocab.txt
python3 ../tools/preprocess_data.py \
       --input $OUTDIR/wiki_all.json \
       --output-prefix $OUTDIR/my-t5 \
       --vocab-file $VOCAB \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --chunk-size 32 \
       --workers $(nproc)
