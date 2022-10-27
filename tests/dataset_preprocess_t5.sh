# Step 1: Download enwiki
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -dk enwiki-latest-pages-articles.xml.bz2


# Step 2: Download WikiExtractor
pip3 install wikiextractor
python3 -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml --json


# Step3: Concat json
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


# Step4: Download Vocab and Do Preprocess
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
VOCAB=./bert-large-uncased-vocab.txt
python3tools/preprocess_data.py \
       --input $OUTDIR/wiki_all.json \
       --output-prefix $OUTDIR/my-t5 \
       --vocab $VOCAB \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers $(nproc)


