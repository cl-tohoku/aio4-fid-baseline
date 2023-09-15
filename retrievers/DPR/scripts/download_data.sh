#!/bin/bash
USAGE="bash $0 [output_dir]"
DATE=`date +%Y%m%d-%H%M`

set -e

DEST=$1

if [ -z $DEST ] ; then
  echo "[ERROR] Please specify the 'output_dir'"
  echo $USAGE
  exit 1
fi


# Datasets
mkdir -p $DEST
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_train.jsonl                  -O $DEST/aio_02_train.jsonl
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_04/aio_04_dev_unlabeled_v1.0.jsonl     -O $DEST/aio_04_dev_unlabeled_v1.0.jsonl
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_04/aio_04_dev_v1.0.jsonl               -O $DEST/aio_04_dev_v1.0.jsonl
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_04/aio_04_test_lb_unlabeled_v1.0.jsonl -O $DEST/aio_04_test_lb_unlabeled_v1.0.jsonl


# DPR
mkdir -p $DEST/retriever
curl -o $DEST/retriever/aio_02_train.json.gz -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/datasets.jawiki-20220404-c400-large.aio_02_train.jsonl.gz
curl -o $DEST/retriever/aio_02_dev.json.gz -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/datasets.jawiki-20220404-c400-large.aio_02_dev.jsonl.gz

curl -o $DEST/retriever/aio_02_train.tsv -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.qas.aio_02_train.tsv
curl -o $DEST/retriever/aio_02_dev.tsv -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.qas.aio_02_dev.tsv

# wikipedia
mkdir -p $DEST/wiki
curl -o $DEST/wiki/jawiki-20220404-c400-large.tsv.gz -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.wikipedia_split.jawiki-20220404-c400-large.tsv.gz


echo -en "\n===========================================\n"
ls -R -lh $DEST
