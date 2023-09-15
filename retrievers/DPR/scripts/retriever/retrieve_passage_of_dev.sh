#!/bin/bash
USAGE="bash $0 [-n NAME] [-m MODEL] [-e EMBED] [-g GPU]"
DATE=`date +%Y%m%d-%H%M`

while getopts n:m:e:g: opt ; do
  case ${opt} in
    n ) FLG_N="TRUE"; NAME=${OPTARG};;
    m ) FLG_M="TRUE"; MODEL=${OPTARG};;
    e ) FLG_E="TRUE"; EMBEDDING=${OPTARG};;
    g ) FLG_G="TRUE"; GPU=${OPTARG};;
    * ) echo ${USAGE} 1>&2; exit 1 ;;
  esac
done

test "${FLG_N}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_M}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_E}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_G}" = "TRUE" && export CUDA_VISIBLE_DEVICES=$GPU


# Retrieve Passages ======================================

set -ex
source scripts/configs/config.pth

DIR_PROJECT=$DIR_RESULT/$NAME
mkdir -p $DIR_PROJECT/retrieved

BASENAME=${MODEL##*/}
TMP=${MODEL#*.}
EPOCH_STEP=${TMP%%.pt*}  # dpr_biencoder_{EPOCH_STEP}.pt

params="
--n-docs 100
--validation_workers 32
--batch_size 64
--projection_dim 768
"

declare -A QA_FILE
QA_FILE["dev"]=$DEV_FILE


FO_FILE=$DIR_PROJECT/retrieved/dev_aio_${EPOCH_STEP}.json
LOG_FILE=$DIR_PROJECT/retrieved/logs/predict_dev_${DATE}.log
mkdir -p `dirname $LOG_FILE`
echo "# bash $0 $@" > $LOG_FILE

python dense_retriever.py \
  ${params} \
  --model_file $MODEL \
  --ctx_file $WIKI_FILE \
  --encoded_ctx_file $EMBEDDING \
  --qa_file ${QA_FILE[dev]} \
  --out_file $FO_FILE \
| tee -a $LOG_FILE
