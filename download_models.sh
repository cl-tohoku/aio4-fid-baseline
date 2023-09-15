#!/bin/bash
USAGE="bash $0"
DATE=`date +%Y%m%d-%H%M`

set -e

DPR_DEST="retrievers/DPR/models/baseline"
FiD_DEST="generators/fusion_in_decoder/models_and_results/baseline"

if [ ! -z $1 ] ; then
  echo "[ERROR] The arguemnt of $0 doesn't take any positional arguments but 1 (or more) was given"
  echo $USAGE
  exit 1
fi

# DPR models
echo "- Download: retriever(DPR)"
gcloud storage cp gs://aio4_fid_baseline_test/DPR_baseline_model/biencoder.pt.gz $DPR_DEST/biencoder.pt.gz
echo "- Download: embeddings"
gcloud storage cp gs://aio4_fid_baseline_test/DPR_baseline_model/embedding.pickle.gz $DPR_DEST/embedding.pickle.gz

# FiD models
echo "- Download: reader(FiD)"
gcloud storage cp gs://aio4_fid_baseline_test/FiD_baseline_model/config.json $FiD_DEST/config.json
gcloud storage cp gs://aio4_fid_baseline_test/FiD_baseline_model/optimizer.pth.tar.gz $FiD_DEST/optimizer.pth.tar.gz
gcloud storage cp gs://aio4_fid_baseline_test/FiD_baseline_model/pytorch_model.bin.gz $FiD_DEST/pytorch_model.bin.gz

# unzipping
echo "- Unzip: DPR model"
gunzip $DPR_DEST/biencoder.pt.gz
gunzip $DPR_DEST/embedding.pickle.gz
echo "- Unzip: FiD model"
gunzip $FiD_DEST/optimizer.pth.tar.gz
gunzip $FiD_DEST/pytorch_model.bin.gz
