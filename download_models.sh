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

mkdir -p $DPR_DEST
mkdir -p $FiD_DEST

# Download models
echo "- Download: retriever(DPR)"
wget -nc https://storage.googleapis.com/aio-public-tokyo/aio4_fid_baseline_models/DPR_baseline_model/biencoder.pt.gz -P $DPR_DEST

echo "- Download: embeddings"
wget -nc https://storage.googleapis.com/aio-public-tokyo/aio4_fid_baseline_models/DPR_baseline_model/embedding.pickle.gz -P $DPR_DEST

echo "- Download: reader(FiD)"
wget -nc https://storage.googleapis.com/aio-public-tokyo/aio4_fid_baseline_models/FiD_baseline_model/config.json -P $FiD_DEST

wget -nc https://storage.googleapis.com/aio-public-tokyo/aio4_fid_baseline_models/FiD_baseline_model/optimizer.pth.tar.gz -P $FiD_DEST

wget -nc https://storage.googleapis.com/aio-public-tokyo/aio4_fid_baseline_models/FiD_baseline_model/pytorch_model.bin.gz -P $FiD_DEST

# unzipping
echo "- Unzip: DPR model"
gunzip $DPR_DEST/biencoder.pt.gz
gunzip $DPR_DEST/embedding.pickle.gz
echo "- Unzip: FiD model"
gunzip $FiD_DEST/optimizer.pth.tar.gz
gunzip $FiD_DEST/pytorch_model.bin.gz
