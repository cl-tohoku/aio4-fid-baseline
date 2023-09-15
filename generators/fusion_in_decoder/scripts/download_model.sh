#!/bin/bash
USAGE="bash $0 [targets=reader] [output_dir]"
DATE=`date +%Y%m%d-%H%M`

set -e

TARGETS=($(echo $1 | tr ',' "\n"))
DEST=$2

if [ -z $DEST ] ; then
  echo "[ERROR] Please specify the 'output_dir'"
  echo $USAGE
  exit 1
elif [ ! -z $3 ] ; then
  echo "[ERROR] The arguemnt of $0 takes 2 positional arguments but 3 (or more) was given"
  echo $USAGE
  exit 1
fi


for target in ${TARGETS[@]} ; do
  if [ $target = "reader" ] ; then
    echo "- Download: reader"
    gcloud storage cp gs://aio4_fid_baseline_test/FiD_baseline_model/config.json $DEST
    gcloud storage cp gs://aio4_fid_baseline_test/FiD_baseline_model/optimizer.pth.tar.gz $DEST
    gcloud storage cp gs://aio4_fid_baseline_test/FiD_baseline_model/pytorch_model.bin.gz $DEST
  else
    echo "[WARNING] '${target}' is not appropriate"
    echo "[WARNING] Please specify the target from {reader}"
  fi
done
