#!/usr/bin/bash

DATE=`date +%Y%m%d-%H%M`
echo $DATE

python test_generator.py --config_file $1 

DATE=`date +%Y%m%d-%H%M`
echo $DATE
