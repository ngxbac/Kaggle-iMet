#!/usr/bin/env bash

set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

echo "Training..."

for model in densenet121 densenet169 resnet50 inception_v3 inception_v4; do
    for fold in 0 1 2 3 4 5 6; do
        LOGDIR=/media/ngxbac/DATA/logs_iwildcam/${model}/fold_${fold}/
        catalyst-dl run --config=./iwildcam/configs/config.yml \
                        --logdir=$LOGDIR \
                        --model_params/params/arch=$model:str \
                        --stages/stage1/data_params/train_csv=/media/ngxbac/Bac2/fgvc6/iwildcam/csv/kfold/train_$fold.csv.gz:str \
                        --stages/stage1/data_params/valid_csv=/media/ngxbac/Bac2/fgvc6/iwildcam/csv/kfold/valid_$fold.csv.gz:str \
                        --stages/infer/callbacks_params/infer/out_dir=$LOGDIR:str \
                        --expdir=iwildcam \
                        --verbose
    done
done