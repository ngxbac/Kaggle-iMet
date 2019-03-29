#!/usr/bin/env bash

set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

echo "Training..."

for model in resnet34; do
    for fold in 0; do
        LOGDIR=/media/ngxbac/DATA/logs_iwildcam/${model}/fold_${fold}/
        catalyst-dl run --config=./iwildcam/configs/config.yml \
                        --logdir=$LOGDIR \
                        --model_params/params/arch=$model:str \
                        --stages/data_params/train_csv=/media/ngxbac/Bac2/fgvc6/iwildcam/csv/kfold/train_$fold.csv.gz:str \
                        --stages/data_params/valid_csv=/media/ngxbac/Bac2/fgvc6/iwildcam/csv/kfold/valid_$fold.csv.gz:str \
                        --expdir=iwildcam \
                        --verbose
    done
done