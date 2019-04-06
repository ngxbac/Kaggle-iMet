#!/usr/bin/env bash

set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

echo "Training..."

export CUDA_VISIBLE_DEVICES=1

for model in resnet34; do
    for fold in 0; do
        LOGDIR=./logs_imet/${model}_bcef2focal/fold_${fold}/
        catalyst-dl run --config=./imet/configs/config.yml \
                        --logdir=$LOGDIR \
                        --model_params/params/arch=$model:str \
                        --stages/stage1/data_params/train_csv=./imet/csv/kfold6/train_$fold.csv.gz:str \
                        --stages/stage1/data_params/valid_csv=./imet/csv/kfold6/valid_$fold.csv.gz:str \
                        --stages/stage2/data_params/train_csv=./imet/csv/kfold6/train_$fold.csv.gz:str \
                        --stages/stage2/data_params/valid_csv=./imet/csv/kfold6/valid_$fold.csv.gz:str \
                        --stages/infer/callbacks_params/infer/out_dir=$LOGDIR:str \
                        --expdir=imet \
                        --verbose
    done
done
