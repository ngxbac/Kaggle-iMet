#!/usr/bin/env bash

set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

echo "Training..."

export CUDA_VISIBLE_DEVICES=2,3

for model in se_resnext50_32x4d; do
    for fold in 5; do
        LOGDIR=/raid/bac/kaggle/logs/imet2019/finetune/${model}_512/fold_${fold}/
        catalyst-dl run --config=./imet/configs/config.yml \
                        --logdir=$LOGDIR \
                        --model_params/params/arch=$model:str \
                        --stages/stage1/data_params/dataset/train/csv_file=./imet/csv/kfold6/train_$fold.csv.gz:str \
                        --stages/stage1/data_params/dataset/valid/csv_file=./imet/csv/kfold6/valid_$fold.csv.gz:str \
                        --stages/stage2/data_params/dataset/train/csv_file=./imet/csv/kfold6/train_$fold.csv.gz:str \
                        --stages/stage2/data_params/dataset/valid/csv_file=./imet/csv/kfold6/valid_$fold.csv.gz:str \
                        --stages/infer/callbacks_params/infer/out_dir=$LOGDIR:str \
                        --expdir=imet \
                        --verbose
    done
done
