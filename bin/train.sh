#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config.yml

for model in se_resnext50_32x4d; do
    for fold in 0; do
        LOGDIR=/raid/bac/kaggle/logs/imet/test/$model/fold_$fold/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --model_params/arch=$model:str \
            --stages/data_params/train_csv=./csv/kfold6/train_$fold.csv.gz:str \
            --stages/data_params/valid_csv=./csv/kfold6/valid_$fold.csv.gz:str \
            --verbose
    done
done