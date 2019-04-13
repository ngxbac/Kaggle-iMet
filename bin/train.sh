#!/usr/bin/env bash

set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

echo "Training..."

export CUDA_VISIBLE_DEVICES=2,3

for model in resnet34; do
    for fold in 0; do
        LOGDIR=./logs_imet/${model}_warm/fold_${fold}/
        catalyst-dl run --config=./imet/configs/config.yml \
                        --logdir=$LOGDIR \
                        --model_params/params/arch=$model:str \
                        --stages/infer/callbacks_params/infer/out_dir=$LOGDIR:str \
                        --expdir=imet \
                        --verbose
    done
done
