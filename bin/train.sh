#!/usr/bin/env bash

set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

echo "Training..."

export CUDA_VISIBLE_DEVICES=1,2,3

for model in se_resnext50_32x4d; do
    for fold in 1 2 3 4 5; do
        LOGDIR=./logs_imet/finetune/${model}_512/fold_${fold}/
        catalyst-dl run --config=./imet/configs/config.yml \
                        --logdir=$LOGDIR \
                        --model_params/params/arch=$model:str \
                        --stages/infer/callbacks_params/infer/out_dir=$LOGDIR:str \
                        --expdir=imet \
                        --verbose
    done
done
