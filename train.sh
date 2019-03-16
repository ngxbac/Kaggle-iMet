#!/usr/bin/env bash

set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

echo "Training..."

for model in inception_v4 xception se_resnext50_32x4d; do
    for fold in 0 1 2 3 4; do
        LOGDIR=/media/ngxbac/DATA/logs_datahack/intel-scene/${model}_$fold
        catalyst-dl run --config=./intel-scene/config.yml \
                        --logdir=$LOGDIR \
                        --model_params/params/arch=$model:str \
                        --stages/data_params/train_csv=/media/ngxbac/Bac2/datahack/intel-scene/data/kfold/train_$fold.csv:str \
                        --stages/data_params/valid_csv=/media/ngxbac/Bac2/datahack/intel-scene/data/kfold/valid_$fold.csv:str \
                        --expdir=intel-scene \
                        --verbose
    done
done