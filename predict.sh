#!/usr/bin/env bash
set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

LOGDIR=/media/ngxbac/DATA/logs_datahack/intel-scene/

echo "Inference best checkpoint..."
echo "Inference..."

fold=0
model=resnet34
catalyst-dl run \
    --config=./intel-scene/inference.yml \
    --logdir=$LOGDIR \
    --model_params/params/arch=$model:str \
    --expdir=intel-scene \
    --verbose