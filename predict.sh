#!/usr/bin/env bash
set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

LOGDIR=/media/ngxbac/DATA/logs_datahack/intel-scene/

echo "Inference best checkpoint..."
echo "Inference..."


catalyst-dl run \
   --expdir=intel-scene \
   --resume=${LOGDIR}/checkpoints/best.pth \
   --out-prefix=${LOGDIR}/dataset.predictions.{suffix}.npy \
   --config=${LOGDIR}/config.json,./intel-scene/inference.yml \
   --verbose