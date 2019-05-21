#!/usr/bin/env bash
set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

model_name=se_resnext50_32x4d
dataset=infer
infer_csv=/raid/data/kaggle/imet2019/sample_submission.csv
infer_root=/raid/data/kaggle/imet2019/test/

logbase_dir=/raid/bac/kaggle/logs/imet2019/finetune/
image_size=512
image_key=id
label_key=attribute_ids
n_class=1103

python imet/inference.py infer-kfold    --model_name=$model_name \
                                        --dataset=$dataset \
                                        --infer_csv=$infer_csv \
                                        --infer_root=$infer_root \
                                        --logbase_dir=$logbase_dir \
                                        --image_size=$image_size \
                                        --image_key=$image_key \
                                        --label_key=$label_key \
                                        --n_class=$n_class