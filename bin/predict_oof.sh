#!/usr/bin/env bash
set -e

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

model_name=se_resnext50_32x4d
logbase_dir=/raid/bac/kaggle/logs/imet2019/finetune/
image_size=512
image_key=id
label_key=attribute_ids
n_class=1103


for fold in 0 1 2 3 4 5; do
    dataset=valid
    infer_csv=./imet/csv/kfold6/valid_$fold.csv.gz
    infer_root=/raid/data/kaggle/imet2019/train/
    python imet/inference.py infer-one-fold     --model_name=$model_name \
                                                --dataset=$dataset \
                                                --infer_csv=$infer_csv \
                                                --infer_root=$infer_root \
                                                --logbase_dir=$logbase_dir \
                                                --image_size=$image_size \
                                                --image_key=$image_key \
                                                --label_key=$label_key \
                                                --n_class=$n_class \
                                                --fold=$fold
done
