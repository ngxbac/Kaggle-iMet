#!/usr/bin/env python

from collections import OrderedDict  # noqa F401
from torch.utils.data import DataLoader
# from catalyst.dl.experiments import SupervisedRunner
from runner import ModelRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import json
import yaml
import models
from augmentation import infer_tta_aug
from dataset import CsvDataset
import glob

import click

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


@click.group()
def cli():
    print("Inference !!!")


@cli.command()
@click.option('--model_name', type=str, default='se_resnext50_32x4d')
@click.option('--logbase_dir', type=str, default='/raid/bac/kaggle/logs/imet2019/finetune/')
@click.option('--dataset', type=str, default='infer')
@click.option('--infer_csv', type=str)
@click.option('--infer_root', type=str)
@click.option('--image_size', type=int)
@click.option('--image_key', type=str)
@click.option('--label_key', type=str)
@click.option('--n_class', type=int)
def infer_kfold(
    model_name,
    logbase_dir,
    dataset,
    infer_csv,
    infer_root,
    image_size,
    image_key,
    label_key,
    n_class,

):
    for fold in [0, 1, 2, 3, 4, 5]:
        log_dir = f"{logbase_dir}/{model_name}_512/fold_{fold}/"

        model_function = getattr(models, 'finetune')
        params = {
            'arch': model_name,
            'n_class': n_class
        }
        model = model_function(params)
        use_tta = False

        loaders = OrderedDict()
        if infer_csv:
            transforms = infer_tta_aug(image_size)
            if use_tta:
                for i, transform in enumerate(transforms):
                    inferset = CsvDataset(
                        csv_file=infer_csv,
                        root=infer_root,
                        image_key=image_key,
                        label_key=label_key,
                        transform=transform,
                        mode='infer'
                    )

                    infer_loader = DataLoader(
                        dataset=inferset,
                        num_workers=4,
                        shuffle=False,
                        batch_size=32
                    )

                    loaders[f'infer_{i}'] = infer_loader

            else:
                inferset = CsvDataset(
                    csv_file=infer_csv,
                    root=infer_root,
                    image_key=image_key,
                    label_key=label_key,
                    transform=transforms[0],
                    mode='infer'
                )

                infer_loader = DataLoader(
                    dataset=inferset,
                    num_workers=4,
                    shuffle=False,
                    batch_size=32
                )

                loaders[f'{dataset}'] = infer_loader

        all_checkpoints = f"{log_dir}/checkpoints/best.pth"

        callbacks = [
            CheckpointCallback(resume=all_checkpoints),
            InferCallback(out_dir=log_dir, out_prefix="/predicts/")
        ]

        runner = ModelRunner()
        runner.infer(
            model,
            loaders,
            callbacks,
            verbose=True,
        )


@cli.command()
@click.option('--model_name', type=str, default='se_resnext50_32x4d')
@click.option('--logbase_dir', type=str, default='/raid/bac/kaggle/logs/imet2019/finetune/')
@click.option('--dataset', type=str, default='infer')
@click.option('--infer_csv', type=str)
@click.option('--infer_root', type=str)
@click.option('--image_size', type=int)
@click.option('--image_key', type=str)
@click.option('--label_key', type=str)
@click.option('--n_class', type=int)
@click.option('--fold', type=int)
def infer_one_fold(
    model_name,
    logbase_dir,
    dataset,
    infer_csv,
    infer_root,
    image_size,
    image_key,
    label_key,
    n_class,
    fold,
):
    log_dir = f"{logbase_dir}/{model_name}_512/fold_{fold}/"

    model_function = getattr(models, 'finetune')
    params = {
        'arch': model_name,
        'n_class': n_class
    }
    model = model_function(params)
    use_tta = False

    loaders = OrderedDict()
    if infer_csv:
        transforms = infer_tta_aug(image_size)
        if use_tta:
            for i, transform in enumerate(transforms):
                inferset = CsvDataset(
                    csv_file=infer_csv,
                    root=infer_root,
                    image_key=image_key,
                    label_key=label_key,
                    transform=transform,
                    mode='infer'
                )

                infer_loader = DataLoader(
                    dataset=inferset,
                    num_workers=4,
                    shuffle=False,
                    batch_size=32
                )

                loaders[f'infer_{i}'] = infer_loader

        else:
            inferset = CsvDataset(
                csv_file=infer_csv,
                root=infer_root,
                image_key=image_key,
                label_key=label_key,
                transform=transforms[0],
                mode='infer'
            )

            infer_loader = DataLoader(
                dataset=inferset,
                num_workers=4,
                shuffle=False,
                batch_size=32
            )

            loaders[f'{dataset}'] = infer_loader

    all_checkpoints = f"{log_dir}/checkpoints/best.pth"

    callbacks = [
        CheckpointCallback(resume=all_checkpoints),
        InferCallback(out_dir=log_dir, out_prefix="/predicts/")
    ]

    runner = ModelRunner()
    runner.infer(
        model,
        loaders,
        callbacks,
        verbose=True,
    )


if __name__ == "__main__":
    cli()