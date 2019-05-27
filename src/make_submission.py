import pandas as pd
import numpy as np

import torch
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
import glob
import click
from tqdm import *

from models import Finetune
from augmentation import *
from dataset import CassavaDataset


device = torch.device('cuda')


@click.group()
def cli():
    print("Making submission")


labels = sorted(['cmd', 'healthy', 'cgm', 'cbsd', 'cbb'])
i2c = {}
for i, label in enumerate(labels):
    i2c[i] = label


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = Ftorch.softmax(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


@cli.command()
def predict_all():
    test_csv = '/raid/bac/kaggle/cassava-disease/notebooks/csv/test.csv'
    log_dir = "/raid/bac/kaggle/logs/cassava-disease/finetune/"

    all_preds = []
    for model_name in ['resnet50', 'se_resnet50', 'densenet121']:

        test_augs = [valid_aug(320)]

        one_model_kfold = []
        for fold in range(5):
            model = Finetune(
                model_name=model_name,
                num_classes=5,
            )

            all_checkpoints = glob.glob(f"{log_dir}/{model_name}/fold_{fold}/checkpoints/stage2*")
            for checkpoint in all_checkpoints:
                checkpoint = torch.load(checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)

                for tta in test_augs:
                    # Dataset
                    dataset = CassavaDataset(
                        df=test_csv,
                        root='/',
                        transform=tta,
                        mode='infer'
                    )

                    loader = DataLoader(
                        dataset=dataset,
                        batch_size=128,
                        shuffle=False,
                        num_workers=4,
                    )

                    fold_pred = predict(model, loader)
                    one_model_kfold.append(fold_pred)

        one_model_kfold = np.stack(one_model_kfold, axis=0).mean(axis=0)
        np.save(f"./submission/{model_name}.csv", one_model_kfold)
        all_preds.append(one_model_kfold)
    all_preds = np.stack(all_preds, axis=0).mean(axis=0)

    all_preds = np.argmax(all_preds, axis=1)
    all_preds = [i2c[i] for i in all_preds]
    submission = dataset.df.copy()
    submission['Id'] = submission['files'].apply(lambda x: x.split("/")[-1])
    submission['Category'] = all_preds
    os.makedirs('submission', exist_ok=True)
    submission[['Id', 'Category']].to_csv(f'./submission/ensemble_kfold.csv', index=False)


@cli.command()
def from_numpy():
    test_csv = '/raid/bac/kaggle/cassava-disease/notebooks/csv/test.csv'
    submission = pd.read_csv(test_csv)
    all_preds = []
    for model_name in ['resnet50', 'se_resnet50', 'densenet121']:
        one_model_kfold = np.load(f"./submission/{model_name}.csv.npy")
        all_preds.append(one_model_kfold)
    all_preds = np.stack(all_preds, axis=0).mean(axis=0)

    all_preds = np.argmax(all_preds, axis=1)
    all_preds = [i2c[i] for i in all_preds]
    submission['Id'] = submission['files'].apply(lambda x: x.split("/")[-1])
    submission['Category'] = all_preds
    os.makedirs('submission', exist_ok=True)
    submission[['Id', 'Category']].to_csv(f'./submission/ensemble_kfold.csv', index=False)



def calibaraion(sub):
    prior_y0_train = {
        "Angry": 64 / 383,
        "Neutral": 63 / 383,
        "Happy": 63 / 383,
        "Sad": 61 / 383,
        "Fear": 46 / 383,
        "Surprise": 46 / 383,
        "Disgust": 40 / 383
    }

    arr_prior_y0_test = {
        "Angry": 99 / 653,
        "Neutral": 191 / 653,
        "Happy": 144 / 653,
        "Sad": 80 / 653,
        "Fear": 70 / 653,
        "Surprise": 29 / 653,
        "Disgust": 40 / 653
    }

    for prior in [prior_y0_train, arr_prior_y0_test]:
        total = 0
        for k in prior.keys():
            total += prior[k]

        assert abs(total - 1) <= 0.001, "Sum of prob {} is not equal to 1. Details: {}".format(total, prior)

    cols = ["Surprise", "Fear",
            "Disgust", "Happy",
            "Sad", "Angry",
            "Neutral"]

    labels = cols

    assert cols == list(sub.columns[1:]), "Sub filename is mandatory"

    def calibrate(prior_y0_train, prior_y0_test,
                  prior_y1_train, prior_y1_test,
                  predicted_prob_y0):
        predicted_prob_y1 = (1 - predicted_prob_y0)

        p_y0 = prior_y0_test * (predicted_prob_y0 / prior_y0_train)
        p_y1 = prior_y1_test * (predicted_prob_y1 / prior_y1_train)
        return p_y0 / (p_y0 + p_y1)  # normalization

    def calibrate_probs(prob, prior_train, prior_test):
        calibrated_prob = np.zeros_like(prob)

        for class_ in range(7):  # enumerate all classes
            label = labels[class_]

            prior_y0_train = prior_train[label]
            prior_y1_train = 1 - prior_y0_train

            prior_y0_test = prior_test[label]
            prior_y1_test = 1 - prior_y0_test

            for i in range(prob.shape[0]):  # enumerate every probability for a class
                predicted_prob_y0 = prob[i, class_]
                calibrated_prob_y0 = calibrate(
                    prior_y0_train, prior_y0_test,
                    prior_y1_train, prior_y1_test,
                    predicted_prob_y0)
                calibrated_prob[i, class_] = calibrated_prob_y0
        return calibrated_prob

    output_filenames = []
    prior_test = arr_prior_y0_test
    print("---------------------------------------")
    print("Calibrating to {} from {}".format(prior_test, prior_y0_train))
    val_prob = sub[cols].values
    calibrated_val_prob = calibrate_probs(val_prob, prior_train=prior_y0_train, prior_test=prior_test)

    calibrated_sub = sub[["video"]].copy()
    for class_ in range(7):  # enumerate all classes
        label = labels[class_]
        calibrated_sub[label] = calibrated_val_prob[:, class_]

    print(sub.describe())
    print(calibrated_sub.describe())

    print(sub.head(3))
    print(calibrated_sub.head(3))

    return calibrated_sub


if __name__ == '__main__':
    cli()