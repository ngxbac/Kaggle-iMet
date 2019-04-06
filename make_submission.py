import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))


if __name__ == '__main__':
    os.makedirs("submission", exist_ok=True)
    threshold = 0.3
    preds = []
    for model_name in ["resnet34"]:
        for fold in [0]:
            # for checkpoint in range(5):
            pred = np.load(f"./logs_imet/{model_name}_bcef2focal/fold_{fold}/predicts/infer.logits.npy")
            pred = sigmoid(pred)
            preds.append(pred)

    print(len(preds))
    preds = np.asarray(preds)
    preds = np.mean(preds, axis=0)
    print(preds.shape)
    preds = preds > threshold

    prediction = []
    for i in range(preds.shape[0]):
        pred1 = np.argwhere(preds[i] == 1.0).reshape(-1).tolist()
        pred_str = " ".join(list(map(str, pred1)))
        prediction.append(pred_str)

    test_df = pd.read_csv("./data/sample_submission.csv")
    test_df.attribute_ids = prediction
    test_df.to_csv(f"./submission/resnet34.csv", index=False)
    print(test_df.head())
