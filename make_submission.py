import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


if __name__ == '__main__':
    preds = []
    for model_name in ["densenet121", "inception_v3", "resnet50"]:
        for fold in range(5):
            for checkpoint in range(5):
                pred = np.load(f"/media/ngxbac/DATA/logs_datahack/intel-scene/{model_name}_{fold}/predict_swa_2/predictions.infer_0.logits.{checkpoint}.npy")
                pred = softmax(pred, axis=1)
                preds.append(pred)

    print(len(preds))
    preds = np.asarray(preds)
    preds = np.mean(preds, axis=1)
    print(preds.shape)
    preds = np.argmax(preds, axis=1)

    submission = pd.read_csv("./data/test.csv")
    submission['label'] = preds
    submission.to_csv(f"kfold_5swa_blend.csv", index=False)