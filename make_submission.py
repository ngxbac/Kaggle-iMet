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
    for model_name in ["resnet34"]:
        for fold in [0, 1, 2, 3, 4, 5, 6]:
            # for checkpoint in range(5):
            pred = np.load(f"/media/ngxbac/DATA/logs_iwildcam/{model_name}_noempty/fold_{fold}/predicts/infer_0.logits.npy")
            pred = softmax(pred, axis=1)
            preds.append(pred)

    print(len(preds))
    preds = np.asarray(preds)
    preds = np.mean(preds, axis=0)
    print(preds.shape)
    preds = np.argmax(preds, axis=1)

    test_df = pd.read_csv("/media/ngxbac/Bac2/fgvc6/data/test.csv")
    submission = pd.DataFrame()
    submission['Id'] = test_df['file_name']
    submission['Id'] = submission['Id'].apply(lambda x: x.split(".")[0])
    submission['Predicted'] = preds
    submission.to_csv(f"./submission/no_empty.csv", index=False)
    submission.to_csv(f"./submission/no_empty.csv.gz", index=False, compression='gzip')