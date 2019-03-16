import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


if __name__ == '__main__':
    train_df = pd.read_csv("./data/train-scene-classification/train.csv")
    y = train_df['label'].values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2411)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, y)):
        train_fold = train_df.iloc[train_idx]
        valid_fold = train_df.iloc[valid_idx]

        os.makedirs("./data/kfold/", exist_ok=True)

        train_fold.to_csv(f"./data/kfold/train_{fold}.csv", index=False)
        valid_fold.to_csv(f"./data/kfold/valid_{fold}.csv", index=False)