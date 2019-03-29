import os
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold


def main(args):
    train_df = pd.read_csv(args.train_csv)
    y = train_df['category_id'].values

    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=2411)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, y)):
        train_fold = train_df.iloc[train_idx]
        valid_fold = train_df.iloc[valid_idx]

        os.makedirs(args.output_folder, exist_ok=True)

        train_fold.to_csv(f"{args.output_folder}/train_{fold}.csv.gz", index=False, compression='gzip')
        valid_fold.to_csv(f"{args.output_folder}/valid_{fold}.csv.gz", index=False, compression='gzip')


def parse_args():
    description = 'Resize image'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--train_csv', dest='train_csv',
                        help='Size of image',
                        default=224, type=str)

    parser.add_argument('--n_splits', dest='n_splits',
                        help='Size of image',
                        default=7, type=int)

    parser.add_argument('--output_folder', dest='output_folder',
                        help='Output folder',
                        default=None, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    print('Split KFOLD data')
    args = parse_args()
    main(args)
