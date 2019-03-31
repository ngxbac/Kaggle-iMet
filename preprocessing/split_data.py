import os
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold, GroupKFold


def main(args):
    train_df = pd.read_csv(args.train_csv)
    
    train_iwild = train_df[train_df["is_inat"] == False]
    train_inat = train_df[train_df["is_inat"] == True]
    
    locations = train_iwild['location'].values
    y = train_iwild['category_id'].values
    kf = GroupKFold(n_splits=args.n_splits)
    
    train_fold_iwild = []
    valid_fold_iwild = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(locations, y, locations)):

        train_fold = train_iwild.iloc[train_idx]
        valid_fold = train_iwild.iloc[valid_idx]
        
        train_fold_iwild.append(train_fold)
        valid_fold_iwild.append(valid_fold)
        
    train_fold_inat = []
    valid_fold_inat = []
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=2411)
    y = train_inat['category_id'].values
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_inat, y)):

        train_fold = train_inat.iloc[train_idx]
        valid_fold = train_inat.iloc[valid_idx]
        
        train_fold_inat.append(train_fold)
        valid_fold_inat.append(valid_fold)
        
    os.makedirs(args.output_folder, exist_ok=True)
    for i in range(args.n_splits):
        train_fold = pd.concat([
            train_fold_iwild[i],
            train_fold_inat[i]
        ], axis=0)
        
        valid_fold = pd.concat([
            valid_fold_iwild[i],
            valid_fold_inat[i]
        ], axis=0)
        
        train_fold.to_csv(f"{args.output_folder}/train_{i}.csv.gz", index=False, compression='gzip')
        valid_fold.to_csv(f"{args.output_folder}/valid_{i}.csv.gz", index=False, compression='gzip')


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
