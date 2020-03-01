import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else MultilabelStratifiedKFold
    outer_spl = splitter(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_counter = 0
    for outer_train, outer_test in outer_spl.split(df) if stratified is None else outer_spl.split(df, df[stratified]):
        df.loc[outer_test, 'outer'] = outer_counter
        inner_spl = splitter(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_counter = 0
        df['inner{}'.format(outer_counter)] = 888
        inner_df = df[df['outer'] != outer_counter].reset_index(drop=True)
        # Determine which IDs should be assigned to inner train
        for inner_train, inner_valid in inner_spl.split(inner_df) if stratified is None else inner_spl.split(inner_df, inner_df[stratified]):
            inner_train_ids = inner_df.loc[inner_valid, id_col]
            df.loc[df[id_col].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
            inner_counter += 1
        outer_counter += 1
    return df


dset = 'cumc'
#dset = 'rsna'

LABELS = '../../../data/{}/train.csv'.format(dset)
df = pd.read_csv(LABELS)
df = df[df['boneage'] > 0].reset_index(drop=True)
df['strata'] = pd.cut(df['boneage'], bins=[0,24]+list(np.arange(12*3, 12*17, 12))+[228])
df = create_double_cv(df, 'imgfile', 10, 10, stratified=['strata', 'male'])

df.to_csv('../../../data/{}/train_with_splits.csv'.format(dset), index=False)