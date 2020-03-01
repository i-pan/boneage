import pandas as pd
import numpy as np
import pickle
import glob

pkl_files = np.sort(glob.glob('../test-predictions/ensemble/*/*.pkl'))

preds = []
for p in pkl_files:
    with open(p, 'rb') as f:
        preds.append(pickle.load(f))


dfs = [pd.DataFrame(_) for _ in preds]
test_df = pd.read_csv('../../data/rsna/test_with_coords.csv')

for ind, df in enumerate(dfs):
    print('{}: MAE={}'.format(pkl_files[ind].split('/')[-2], np.mean(np.abs(test_df['boneage']-df['y_pred']))))

y_pred_avg = np.array([df['y_pred'] for df in dfs]).mean(axis=0)

np.mean(np.abs(y_pred_avg-test_df['boneage']))

np.mean(np.abs((dfs[0]['y_pred']+dfs[1]['y_pred'])/2. - test_df['boneage']))