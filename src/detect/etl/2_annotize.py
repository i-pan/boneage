import pandas as pd
import numpy as np
import pickle
import cv2
import re
import os.path as osp

from tqdm import tqdm

df = pd.read_csv('../../../data/rsna/train_coords_with_splits.csv')

# Now, we need to turn it into MMDetection format ...
inner_cols = [col for col in df.columns if re.search(r'inner[0-9]+', col)]
annotations = []
for rownum in tqdm(range(len(df)), total=len(df)):
    cv_splits = {col : df[col].iloc[rownum] for col in inner_cols}
    cv_splits['outer'] = df['outer'].iloc[rownum]
    tmp_img = cv2.imread(osp.join('../../../data/rsna/images/train/original/{}'.format(df['imgfile'].iloc[rownum])))
    tmp_dict = {
        'filename': df['imgfile'].iloc[rownum],
        'height': tmp_img.shape[0],
        'width':  tmp_img.shape[1],
        'ann': {
            'bboxes': np.asarray([df[['x1','y1','x2','y2']].iloc[rownum]]),
            'labels': np.asarray([1])
        },
        'img_class': 1,
        'cv_splits': cv_splits
    }
    annotations.append(tmp_dict)

with open('../../../data/rsna/train_bbox_annotations_with_splits.pkl', 'wb') as f:
    pickle.dump(annotations, f)
