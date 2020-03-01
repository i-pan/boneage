import pandas as pd
import numpy as np
import glob
import cv2
import os, os.path as osp

from tqdm import tqdm

def get_bbox(img, template): 
    '''
    img : original image
    template : cropped section of img
    '''
    w, h = template.shape[::-1]
    method = eval("cv2.TM_CCORR_NORMED")
    # Apply template matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    x1 = max_loc[0] ; y1 = max_loc[1] 
    x2 = x1 + w ; y2 = y1 + h 
    return (x1, y1, x2, y2) 

#split = 'train'
split = 'test'

print('SPLIT : {}'.format(split.upper()))

# Might need to change depending on your directory structure
ORIG_DIR = '../../../data/rsna/images/{}/original/'.format(split)
CROP_DIR = '../../../data/rsna/images/{}/cropped/'.format(split)

images = glob.glob(osp.join(CROP_DIR, '*.png'))

hand_bbox_coords = {
    'x1': [], 'y1': [], 'x2': [], 'y2': []
}

bad_images = []
for im in tqdm(images, total=len(images)):
    img = cv2.imread(im.replace(CROP_DIR, ORIG_DIR), 0)
    template = cv2.imread(im, 0)
    try:
        bbox = get_bbox(img, template)
        for ind, coord in enumerate(['x1','y1','x2','y2']):
            hand_bbox_coords[coord].append(bbox[ind])
    except:
        print ('{} failed ! '.format(im.split('/')[-1]))
        bad_images.append(im)


df = pd.DataFrame(hand_bbox_coords)
df['imgfile'] = [im.split('/')[-1] for im in images]

train_df = pd.read_csv('../../../data/rsna/{}.csv'.format(split))
train_df['imgfile'] = ['{}.png'.format(i) for i in train_df['id']]

df = df.merge(train_df, on='imgfile')
df.to_csv('../../../data/rsna/{}_with_coords.csv'.format(split), index=False)

# Test
img = cv2.imread(osp.join(ORIG_DIR, df['imgfile'].iloc[0]), 0)
crop = cv2.imread(osp.join(CROP_DIR, df['imgfile'].iloc[0]), 0)

img_crop = img[df['y1'].iloc[0]:df['y2'].iloc[0],df['x1'].iloc[0]:df['x2'].iloc[0]]
img_crop.shape == crop.shape
np.sum(img_crop == crop) == np.product(crop.shape)




