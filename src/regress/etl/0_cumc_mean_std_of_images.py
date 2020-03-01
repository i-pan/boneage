import pandas as pd
import numpy as np
import os.path as osp
import cv2

from tqdm import tqdm

from albumentations import Compose, LongestMaxSize

IMAGE_DIR = '../../../data/cumc/images/cropped/'
coords = pd.read_csv('../../../data/cumc/train_with_splits.csv')

resize = Compose([LongestMaxSize(max_size=256)], p=1)


# Load in 1-by-1
pixel_mean = np.zeros(1)
pixel_std  = np.zeros(1)

k = 1

for rownum in tqdm(range(len(coords)), total=len(coords)):
    img = cv2.imread(osp.join(IMAGE_DIR, coords['imgfile'].iloc[rownum]), 0)
    #x1, y1, x2, y2 = coords[['x1','y1','x2','y2']].iloc[rownum]
    #img = img[y1:y2,x1:x2]
    img = resize(image=img)['image']
    img = img / 255.
    pixels = img.reshape((-1, 1)) # img.shape[2]
    for pixel in pixels:
        diff = pixel - pixel_mean
        pixel_mean = pixel_mean + diff / k
        pixel_std = pixel_std + diff * (pixel - pixel_mean)
        k += 1

pixel_std = np.sqrt(pixel_std / (k - 2))

pixel_mean
pixel_std
#


# Load in all at once
def get_pixels(i):
    img = cv2.imread(osp.join(IMAGE_DIR, coords['imgfile'].iloc[i]), 0)
    if img is None:
        return
    #x1, y1, x2, y2 = coords[['x1','y1','x2','y2']].iloc[i]
    #img = img[y1:y2,x1:x2]
    img = resize(image=img)['image']
    img = img / 255.
    return img.reshape(-1)

pixels = [get_pixels(rownum) for rownum in tqdm(range(len(coords)), total=len(coords))]
pixels = [_ for _ in pixels if type(_) != type(None)]
pixels = np.concatenate(pixels)

np.mean(pixels) # 0.2023
np.std(pixels) # 0.1930
#

