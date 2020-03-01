import numpy as np
import cv2

from albumentations import Compose, Resize, RandomCrop
from albumentations import (
    OneOf, HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    IAAAdditiveGaussianNoise, IAAPerspective, IAASharpen,
    CLAHE, RandomBrightness, RandomGamma, LongestMaxSize,
    Blur, MotionBlur, RandomContrast, HueSaturationValue
)

def pad_to_ratio(array, ratio):
    # Default is ratio=1 aka pad to create square image
    ratio = float(ratio)
    # Given ratio, what should the height be given the width? 
    h, w = array.shape[:2]
    desired_h = int(w * ratio)
    # If the height should be greater than it is, then pad height
    if desired_h > h: 
        hdiff = int(desired_h - h) ; hdiff = int(hdiff / 2)
        pad_list = [(hdiff, desired_h-h-hdiff), (0,0), (0,0)]
    # If height should be smaller than it is, then pad width
    elif desired_h < h: 
        desired_w = int(h / ratio)
        wdiff = int(desired_w - w) ; wdiff = int(wdiff / 2)
        pad_list = [(0,0), (wdiff, desired_w-w-wdiff), (0,0)]
    elif desired_h == h: 
        return array 
    return np.pad(array, pad_list, 'constant', constant_values=np.min(array))

def resize(x, y=None):
    if y is None: y = x
    return Compose([
        Resize(x, y, always_apply=True, interpolation=cv2.INTER_CUBIC, p=1)
        ], p=1)

def resize_longest(x):
    return Compose([
        LongestMaxSize(x, always_apply=True, interpolation=cv2.INTER_CUBIC, p=1)
        ], p=1)

def crop(x, y=None):
    if y is None: y = x
    return Compose([
        RandomCrop(x, y, always_apply=True, p=1)
        ], p=1)

def soft_augmentation(p):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(rotate_limit=30, 
                         scale_limit=0.1,  
                         border_mode=cv2.BORDER_CONSTANT, 
                         value=[0,0,0],
                         p=0.5),
        OneOf(
            [
                RandomContrast(p=1),
                RandomBrightness(p=1),
                RandomGamma(p=1),
            ],
            p=0.5,
        ),
    ], p=p)

def vanilla_transform(p):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(rotate_limit=30, 
                         scale_limit=0.15,  
                         border_mode=cv2.BORDER_CONSTANT, 
                         value=[0,0,0],
                         p=0.5),
        IAAAdditiveGaussianNoise(p=0.2),
        IAAPerspective(p=0.5),
        OneOf(
            [
                CLAHE(p=1),
                RandomBrightness(p=1),
                RandomGamma(p=1),
            ],
            p=0.5,
        ),
        OneOf(
            [
                IAASharpen(p=1),
                Blur(blur_limit=3, p=1),
                MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        OneOf(
            [
                RandomContrast(p=1),
                HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ], p=p)

class Preprocessor(object):
    '''
    Object to deal with preprocessing.
    Easier than defining a function.
    '''
    def __init__(self, image_range, input_range, mean, sdev):
        self.image_range = image_range
        self.input_range = input_range
        self.mean = mean 
        self.sdev = sdev

    def preprocess(self, img): 
        ''' 
        Preprocess an input image. 
        '''
        # Assume image is RGB 
        # Unconvinced that RGB<>BGR matters for transfer learning ...
        img = img[..., ::-1].astype('float32')

        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])

        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])

        image_range = image_max - image_min
        model_range = model_max - model_min 

        img = (((img - image_min) * model_range) / image_range) + model_min 
        img[..., 0] -= self.mean[0] 
        img[..., 1] -= self.mean[1] 
        img[..., 2] -= self.mean[2] 
        img[..., 0] /= self.sdev[0] 
        img[..., 1] /= self.sdev[1] 
        img[..., 2] /= self.sdev[2] 

        return img
