import cv2
import numpy as np
import copy
import os, os.path as osp

from scipy.ndimage.interpolation import zoom

from mmdet.datasets.registry import PIPELINES

@PIPELINES.register_module
class LoadImage(object):

    def __init__(self, to_float32=True, concat=False):
        self.to_float32 = to_float32
        self.concat = concat

    def __call__(self, results):

        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = cv2.imread(filename)

        if self.concat:
            # MAKE A COPY
            # Or else you will screw things up when concat_left=True
            concat_results = copy.deepcopy(results)
            # Can still use results here
            negative = osp.join(results['img_prefix'],
                                results['img_info']['negative'])
            empty_im = pydicom.dcmread(negative).pixel_array
            empty_im = self.to_rgb(empty_im)
            # Make sure that empty_im height is equal to 
            # img height
            if empty_im.shape[0] != img.shape[0]:
                empty_im = self.resize(empty_im, img.shape[0])
            concat_left = np.random.binomial(1, 0.5)
            if concat_left:
                # If concatenating empty image on left, need to 
                # add width of empty_im to X coord of bounding boxes
                img = np.hstack((empty_im, img))
                # Must edit concat_results here
                concat_results['ann_info']['bboxes'][:,[0,2]] = results['ann_info']['bboxes'][:,[0,2]] \
                                                                + empty_im.shape[1]
            else:
                img = np.hstack((img, empty_im))

        if self.to_float32:
            img = img.astype(np.float32)


        # There may be a more elegant way of doing this
        # but I think this is fullproof 
        if self.concat:
            concat_results['filename'] = filename
            concat_results['img'] = img
            concat_results['img_shape'] = img.shape
            concat_results['ori_shape'] = img.shape
            return concat_results
        else:
            results['filename'] = filename
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

    @staticmethod
    def resize(im, height):
        ratio = height / im.shape[0]
        im = zoom(im, [ratio, ratio, 1.], order=1, prefilter=False)
        return im


@PIPELINES.register_module
class Preprocess(object):
    """
    Preprocessing module.
    """

    def __init__(self, mean, sdev, image_range, input_range):
        self.mean = np.array(mean, dtype=np.float32)
        self.sdev = np.array(sdev, dtype=np.float32)
        self.image_range = image_range
        self.input_range = input_range

    def __call__(self, results):
        results['img'] = self.preprocess(results['img'])
        results['img_norm_cfg'] = dict(
            mean=self.mean, 
            sdev=self.sdev, 
            image_range=self.image_range,
            input_range=self.input_range)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, sdev={}, image_range={}, input_range={})'.format(
            self.mean, self.sdev, self.image_range, self.input_range)
        return repr_str


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

vanilla_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=0.0,
        rotate_limit=90,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.5),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, 
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
]

def train_pipeline(resize_to, mean, sdev, image_range, input_range, augmentations=None, size_divisor=32, concat=True, mode='train'):
    img_norm_cfg = {
        'mean':  mean,
        'sdev':  sdev,
        'image_range': image_range,        
        'input_range': input_range
    }
    if mode == 'train':
        # For training using concatenated image, double the width
        if concat:
            resize_to[1] = resize_to[1]*2
        pipeline_list = [
            dict(type='LoadImage', concat=concat),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=tuple(resize_to), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5)
        ]
        if augmentations:
                pipeline_list.append(
                dict(
                    type='Albu',
                    transforms=eval(augmentations),
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.0,
                        filter_lost_elements=True),
                    keymap={
                        'img': 'image',
                        'gt_masks': 'masks',
                        'gt_bboxes': 'bboxes'
                    },
                    update_pad_shape=False,
                    skip_img_without_anno=False)
            )
        pipeline_list.extend([
            dict(type='Preprocess', **img_norm_cfg),
            dict(type='Pad', size_divisor=size_divisor),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
        return pipeline_list
    elif mode == 'valid':
        return [
            dict(type='LoadImage'),
            dict(type='Resize', img_scale=tuple(resize_to), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Preprocess', **img_norm_cfg),
            dict(type='Pad', size_divisor=size_divisor),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]


def test_pipeline(resize_to, mean, sdev, image_range, input_range, size_divisor=32):
    img_norm_cfg = {
        'mean':  mean,
        'sdev':  sdev,
        'image_range': image_range,        
        'input_range': input_range
    }
    return [
    dict(type='LoadImage'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=tuple(resize_to),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Preprocess', **img_norm_cfg),
            dict(type='Pad', size_divisor=size_divisor),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
    ]
