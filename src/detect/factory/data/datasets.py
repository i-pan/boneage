import os.path as osp

import time
import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.datasets.registry import DATASETS
from mmdet.datasets.pipelines import Compose

@DATASETS.register_module
class XrayDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 annotations,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 filter_empty=True,
                 test_mode=False):
    
        # MMDetection takes the filepath as input, then loads the 
        # annotations. Here, annotations are already loaded, then
        # fed as input. Makes it easier to deal with CV splits.

        self.img_infos = annotations
        self.neg_infos = [_ for _ in annotations if _['ann']['bboxes'].shape[0] == 0]
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.filter_empty = filter_empty
        self.test_mode = test_mode

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load proposals
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode and filter_empty:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            # Filter out empty images
            if img_info['ann']['bboxes'].shape[0] > 0:
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)


    def __getitem__(self, idx):
        start_time = time.time()
        if self.test_mode:
            return self.prepare_test_img(idx), time.time() - start_time
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
            
    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)