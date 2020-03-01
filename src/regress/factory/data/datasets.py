from torch.utils.data import Dataset, Sampler

import torch
import numpy as np
import time
import cv2

class XrayDataset(Dataset):

    def __init__(self, 
                 imgfiles, 
                 labels, 
                 male,
                 coords,
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor,
                 scale=228.):

        self.imgfiles = imgfiles
        self.labels = labels
        self.male = male
        self.coords = coords
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor
        self.scale = scale

    def __len__(self):
        return len(self.imgfiles)

    @staticmethod
    def crop_image(img, bbox):
        x1, y1, x2, y2 = bbox
        return img[y1:y2, x1:x2, :]

    def process_image(self, X):
        if type(self.coords) != type(None):
            bbox = [self.coords[k][i] for k in ['x1','y1','x2','y2']]
            X = self.crop_image(X, bbox)
        if self.pad: X = self.pad(X) 
        if self.resize: X = self.resize(image=X)['image'] 
        if self.transform: X = self.transform(image=X)['image'] 
        if self.crop: X = self.crop(image=X)['image']
        X = self.preprocessor.preprocess(X) 
        return X.transpose(2, 0, 1)

    def __getitem__(self, i):
        X = cv2.imread(self.imgfiles[i])
        while X is None:
            print('Failed to read {}'.format(self.imgfiles[i]))
            i = np.random.choice(range(len(self.imgfiles)))
            X = cv2.imread(self.imgfiles[i])
        X = self.process_image(X)
        y = self.labels[i] / self.scale if self.labels is not None else 0
        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        return X, y, torch.tensor(self.male[i]).long()


class BalancedSampler(Sampler):

    def __init__(self,
        dataset,
        strata):
    #
    # strata : dict 
    #    - key : stratum
    #    - value : indices belonging to stratum
    #
        super(BalancedSampler, self).__init__(data_source=dataset)
        self.strata = strata
        length = np.sum([len(v) for k,v in strata.items()])
        self.num_samples_per_stratum = int(length / len(strata.keys()))
        self.length = self.num_samples_per_stratum * len(strata.keys())

    def __iter__(self):
        # Equal number per stratum
        # Custom number per stratum will require additional code
        indices = [] 
        for k,v in self.strata.items():
            indices.append(np.random.choice(v, self.num_samples_per_stratum, replace=len(v) < self.num_samples_per_stratum))
        shuffled = np.random.permutation(np.concatenate(indices))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.length


