import random
import pickle
import numpy as np
from mxnet import cpu
import mxnet.ndarray as F
from lib.data.segmentation_base import SegmentationDataset
from pathlib import Path
import cv2


class LSUNBedroomsSegmentation(SegmentationDataset):
    def __init__(self, dataset_path, split='train', num_classes=150,
                 transform=None, augmentator=None, return_path=False,
                 decimation_factor=1, not_ignore_classes=None, max_samples=None,
                 scale_factor=1.0,
                 train_epoch_len=-1):
        super(LSUNBedroomsSegmentation, self).__init__()
        dataset_path = Path(dataset_path)
        self.train_epoch_len = train_epoch_len
        self.split = split
        self.scale_factor = scale_factor
        self._not_ignore_classes = not_ignore_classes

        _images_list = []

        _images_list = sorted((dataset_path / split).rglob('*.jpg'))
        print('number of images: {}'.format(len(_images_list)))
        if max_samples is not None:
            _images_list = random.sample(_images_list, min(len(_images_list), max_samples))

        if decimation_factor > 1:
            _images_list = [x for x in _images_list
                            if int(x.stem.split('_')[0]) % decimation_factor == 0]

        self.images = []
        self.masks = []

        self._num_class = num_classes
        self.NUM_CLASS = self._num_class

        for image_path in _images_list:
            image_path = str(image_path)
            mask_path = image_path.replace('img_', 'mask_').replace('.jpg', '.png')

            self.images.append(image_path)
            self.masks.append(mask_path)

        self.transform = transform
        self.augmentator = augmentator
        self.return_path = return_path
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        if 'train' in self.split and self.train_epoch_len > 0:
            index = random.randint(0, len(self.images) - 1)

        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.scale_factor != 1.:
            img = cv2.resize(img, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

        mask = cv2.imread(self.masks[index], cv2.IMREAD_UNCHANGED).astype('int32')
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self._not_ignore_classes is not None:
            test_elements = self._not_ignore_classes
            mask_not_ignore = np.isin(mask, test_elements)
            mask[np.logical_not(mask_not_ignore)] = -1

        mask = mask.astype(np.float32)

        if self.augmentator is not None:
            img, mask = self.augmentator(img, mask)

        img = F.array(img, cpu(0))
        mask = F.array(mask, cpu(0), dtype=np.int32)

        if self.transform is not None:
            img = self.transform(img)

        data = img
        if self.return_path:
            return data, mask, self.images[index]
        else:
            return data, mask

    def __len__(self):
        if 'train' in self.split and self.train_epoch_len > 0:
            return self.train_epoch_len
        else:
            return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        return None

    @property
    def num_class(self):
        return self._num_class
