from mxnet import cpu
import mxnet.ndarray as F
from lib.data.segmentation_base import SegmentationDataset
from pathlib import Path
import cv2
import numpy as np


class ImagesDirectory(SegmentationDataset):
    def __init__(self, dataset_path, num_class, transform=None,
                 images_mask='*.png', depth_mask=None, pred_offset=1,
                 depth_k=None, depth_mean=None, depth_std=None):
        super(ImagesDirectory, self).__init__()
        dataset_path = Path(dataset_path)

        self.images = sorted(str(x) for x in dataset_path.glob(images_mask))
        self.depths = None
        if depth_mask is not None:
            self.depths = sorted(str(x) for x in dataset_path.glob(depth_mask))
            assert len(self.images) == len(self.depths), f'{len(self.images)} != {len(self.depths)}'

        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.depth_k = depth_k
        self.transform = transform
        self._pred_offset = pred_offset
        self._num_class = num_class

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.array(img, cpu(0))

        if self.transform is not None:
            img = self.transform(img)

        fake_target = -1 * np.ones(img.shape[:2]).astype(np.int32)
        if self.depths is not None:
            depth = cv2.imread(self.depths[index], cv2.IMREAD_UNCHANGED)

            depth[depth == 0] = self.depth_k / self.depth_mean
            depth = np.minimum(self.depth_k / (depth + 1), 1)
            depth = depth[np.newaxis, :, :]
            depth = F.array((depth - self.depth_mean) / self.depth_std, cpu(0))

            return (img, depth), fake_target, self.images[index]
        else:
            return img, fake_target, self.images[index]

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return self._pred_offset

    @property
    def classes(self):
        return None

    @property
    def num_class(self):
        return self._num_class
