import random
import cv2
import numpy as np
from albumentations import Compose


class RGBSegmentationAug(object):
    def __init__(self, augmentations_list, ignore_class=0, temp_class=250):
        self.ignore_class = ignore_class
        self.temp_class = temp_class
        self.augmentation = Compose(augmentations_list, p=1.0)

    def __call__(self, image, mask):
        if self.ignore_class != 0:
            mask = mask.copy()
            mask[mask == 0] = self.temp_class
            if self.ignore_class != -1:
                mask[mask == self.ignore_class] = 0

        augmented = self.augmentation(image=image, mask=mask)
        aug_image, aug_mask = augmented['image'], augmented['mask']

        if self.ignore_class != 0:
            if self.ignore_class != -1:
                aug_mask[aug_mask == 0] = self.ignore_class
            aug_mask[aug_mask == self.temp_class] = 0

        return aug_image, aug_mask


class OriginalRGBSegmentationAug(object):
    def __init__(self, base_size, crop_size, mode):
        self.base_size = base_size
        self.crop_size = crop_size
        self.mode = mode

        assert mode in {'val', 'train'}

    def __call__(self, image, mask):
        if self.mode == 'val':
            outsize = self.crop_size
            short_size = outsize
            h, w = image.shape[:2]
            if w > h:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            else:
                ow = short_size
                oh = int(1.0 * h * ow / w)

            image = cv2.resize(image, (ow, oh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

            h, w = image.shape[:2]
            x1 = int(round((w - outsize) / 2.))
            y1 = int(round((h - outsize) / 2.))
            image = image[y1:y1 + outsize, x1:x1 + outsize, :]
            mask = mask[y1:y1 + outsize, x1:x1 + outsize]

            return image, mask

        elif self.mode == 'train':
            if random.random() < 0.5:
                image = image[:, ::-1, :]
                mask = mask[:, ::-1]

            crop_size = self.crop_size
            # random scale (short edge)
            short_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.6))
            h, w = image.shape[:2]
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)

            image = cv2.resize(image, (ow, oh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

            # pad crop
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                nimage = np.zeros((oh + padh, ow + padw, 3), dtype=image.dtype)
                nimage[:oh, :ow, :] = image
                image = nimage

                nmask = np.zeros((oh + padh, ow + padw), dtype=mask.dtype)
                nmask[:oh, :ow] = mask
                mask = nmask

            # random crop crop_size
            h, w = image.shape[:2]
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            image = image[y1:y1 + crop_size, x1:x1 + crop_size, :]
            mask = mask[y1:y1 + crop_size, x1:x1 + crop_size]

            # gaussian blur as in PSP
            if random.random() < 0.5:
                image = cv2.GaussianBlur(image, (0, 0), random.random() / 3)

            return image, mask
