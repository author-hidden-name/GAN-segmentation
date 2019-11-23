import _init_path

import numpy as np
from matplotlib import pyplot as plt

import sys
import mxnet as mx
from mxnet import gluon
from functools import partial
from easydict import EasyDict as edict
from albumentations import (
    ShiftScaleRotate, Blur, IAAAdditiveGaussianNoise, GaussNoise, HorizontalFlip, OneOf, RandomCrop, PadIfNeeded, CenterCrop,
    RandomContrast, RandomBrightness, RGBShift
)

from mxnet.gluon.data.vision import transforms

from lib.model.deeplabv3plus import DeepLabV3Plus
from lib.data.augmentation.rgb_segmentation import RGBSegmentationAug
from lib.data.segmentation.ffhq_hair_segmentation import FFHQHairSegmentation
from lib.core.segmentation import SegmentationTrainer, SegmentationTester
from lib.utils.exps_utils import init_exp
from lib.utils.log import logger
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss


def add_exp_args(parser):
    parser.add_argument('--input-path', type=str, help='Path to dataset',
                        default='../../../../experiments/ffhq-hair/dataset')

    return parser


def init_model():
    model_cfg = edict()

    model_cfg.num_classes = 2
    model_cfg.crop_size = 480
    model_cfg.base_size = 512
    model_cfg.syncbn = True
    model_cfg.aux = True
    model_cfg.aux_weight = 0.5

    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'],
                             model_cfg.input_normalization['std']),
    ])

    if args.ngpus > 1 and model_cfg.syncbn:
        norm_layer = partial(mx.gluon.contrib.nn.SyncBatchNorm, num_devices=args.ngpus)
    else:
        norm_layer = mx.gluon.nn.BatchNorm

    model = DeepLabV3Plus(model_cfg.num_classes, backbone='resnet50',
                   norm_layer=norm_layer,
                   pretrained_base=True, ctx=mx.cpu(0),
                   base_size=model_cfg.base_size, crop_size=model_cfg.crop_size,
                   aux=model_cfg.aux)
    model.initialize(ctx=mx.cpu(0))

    return model, model_cfg


def train(args):

    print('start training..')

    model, model_cfg = init_model()

    args.input_normalization = model_cfg.input_normalization

    crop_size = model_cfg.crop_size

    lr = 0.005
    weight_decay = 2e-4
    momentum = 0.9
    num_epochs = 20

    train_augmentator = RGBSegmentationAug([
        HorizontalFlip(),
        ShiftScaleRotate(scale_limit=(-0.25, 0.25), rotate_limit=15, border_mode=0, p=1),
        PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=0),
        RandomCrop(crop_size, crop_size),
    ], ignore_class=-1)

    val_augmentator = RGBSegmentationAug([
        PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=0),
        CenterCrop(crop_size, crop_size)
    ], ignore_class=-1)

    trainset = FFHQHairSegmentation(
        args.input_path,
        scale_factor=0.5, train_epoch_len=10000,
        split='train', subdir='train_generated',
        transform=model_cfg.input_transform, augmentator=train_augmentator
    )

    valset = FFHQHairSegmentation(
        args.input_path,
        scale_factor=0.5,
        split='val', transform=model_cfg.input_transform, augmentator=val_augmentator)

    # optimizer and lr scheduling
    optimizer_params = {
        'mode': 'poly',
        'baselr': lr,
        'nepochs': num_epochs,
        'wd': weight_decay,
        'momentum': momentum
    }

    trainer = SegmentationTrainer(args, model, model_cfg, trainset, valset,
                                  optimizer_params,
                                  criterion=SegmentationLoss01,
                                  image_dump_interval=50)

    logger.info(f'Starting Epoch: {args.start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(args.start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


def test(args):
    args.batch_size = args.ngpus
    model, model_cfg = init_model()

    # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    scales = [1.0]
    optimal_thres = 1e-15

    tester = SegmentationTester(model, args,
                                num_classes=model_cfg.num_classes,
                                use_flip=True,
                                scales=scales, threshold=optimal_thres)

    if args.vizualization:
        testset = FFHQHairSegmentation(
            args.input_path, scale_factor=0.5,
            split='val', transform=model_cfg.input_transform, augmentator=None,
            return_path=args.vizualization,
        )

        tester.vizualizate(testset, args.viz_path, suffix='_rgb', save_gt=True)
    else:
        testset = FFHQHairSegmentation(
            args.input_path, scale_factor=0.5,
            split='val', transform=model_cfg.input_transform, augmentator=None,
            return_path=args.vizualization,
        )
        tester.test(testset)


class SegmentationLoss01(gluon.HybridBlock):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(SegmentationLoss01, self).__init__()

        self.aux = aux
        self.aux_weight = aux_weight
        self.fnl_loss = SoftmaxCrossEntropyLoss(axis=1, **kwargs)
        self.sm_loss = SoftmaxCrossEntropyLoss(axis=1, **kwargs)
        self.k_sum = 0
        self.ignore_label = ignore_label

    def hybrid_forward(self, F, pred1, pred2, label):
        tlabel = F.expand_dims(label, axis=1)

        assert tlabel.dtype == np.int32
        sample_weight = (tlabel != self.ignore_label).astype(np.float32)

        loss1 = self.fnl_loss(pred1, tlabel, sample_weight)
        loss2 = self.sm_loss(pred2, tlabel, sample_weight)

        self.k_sum = 0

        return loss1 + self.aux_weight * loss2


if __name__ == '__main__':
    args = init_exp(__file__, add_exp_args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


