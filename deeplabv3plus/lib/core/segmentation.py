import os
import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from mxboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import logging
import shutil
import cv2
import types

from gluoncv.data.batchify import Tuple, Stack, Pad, Append
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import SegEvalModel, MultiEvalModel
from gluoncv.loss import *
from gluoncv.utils.viz.segmentation import DeNormalize, _getvocpallete

from lib.utils.log import logger, TqdmToLogger
from lib.utils.viz import visualize_mask
from lib.utils.utils import split_and_load, save_checkpoint
from lib.utils.metrics.segmentation import SegmentationMetric, SegmentationMetricDetailed


class SegmentationTrainer(object):
    def __init__(self, args, model, model_cfg, trainset, valset, optimizer_params,
                 with_depth=False, image_dump_interval=200, criterion=MixSoftmaxCrossEntropyLoss):
        self.args = args
        self.model_cfg = model_cfg
        self.with_depth = with_depth

        if with_depth:
            batchify_fn = Tuple(Tuple(Stack(), Stack()), Stack())
        else:
            batchify_fn = Tuple(Stack(), Stack())

        self.trainset = trainset
        self.valset = valset
        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            batchify_fn=batchify_fn,
            num_workers=args.workers)

        self.val_data = gluon.data.DataLoader(
            valset, args.test_batch_size,
            batchify_fn=batchify_fn,
            last_batch='rollover', num_workers=args.workers)

        logger.info(model)
        model.cast(args.dtype)
        model.collect_params().reset_ctx(ctx=args.ctx)
        self.net = model
        self.evaluator = SegEvalModel(model)

        if args.weights is not None:
            if os.path.isfile(args.weights):
                model.load_parameters(args.weights, ctx=args.ctx)
            else:
                raise RuntimeError(f"=> no checkpoint found at '{args.weights}'")

        # create criterion
        self.criterion = criterion(model_cfg.aux, aux_weight=model_cfg.aux_weight)

        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        self.lr_scheduler = LRScheduler(mode=optimizer_params['mode'],
                                        baselr=optimizer_params['baselr'],
                                        niters=len(self.train_data)*optimizer_params['nepochs'],
                                        nepochs=optimizer_params['nepochs'])
        del optimizer_params['mode'], optimizer_params['baselr'], optimizer_params['nepochs']
        optimizer_params['lr_scheduler'] = self.lr_scheduler

        kv = mx.kv.create(args.kvstore)
        self.optimizer = gluon.Trainer(self.net.collect_params(), 'sgd',
                                       optimizer_params, kvstore=kv)
        # evaluation metrics
        self.metric = SegmentationMetric(trainset.NUM_CLASS)
        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        self.denormalizator = DeNormalize(args.input_normalization['mean'],
                                          args.input_normalization['std'])

        self.sw = None
        self.viz_pallete = _getvocpallete(trainset.NUM_CLASS)
        self.image_dump_interval = image_dump_interval

    def training(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriter(logdir=str(self.args.logs_path), flush_secs=5)

        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)
        self.metric.reset()
        train_loss = 0.0

        iter_per_epoch = len(self.train_data)

        for i, (data, target) in enumerate(tbar):
            global_step = iter_per_epoch * epoch + i
            data = split_and_load(data, ctx_list=self.args.ctx,
                                  batch_axis=0, even_split=False)
            target = split_and_load(target, ctx_list=self.args.ctx,
                                                batch_axis=0, even_split=False)
            with autograd.record(True):
                if self.with_depth:
                    outputs = [self.net(*X) for X in data]
                else:
                    outputs = [self.net(X) for X in data]

                losses = [self.criterion(*X, Y)
                          for X, Y in zip(outputs, target)]

                autograd.backward(losses)

            self.optimizer.step(self.args.batch_size)

            batch_loss = sum(loss.asnumpy()[0] for loss in losses) / len(losses)
            train_loss += batch_loss

            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                image_blob = data[0][0][0] if self.with_depth else data[0][0]
                image = self.denormalizator(image_blob.as_in_context(mx.cpu(0))).asnumpy() * 255

                gt_mask = target[0][0].asnumpy() + self.trainset.pred_offset
                predicted_mask = mx.nd.squeeze(mx.nd.argmax(outputs[0][0][0], 0)).asnumpy() + self.trainset.pred_offset

                gt_mask = visualize_mask(gt_mask.astype(np.int32), self.trainset.NUM_CLASS + 1)
                predicted_mask = visualize_mask(predicted_mask.astype(np.int32), self.trainset.NUM_CLASS + 1)

                image = image.transpose((1, 2, 0))
                if gt_mask.shape[:2] == image.shape[:2]:
                    result = np.hstack((image, gt_mask, predicted_mask)).transpose((2, 0, 1)).astype(np.uint8)
                    self.sw.add_image('Images/input_image', result, global_step=global_step)
                else:
                    self.sw.add_image('Images/input_image',
                                      image.transpose((2, 0, 1)).astype(np.uint8), global_step=global_step)
                    result = np.hstack((gt_mask, predicted_mask)).transpose((2, 0, 1)).astype(np.uint8)
                    self.sw.add_image('Images/predicted', result, global_step=global_step)

            self.sw.add_scalar(tag='Loss/ce',
                               value={'batch': batch_loss, 'epoch_avg': train_loss/(i+1)},
                               global_step=global_step)
            self.sw.add_scalar(tag='learning_rate', value=self.lr_scheduler.learning_rate,
                               global_step=global_step)

            if hasattr(self.criterion, 'k_sum'):
                self.sw.add_scalar(tag='nfl_mult', value=self.criterion.k_sum,
                                   global_step=global_step)

            tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.3f}')
            mx.nd.waitall()
            self.net.hybridize()

        save_checkpoint(self.net, self.args, epoch=None)

    def validation(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriter(logdir=str(self.args.logs_path), flush_secs=5)

        self.metric.reset()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100)

        for i, (data, target) in enumerate(tbar):
            data = split_and_load(data, ctx_list=self.args.ctx,
                                  batch_axis=0, even_split=False)
            if self.with_depth:
                outputs = [self.net(*X)[0] for X in data]
            else:
                outputs = [self.net(X)[0] for X in data]

            targets = mx.gluon.utils.split_and_load(target, self.args.ctx, even_split=False)
            self.metric.update(targets, outputs)

            names, values = self.metric.get()
            result_str = ', '.join([f'{name}: {value:4f}' for name, value in zip(names, values)])
            tbar.set_description(f'Epoch {epoch}, validation {result_str}')

        names, values = self.metric.get()
        result_str = ', '.join([f'{name}: {value:4f}' for name, value in zip(names, values)])
        tbar.set_description(f'Epoch {epoch}, validation {result_str}')
        logging.info(result_str)

        for name, value in zip(names, values):
            self.sw.add_scalar(tag=f'Metrics/{name}', value={'val': value}, global_step=epoch)


class SegmentationTester(object):
    def __init__(self, model, args, num_classes, use_flip, scales, skip_bg=True,
                 custom_evaluator=None, use_prob_avg=False, class_names=None, threshold=0.5):
        self.args = args

        if class_names is None:
            class_names = [f'cls-{i}' for i in range(num_classes)]

        self.metric_orig = SegmentationMetric(num_classes, skip_bg=skip_bg, threshold=threshold)
        self.metric = SegmentationMetricDetailed(num_classes, class_names, full_output=False,
                                                 compute_auc=False, skip_bg=skip_bg, threshold=threshold)
        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO, mininterval=1)

        model.cast(args.dtype)
        model.collect_params().reset_ctx(ctx=args.ctx)
        model.load_parameters(args.weights, ctx=args.ctx)

        if custom_evaluator is not None:
            self.evaluator = custom_evaluator(model, num_classes, ctx_list=args.ctx,
                                              flip=use_flip, scales=scales)
        else:
            self.evaluator = MultiEvalModel(model, num_classes, ctx_list=args.ctx,
                                            flip=use_flip, scales=scales)
            if use_prob_avg:
                self.evaluator.flip_inference = types.MethodType(prob_avg_flip_inference, self.evaluator)

        logger.info(f"\nLoaded model weights from file: {args.weights}\n")

    def test(self, testset):
        test_data = gluon.data.DataLoader(
            testset, self.args.batch_size,
            shuffle=False, last_batch='keep',
            num_workers=self.args.workers)

        self.metric.reset()
        self.metric_orig.reset()
        tbar = tqdm(test_data, file=self.tqdm_out, ncols=100)
        for i, (data, dsts) in enumerate(tbar):
            predicts = self.evaluator.parallel_forward(data)
            if len(self.args.ctx) == 1:
                predicts = [predicts[0]]
                dsts = [dsts]
            else:
                predicts = [pred[0] for pred in predicts]
            targets = [target.as_in_context(predicts[0].context) \
                       for target in dsts]

            predicts = [mx.nd.softmax(p, axis=1) for p in predicts]

            self.metric.update(targets, predicts)
            self.metric_orig.update(targets, predicts)

            names, values = self.metric_orig.get()
            metrics_map = {}
            for name, value in zip(names, values):
                metrics_map[name] = value

            tbar.set_description(f'accuracy: {metrics_map["pixAcc"]:.3f}, mean-iou: {metrics_map["mIoU"]:.3f}')

        print('----- new metric ------')
        names, values = self.metric.get()
        for name, value in zip(names, values):
            logger.info(f'{name}: {value:.5%}')

        print('----- original metric ------')
        names, values = self.metric_orig.get()
        for name, value in zip(names, values):
            logger.info(f'{name}: {value:.5%}')


    def vizualizate(self, testset, output_path, suffix='', save_gt=True):
        batchify_fn = Tuple(Stack(), Stack(), Identity)

        test_data = gluon.data.DataLoader(
            testset, self.args.batch_size,
            shuffle=False, last_batch='keep',
            num_workers=self.args.workers,
            batchify_fn=batchify_fn
        )

        tbar = tqdm(test_data, file=self.tqdm_out, ncols=100)
        for i, (data, dsts, paths) in enumerate(tbar):
            predicts =  self.evaluator.parallel_forward(data)
            for predict, gt_mask, im_path in zip(predicts, dsts, paths):
                im_path = Path(im_path)

                predict = predict[0].asnumpy()
                predict = np.squeeze(np.argmax(predict, 1)) + testset.pred_offset
                predicted_mask = visualize_mask(predict.astype(np.int32), testset.NUM_CLASS + 1)

                image_dst_parent = output_path / im_path.parent.stem
                if not image_dst_parent.exists():
                    image_dst_parent.mkdir(parents=True)

                im_dst = image_dst_parent / (im_path.stem + '_image.jpg')
                if save_gt:
                    gt_mask = gt_mask.asnumpy()

                    gt_mask = visualize_mask((gt_mask + testset.pred_offset).astype(np.int32), testset.NUM_CLASS + 1)
                    gt_dst = image_dst_parent / (im_dst.stem + f'_gt.jpg')
                    cv2.imwrite(str(gt_dst), gt_mask)

                predicted_dst = image_dst_parent / (im_dst.stem + f'_predicted{suffix}.jpg')

                shutil.copy(str(im_path), str(im_dst))
                cv2.imwrite(str(predicted_dst), predicted_mask)

                tbar.set_description(im_path.name)


def prob_avg_flip_inference(self, image):
    from gluoncv.model_zoo.segbase import _flip_image, NDArray

    assert (isinstance(image, NDArray))
    output = self.evalmodule(image)
    output = mx.nd.softmax(output, axis=1)
    if self.flip:
        fimg = _flip_image(image)
        foutput = mx.nd.softmax(self.evalmodule(fimg), axis=1)
        output = 0.5 * (_flip_image(foutput) + output)
    return output


def Identity(x):
    return x
