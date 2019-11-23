import numpy as np
import mxnet as mx
import threading
from sklearn.metrics import roc_auc_score, average_precision_score


def get_classes_names(cfg):
    class_map = {}
    if cfg['cls_type'] == 'hair_with_skin':
        class_map[0] = 'background'
        class_map[1] = 'hair'
        class_map[2] = 'skin'
    elif cfg['cls_type'] == 'face':
        class_map[0] = 'background'
        class_map[1] = 'hair'
        class_map[2] = 'brows'
        class_map[3] = 'eyes'
        class_map[4] = 'nose'
        class_map[5] = 'mouth'
        class_map[6] = 'ears'
        class_map[7] = 'skin'
        class_map[8] = 'beard'
    elif cfg['cls_type'] == 'glasses':
        class_map[0] = 'background'
        class_map[1] = 'glasses'
    elif cfg['cls_type'] == 'car':
        class_map[0] = 'background'
        class_map[1] = 'cars'
    elif cfg['cls_type'] == 'hair':
        class_map[0] = 'background'
        class_map[1] = 'hair'
    else:
        raise ValueError

    return class_map, len(class_map.keys())


def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))


class SegMetric(mx.metric.EvalMetric):
    """Computes segmentation metrics.
    """
    def __init__(self, num_classes, class_names=None, axis=1,
                 full_output=False, ignore_label=-1, skip_bg=True,
                 output_names=None, label_names=None):
        super(SegMetric, self).__init__(
            'SegMetric', axis=axis,
            output_names=output_names, label_names=label_names)

        self.axis = axis
        self.num_classes = num_classes
        self.full_output = full_output
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        self.class_names = class_names
        self._ignore_label = ignore_label
        self._skip_bg = skip_bg
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if isinstance(pred_label, mx.nd.NDArray):
                pred_label = pred_label.asnumpy()
            if isinstance(label, mx.nd.NDArray):
                label = label.asnumpy()

            if len(label.shape) > 3:
                label = np.squeeze(label, axis=1)

            if pred_label.shape != label.shape:
                pred_label = np.argmax(pred_label, axis=self.axis)

            if len(pred_label.shape) > 3:
                pred_label = np.squeeze(pred_label, axis=1)

            pred_label = pred_label.astype('int32')
            label = label.astype('int32')

            not_ignore_mask = np.logical_not(label == self._ignore_label)

            check_label_shapes(label, pred_label, shape=1)

            label = label[not_ignore_mask].copy()
            pred_label = pred_label[not_ignore_mask].copy()

            self.sum_corr += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

            self.update_tp_fp_fn(pred_label.flat, label.flat)

    def update_tp_fp_fn(self, predicted_mask, gt_mask):

        cls_range = self.num_classes - 1
        if not self._skip_bg:
            cls_range += 1

        for i in range(cls_range):
            clsid = i
            if self._skip_bg:
                clsid += 1
            masks_cls = predicted_mask == clsid
            masks_gt = gt_mask == clsid

            tp_masks = np.logical_and(masks_cls, masks_gt)
            fp_masks = np.logical_and(masks_cls, np.logical_not(masks_gt))
            fn_masks = np.logical_and(np.logical_not(masks_cls), masks_gt)

            tp = np.count_nonzero(tp_masks)
            fp = np.count_nonzero(fp_masks)
            fn = np.count_nonzero(fn_masks)

            self.sum_tp[i] += tp
            self.sum_fp[i] += fp
            self.sum_fn[i] += fn
            self.sum_num[i] += np.count_nonzero(masks_gt) + np.count_nonzero(masks_cls)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:
            cls_num = self.num_classes-1
            if not self._skip_bg:
                cls_num += 1

            self.sum_corr = 0
            self.num_inst = 0
            self.sum_tp = np.zeros((cls_num,), dtype=np.int32)
            self.sum_fp = np.zeros((cls_num,), dtype=np.int32)
            self.sum_fn = np.zeros((cls_num,), dtype=np.int32)
            self.sum_num = np.zeros((cls_num,), dtype=np.int32)
        except AttributeError:
            pass

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        res = []

        accuracy = 0.0
        if self.num_inst > 0:
            accuracy = float(self.sum_corr) / self.num_inst

        res.append(('accuracy', accuracy))

        cls_range = self.num_classes - 1
        offset = 1
        if not self._skip_bg:
            cls_range += 1
            offset = 0

        recall_cls = []
        precision_cls = []
        iou_cls = []

        for i in range(cls_range):
            # recall
            recall = 0.0
            if self.sum_tp[i] + self.sum_fn[i] > 0:
                recall = float(self.sum_tp[i]) / (self.sum_tp[i] + self.sum_fn[i])
            class_name = self.class_names[i + offset]
            if self.full_output:
                res.append(('{}-recall'.format(class_name), recall))

            # precision
            precision = 0.0
            if self.sum_tp[i] + self.sum_fp[i] > 0:
                precision = float(self.sum_tp[i]) / (self.sum_tp[i] + self.sum_fp[i])
            if self.full_output:
                res.append(('{}-precision'.format(class_name), precision))
            iou = 0.0
            if self.sum_tp[i] + self.sum_fp[i] + self.sum_fn[i] > 0:
                iou = float(self.sum_tp[i]) / (self.sum_tp[i] + self.sum_fp[i] + self.sum_fn[i])
            if self.full_output:
                res.append(('{}-iou'.format(class_name), iou))

            if self.sum_num[i] > 0:
                recall_cls.append(recall)
                precision_cls.append(precision)
                iou_cls.append(iou)

        mean_recall = np.mean(recall_cls)
        mean_precision = np.mean(precision_cls)
        mean_iou = np.mean(iou_cls)

        res.append(('mean-recall', mean_recall))
        res.append(('mean-precision', mean_precision))
        res.append(('mean-iou', mean_iou))

        names, values = zip(*res)
        names, values = list(names), list(values)

        return names, values


class SegMetricFull(mx.metric.EvalMetric):
    """Computes segmentation metrics.
    """
    def __init__(self, num_classes, class_names=None, axis=1,
                 skip_bg=True, full_output=False, ignore_label=-1, compute_auc=False,
                 output_names=None, label_names=None):
        super(SegMetricFull, self).__init__(
            'SegMetricFull', axis=axis,
            output_names=output_names, label_names=label_names)
        assert(num_classes > 1)
        self.axis = axis
        self.num_classes = num_classes
        self.full_output = full_output
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        self.class_names = class_names
        self.compute_auc = compute_auc
        self.ignore_label = ignore_label
        self.skip_bg = skip_bg
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """

        if not isinstance(labels, list):
            labels = [labels]

        if not isinstance(preds, list):
            preds = [preds]

        for label, pred_prob in zip(labels, preds):

            if isinstance(pred_prob, mx.nd.NDArray):
                pred_prob = pred_prob.asnumpy()
            if isinstance(label, mx.nd.NDArray):
                label = label.asnumpy()

            assert pred_prob.shape[self.axis] == self.num_classes

            if len(label.shape) > 3:
                label = np.squeeze(label, axis=1)

            if len(label.shape) < 3:
                label = np.expand_dims(label, axis=0)

            pred_label = np.argmax(pred_prob, axis=self.axis).astype(np.int32)
            if len(pred_label.shape) > 3:
                pred_label = np.squeeze(pred_label, axis=1)

            if len(pred_label.shape) < 3:
                pred_label = np.expand_dims(pred_label, axis=0)

            pred_label = pred_label.astype(np.int32)
            label = label.astype(np.int32)

            not_ignore_mask = np.logical_not(label == self.ignore_label)

            label = label[not_ignore_mask].copy()
            pred_label = pred_label[not_ignore_mask].copy()

            self.sum_corr += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

            self.update_tp_fp_fn(pred_label.flat, label.flat)

            if self.compute_auc:
                self.update_probs(pred_prob, label, not_ignore_mask)

    def update_probs(self, prob_mask, gt_mask_no_ignore, not_ignore_mask):

        cls_range = self.num_classes - 1
        if not self.skip_bg:
            cls_range += 1

        for i in range(cls_range):
            clsid = i
            if self.skip_bg:
                clsid += 1

            y_score = prob_mask[:, clsid, :, :]
            y_score = y_score[not_ignore_mask]
            y_score = y_score.reshape((-1,))

            y_true = (gt_mask_no_ignore == clsid).astype(np.int32)
            y_true = y_true.reshape((-1,))

            if self.stored_pred[i] is None:
                self.stored_pred[i] = [y_true, y_score]
            else:
                self.stored_pred[i][0] = np.concatenate((self.stored_pred[i][0], y_true), axis=0)
                self.stored_pred[i][1] = np.concatenate((self.stored_pred[i][1], y_score), axis=0)

    def update_tp_fp_fn(self, predicted_mask, gt_mask):

        cls_range = self.num_classes - 1
        if not self.skip_bg:
            cls_range += 1

        for i in range(cls_range):
            clsid = i
            if self.skip_bg:
                clsid += 1

            masks_cls = predicted_mask == clsid
            masks_gt = gt_mask == clsid

            tp_masks = np.logical_and(masks_cls, masks_gt)
            fp_masks = np.logical_and(masks_cls, np.logical_not(masks_gt))
            fn_masks = np.logical_and(np.logical_not(masks_cls), masks_gt)

            tp = np.count_nonzero(tp_masks)
            fp = np.count_nonzero(fp_masks)
            fn = np.count_nonzero(fn_masks)

            self.sum_tp[i] += tp
            self.sum_fp[i] += fp
            self.sum_fn[i] += fn

            dice = 0.0
            if 2 * tp + fp + fn > 0:
                dice = float(2 * tp) / (2 * tp + fp + fn)
            self.sum_dice[i] += dice
            self.num_dice[i] += 1

            self.sum_num[i] += np.count_nonzero(masks_gt) + np.count_nonzero(masks_cls)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:

            cls_range = self.num_classes - 1
            if not self.skip_bg:
                cls_range += 1

            self.sum_corr = 0
            self.num_inst = 0
            self.sum_tp = np.zeros((cls_range,), dtype=np.int32)
            self.sum_fp = np.zeros((cls_range,), dtype=np.int32)
            self.sum_fn = np.zeros((cls_range,), dtype=np.int32)
            self.sum_dice = np.zeros((cls_range,), dtype=np.float32)
            self.num_dice = np.zeros((cls_range,), dtype=np.int32)
            self.sum_num = np.zeros((cls_range,), dtype=np.int32)
            self.stored_pred = [None] * (cls_range)
        except AttributeError:
            pass

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        res = []

        accuracy = 0.0
        if self.num_inst > 0:
            accuracy = float(self.sum_corr) / self.num_inst

        res.append(('accuracy', accuracy))

        cls_range = self.num_classes - 1
        if not self.skip_bg:
            cls_range += 1

        recall_cls = []
        precision_cls = []
        iou_cls = []
        dice_cls = []
        macro_dice_cls = []
        ap_cls = []
        auc_cls = []

        mean_auc = 0.0
        mean_ap = 0.0

        for i in range(cls_range):

            cls_id = i
            if self.skip_bg:
                cls_id += 1

            # recall
            recall = 0.0
            if self.sum_tp[i] + self.sum_fn[i] > 0:
                recall = float(self.sum_tp[i]) / (self.sum_tp[i] + self.sum_fn[i])
            class_name = self.class_names[cls_id]
            if self.full_output:
                res.append(('{}-recall'.format(class_name), recall))

            # precision
            precision = 0.0
            if self.sum_tp[i] + self.sum_fp[i] > 0:
                precision = float(self.sum_tp[i]) / (self.sum_tp[i] + self.sum_fp[i])
            if self.full_output:
                res.append(('{}-precision'.format(class_name), precision))

            # iou
            iou = 0.0
            if self.sum_tp[i] + self.sum_fp[i] + self.sum_fn[i] > 0:
                iou = float(self.sum_tp[i]) / (self.sum_tp[i] + self.sum_fp[i] + self.sum_fn[i])
            if self.full_output:
                res.append(('{}-iou'.format(class_name), iou))

            # dice
            dice = 0.0
            if 2 * self.sum_tp[i] + self.sum_fp[i] + self.sum_fn[i] > 0:
                dice = float(2 * self.sum_tp[i]) / (2 * self.sum_tp[i] + self.sum_fp[i] + self.sum_fn[i])
            if self.full_output:
                res.append(('{}-dice'.format(class_name), dice))

            # macro-dice
            macro_dice = 0.0
            if self.num_dice[i] > 0:
                macro_dice = float(self.sum_dice[i]) / (self.num_dice[i])
            if self.full_output:
                res.append(('{}-macro-dice'.format(class_name), macro_dice))

            # average precision and auc
            auc_score = 0.0
            ap_score = 0.0
            if self.compute_auc and self.stored_pred[i] is not None:
                y_true, y_prob = self.stored_pred[i]
                if len(np.unique(y_true)) > 1:
                    auc_score = roc_auc_score(y_true, y_prob)
                    ap_score = average_precision_score(y_true, y_prob)

            if self.full_output:
                res.append(('{}-auc-score'.format(class_name), auc_score))
                res.append(('{}-ap-score'.format(class_name), ap_score))

            if self.sum_num[i] > 0:
                recall_cls.append(recall)
                precision_cls.append(precision)
                iou_cls.append(iou)
                dice_cls.append(dice)
                macro_dice_cls.append(macro_dice)

                if self.compute_auc:
                    auc_cls.append(auc_score)
                    ap_cls.append(ap_score)

        mean_recall = np.mean(recall_cls)
        mean_precision = np.mean(precision_cls)
        mean_iou = np.mean(iou_cls)
        mean_dice = np.mean(dice_cls)
        mean_macro_dice = np.mean(macro_dice_cls)
        if self.compute_auc:
            mean_auc = np.mean(auc_cls)
            mean_ap = np.mean(ap_cls)

        res.append(('mean-recall', mean_recall))
        res.append(('mean-precision', mean_precision))
        res.append(('mean-iou', mean_iou))
        res.append(('mean-dice', mean_dice))
        res.append(('mean-macro-dice', mean_macro_dice))
        if self.compute_auc:
            res.append(('mean-auc', mean_auc))
            res.append(('mean-ap', mean_ap))

            res.append(('100*(1-mean-auc)', 100*(1 - mean_auc)))
            res.append(('100*(1-mean-ap)', 100*(1 - mean_ap)))

        names, values = zip(*res)
        names, values = list(names), list(values)

        return names, values


class SegmentationMetric(mx.metric.EvalMetric):
    """Computes pixAcc and mIoU metric scroes
    """
    def __init__(self, nclass, skip_bg=True):
        super(SegmentationMetric, self).__init__('pixAcc & mIoU')
        self.nclass = nclass
        self.lock = threading.Lock()
        self._skip_bg = skip_bg
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union

        if isinstance(preds, mx.nd.NDArray):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        IoU = IoU[self.total_union > 0]
        if self._skip_bg:
            IoU = IoU[1:] # skip background class
        mIoU = IoU.mean()

        names = ('accuracy', 'mean-iou')
        values = (pixAcc, mIoU)
        names, values = list(names), list(values)

        return names, values

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    predict = np.argmax(output.asnumpy(), 1).astype('int64') + 1

    target = target.asnumpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    pixel_correct1 = np.sum(np.logical_and(predict == target, target > 0))

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = np.argmax(output.asnumpy(), 1).astype('int64') + 1
    target = target.asnumpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab > 0)
    pixel_correct = np.sum((imPred == imLab)*(imLab > 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)