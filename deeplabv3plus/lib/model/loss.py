from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like
import pickle


class NormalizedFocalLossSoftmax(Loss):
    r"""SoftmaxCrossEntropyLoss with ignore labels

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    ignore_label : int, default -1
        The label to ignore.
    size_average : bool, default False
        Whether to re-scale loss with regard to ignored labels.
    """
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, gamma=2, eps=1e-10, **kwargs):
        super(NormalizedFocalLossSoftmax, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._eps = eps
        self._gamma = gamma
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""

        label = F.expand_dims(label, axis=1)
        softmaxout = F.softmax(pred, axis=1)

        t = label != self._ignore_label
        pt = F.pick(softmaxout, label, axis=1, keepdims=True)
        pt = F.where(label == self._ignore_label, F.ones_like(pt), pt)
        beta = ((1 - pt) ** self._gamma)

        t_sum = F.sum(t, axis=(-2, -1), keepdims=True)
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        beta = F.broadcast_mul(beta, mult)
        self._k_sum = 0.9 * self._k_sum + 0.1 * mult.asnumpy().mean()

        loss = -beta * F.log(F.minimum(pt + self._eps, 1))

        if self._size_average:
            tsum = F.sum(t, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return loss


class AreaNormalizedFocalLossSoftmax(Loss):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, gamma=2, area_gamma=0.5, eps=1e-10,
                 debug=None, **kwargs):
        super(AreaNormalizedFocalLossSoftmax, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._eps = eps
        self._gamma = gamma
        self._area_gamma = area_gamma
        self._k_sum = 0

        self._debug = debug
        self._debug_count = 0

    def hybrid_forward(self, F, pred, label):
        label, area_weights = F.split(label, axis=1, num_outputs=2)
        softmaxout = F.softmax(pred, axis=1)

        t = label != self._ignore_label
        pt = F.pick(softmaxout, label, axis=1, keepdims=True)
        pt = F.where(label == self._ignore_label, F.ones_like(pt), pt)

        area_weights = area_weights ** self._area_gamma
        beta = ((1 - pt) ** self._gamma)

        if self._debug is not None:
            dparams = (label.asnumpy(), area_weights.asnumpy(), beta.asnumpy())
            self._dump_debug_params(dparams)

        beta = beta * area_weights

        t_sum = F.sum(t, axis=(-2, -1), keepdims=True)
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        beta = F.broadcast_mul(beta, mult)
        self._k_sum = 0.9 * self._k_sum + 0.1 * mult.asnumpy().mean()

        loss = -beta * F.log(F.minimum(pt + self._eps, 1))

        if self._size_average:
            tsum = F.sum(t, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return loss

    def _dump_debug_params(self, x):
        if self._debug_count % 25 == 0:
            output_path = f'{self._debug}/{self._debug_count:04d}.pickle'
            with open(output_path, 'wb') as f:
                pickle.dump(x, f)
        self._debug_count += 1


class NormalizedFocalLossSigmoid(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-12, size_average=True, scale=1.0,
                 normalize=False, **kwargs):
        super(NormalizedFocalLossSigmoid, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._normalize = normalize
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)

        t = F.ones_like(one_hot)
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        if self._normalize:
            t_sum = F.sum(t, axis=(-2, -1), keepdims=True)
            beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
            mult = t_sum / (beta_sum + self._eps)
            beta = F.broadcast_mul(beta, mult)

            self._k_sum = 0.9 * self._k_sum + 0.1 * mult.asnumpy().mean()

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != -1

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            tsum = F.sum(sample_weight, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)

        t = label != -1
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != -1

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            tsum = F.sum(label == 1, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss


class SoftmaxCrossEntropyLoss(Loss):
    r"""SoftmaxCrossEntropyLoss with ignore labels

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    ignore_label : int, default -1
        The label to ignore.
    size_average : bool, default False
        Whether to re-scale loss with regard to ignored labels.
    """
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, grad_scale=1.0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._grad_scale = grad_scale

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null',
            grad_scale=self._grad_scale,
        )
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
