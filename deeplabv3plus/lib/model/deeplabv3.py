from mxnet import gluon
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.segbase import SegBaseModel
from gluoncv.model_zoo.fcn import _FCNHead


class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxilary loss.


    Reference:

        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).

    """
    def __init__(self, nclass, backbone='resnet50', aux=True, ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)

        with self.name_scope():
            self.head = _DeepLabHead(nclass, **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                self.auxlayer = _FCNHead(1024, nclass, **kwargs)
                self.auxlayer.initialize(ctx=ctx)
                self.auxlayer.collect_params().setattr('lr_mult', 10)

    def hybrid_forward(self, F, x):
        c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class _DeepLabHead(HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs={}, **kwargs):
        super(_DeepLabHead, self).__init__()
        with self.name_scope():
            self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer,
                              norm_kwargs=norm_kwargs, **kwargs)
            self.block = nn.HybridSequential()
            self.block.add(nn.Conv2D(in_channels=256, channels=256,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=256, **norm_kwargs))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=256, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.aspp(x)
        return self.block(x)


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=3, padding=atrous_rate,
                            dilation=atrous_rate, use_bias=False))
        block.add(norm_layer(in_channels=out_channels, **norm_kwargs))
        block.add(nn.Activation('relu'))
    return block

class _AsppPooling(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):
        super(_AsppPooling, self).__init__()
        self._out_h = None
        self._out_w = None
        self.gap = nn.HybridSequential()
        with self.gap.name_scope():
            self.gap.add(nn.GlobalAvgPool2D())
            self.gap.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                   kernel_size=1, use_bias=False))
            self.gap.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            self.gap.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        if hasattr(x, 'shape'):
            _, _, h, w = x.shape
            self._out_h = h
            self._out_w = w
        else:
            h, w = self._out_h, self._out_w
            assert h is not None

        pool = self.gap(x)
        return F.contrib.BilinearResize2D(pool, height=h, width=w)

class _ASPP(nn.HybridBlock):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        b0 = nn.HybridSequential()
        with b0.name_scope():
            b0.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                             kernel_size=1, use_bias=False))
            b0.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            b0.add(nn.Activation("relu"))

        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer,
                          norm_kwargs=norm_kwargs)

        self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
        with self.concurent.name_scope():
            self.concurent.add(b0)
            self.concurent.add(b1)
            self.concurent.add(b2)
            self.concurent.add(b3)
            self.concurent.add(b4)

        self.project = nn.HybridSequential()
        with self.project.name_scope():
            self.project.add(nn.Conv2D(in_channels=5*out_channels, channels=out_channels,
                                       kernel_size=1, use_bias=False))
            self.project.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            self.project.add(nn.Activation("relu"))
            self.project.add(nn.Dropout(0.5))

    def hybrid_forward(self, F, x):
        return self.project(self.concurent(x))
