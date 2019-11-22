from mxnet import gluon
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.fcn import _FCNHead
from gluoncv.model_zoo.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s, ResNetV1b, BottleneckV1b

def resnet50_v1s_lsun_pretrained(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1s-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], classes=1001, deep_stem=True, stem_width=64,
                      name_prefix='resnetv1s_', **kwargs)
    if pretrained:
        path_to_model = '/media/user/c192784f-e992-4c1e-9acb-711282eac532/snapshots/imagenet/lsun_fine_tune/' \
                        'params_resnet50_v1s_mixup_test1/0.1705-imagenet-resnet50_v1s-9-best-float32.params'
        model.load_parameters(path_to_model, ctx=ctx)
        # from ..data import ImageNet1kAttr
        # attrib = ImageNet1kAttr()
        # model.synset = attrib.synset
        # model.classes = attrib.classes
        # model.classes_long = attrib.classes_long
    return model

def resnet50_v1s_lsun2_pretrained(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1s-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], classes=1001, deep_stem=True, stem_width=64,
                      name_prefix='resnetv1s_', **kwargs)
    if pretrained:
        path_to_model = '/media/user/c192784f-e992-4c1e-9acb-711282eac532/snapshots/imagenet/lsun_fine_tune/' \
                        'params_resnet50_v1s_mixup_test2/0.2398-imagenet-resnet50_v1s-9-best.params'
        model.cast('float16')
        model.load_parameters(path_to_model, ctx=ctx)
        model.cast('float32')
        # from ..data import ImageNet1kAttr
        # attrib = ImageNet1kAttr()
        # model.synset = attrib.synset
        # model.classes = attrib.classes
        # model.classes_long = attrib.classes_long
    return model

class SegBaseModel(HybridBlock):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : Block
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, aux, backbone='resnet50', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
            elif backbone == 'resnet50_lsun':
                pretrained = resnet50_v1s_lsun_pretrained(pretrained=pretrained_base, dilated=True, **kwargs)
            elif backbone == 'resnet50_lsun2':
                pretrained = resnet50_v1s_lsun2_pretrained(pretrained=pretrained_base, dilated=True, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
            elif backbone == 'resnet152':
                pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class DeepLabV3Plus(SegBaseModel):
    r"""DeepLabV3+

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

        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation."
        arXiv preprint arXiv:1802.02611 (2018).

    """
    def __init__(self, nclass, backbone='resnet50', aux=True, ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(DeepLabV3Plus, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)

        with self.name_scope():
            self.head = _DeepLabHead(nclass, **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                self.auxlayer = _FCNHead(1024, nclass, **kwargs)
                self.auxlayer.initialize(ctx=ctx)
                self.auxlayer.collect_params().setattr('lr_mult', 10)

            self.skip_project = _SkipProject(32, **kwargs)
            self.skip_project.initialize(ctx=ctx)
            self.skip_project.collect_params().setattr('lr_mult', 10)

            self.aspp = _ASPP(2048, [12, 24, 36], **kwargs)
            self.aspp.initialize(ctx=ctx)
            self.aspp.collect_params().setattr('lr_mult', 10)

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        x = self.layer2(c1)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c1, c3, c4

    def hybrid_forward(self, F, x):
        c1, c3, c4 = self.base_forward(x)
        c1 = self.skip_project(c1)

        if hasattr(c1, 'shape'):
            _, _, c1_h, c1_w = c1.shape
            self._c1_h = c1_h
            self._c1_w = c1_w
        else:
            c1_h, c1_w = self._c1_h, self._c1_w
            assert c1_h is not None

        x = self.aspp(c4)
        x = F.contrib.BilinearResize2D(x,
                                       height=c1_h,
                                       width=c1_w)
        x = F.concat(x, c1, dim=1)

        x = self.head(x)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)

        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class _SkipProject(HybridBlock):
    def __init__(self, out_channels, norm_layer=nn.BatchNorm, norm_kwargs={}, **kwargs):
        super(_SkipProject, self).__init__()

        with self.name_scope():
            self.skip_project = nn.HybridSequential()
            self.skip_project.add(nn.Conv2D(out_channels, kernel_size=1, use_bias=False))
            self.skip_project.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            self.skip_project.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        return self.skip_project(x)


class _DeepLabHead(HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs={}, **kwargs):
        super(_DeepLabHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()

            self.block.add(SeparableConv(256, kernel_size=3, strides=1, dilation=1, depth_activation=True,
                                         in_filters=256 + 32, norm_kwargs=norm_kwargs,
                                         norm_layer=norm_layer, prefix='decoder_conv0_'))
            self.block.add(SeparableConv(256, kernel_size=3, strides=1, dilation=1, depth_activation=True,
                                         in_filters=256, norm_kwargs=norm_kwargs,
                                         norm_layer=norm_layer, prefix='decoder_conv1_'))

            self.block.add(nn.Conv2D(in_channels=256, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
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
    def __init__(self, in_channels, atrous_rates, norm_layer=nn.BatchNorm, norm_kwargs={}):
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


class SeparableConv(nn.HybridBlock):
    def __init__(self, out_filters, kernel_size, strides, dilation, depth_activation,
                 norm_layer, norm_kwargs={}, in_filters=None, prefix=None):
        super(SeparableConv, self).__init__(prefix=prefix)

        if in_filters is None:
            in_filters = out_filters

        self.depth_activation = depth_activation

        padding = compute_same_padding(kernel_size, dilation)
        with self.name_scope():
            self.depthwise_conv = nn.Conv2D(in_filters, in_channels=in_filters, groups=in_filters,
                                            dilation=dilation, use_bias=False,
                                            padding=padding, strides=strides,
                                            kernel_size=kernel_size, prefix='depthwise_')
            self.bn1 = norm_layer(in_channels=in_filters, prefix='depthwise_BN_', **norm_kwargs)
            self.pointwise_conv = nn.Conv2D(out_filters, kernel_size=1, use_bias=False, prefix='pointwise_')
            self.bn2 = norm_layer(in_channels=out_filters, prefix='pointwise_BN_', **norm_kwargs)

    def hybrid_forward(self, F, x):
        if not self.depth_activation:
            x = F.relu(x)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        if self.depth_activation:
            x = F.relu(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        if self.depth_activation:
            x = F.relu(x)
        return x


def compute_same_padding(kernel_size, dilation):
    # TODO: compute `same` padding for stride<=2 ?
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return pad_beg, pad_end
