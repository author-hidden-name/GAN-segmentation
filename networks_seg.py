import mxnet as mx
import numpy as np
from mxnet.gluon import nn
from networks_stylegan import UpSample


class DecoderResBlock(mx.gluon.HybridBlock):
    def __init__(self, conv_size, use_bn, use_sync, num_devices, in_c=None):
        super(DecoderResBlock, self).__init__()

        in_c = conv_size if in_c is None else in_c

        net_block = nn.HybridSequential()
        net_block.add(nn.Conv2D(conv_size, 3, strides=1, padding=1, use_bias=True,
                                in_channels=in_c))
        if use_bn:
            if not use_sync:
                net_block.add(nn.BatchNorm(in_channels=conv_size))
            else:
                net_block.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=conv_size,
                                                                num_devices=num_devices))
        net_block.add(nn.LeakyReLU(0.2))

        net_block.add(nn.Conv2D(conv_size, 3, strides=1, padding=1, use_bias=True,
                                in_channels=conv_size))
        if use_bn:
            if not use_sync:
                net_block.add(nn.BatchNorm(in_channels=conv_size))
            else:
                net_block.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=conv_size,
                                                                num_devices=num_devices))
        net_block.add(nn.LeakyReLU(0.2))
        self.base_layers = net_block

        if conv_size != in_c:
            shortcut = nn.HybridSequential()
            shortcut.add(nn.Conv2D(conv_size, 1, strides=1, padding=0, use_bias=True,
                                   in_channels=in_c))
            self.shortcut = shortcut
        else:
            self.shortcut = None

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self.base_layers(x)
        sc = x if self.shortcut is None else self.shortcut(x)
        return sc + y


class Decoder(mx.gluon.HybridBlock):

    def __init__(self, cfg, num_devices, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self._features = cfg['features']
        self._in_channels = cfg['in_channels']
        self._start_res = cfg['start_res']
        self._num_feats = len(self._in_channels)
        self._num_devices = num_devices

        use_bn = cfg['use_bn']
        use_sync = cfg['use_sync_bn']
        use_dropout = cfg['use_dropout']

        for i in range(self._start_res, self._num_feats):
            conv_size = self._features[i]
            in_c = self._in_channels[i]
            cvt_block = nn.HybridSequential()
            cvt_block.add(nn.Conv2D(conv_size, 3, strides=1, padding=1, use_bias=True, in_channels=in_c))
            if use_bn:
                if not use_sync:
                    cvt_block.add(nn.BatchNorm(in_channels=conv_size))
                else:
                    cvt_block.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=conv_size,
                                                                    num_devices=self._num_devices))

            cvt_block.add(nn.LeakyReLU(0.2))
            if use_dropout:
                cvt_block.add(nn.Dropout(0.5))
            setattr(self, 'cvt_block_{}'.format(i), cvt_block)

        for i in range(self._start_res, self._num_feats):
            conv_size = self._features[i+1]
            in_c = self._features[i]
            in_c = 2 * in_c if i > self._start_res else in_c
            if i < self._num_feats - 1:
                net_block = nn.HybridSequential()
                net_block.add(UpSample(scale=2, sample_type='nearest'))
                net_block.add(DecoderResBlock(conv_size, use_bn, use_sync, self._num_devices, in_c))
            else:
                net_block = nn.HybridSequential()
                net_block.add(nn.Conv2D(conv_size, 3, strides=1, padding=1, use_bias=True,
                                        in_channels=in_c))

            setattr(self, 'main_block_{}'.format(i), net_block)


    def hybrid_forward(self, F, *inputs, **kwargs):

        prev = None
        pred = None

        for i in range(self._start_res, self._num_feats):
            cvt_block = getattr(self, 'cvt_block_{}'.format(i))
            input_i = inputs[i]
            if cvt_block is not None:
                input_i = cvt_block(input_i)

            if i > self._start_res:
                input_i = F.concat(prev, input_i, dim=1)
            net_block = getattr(self, 'main_block_{}'.format(i))
            pred = net_block(input_i)
            prev = pred

        return pred