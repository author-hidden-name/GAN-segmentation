import mxnet as mx
import numpy as np
from mxnet.gluon import nn


class StyleGeneratorBlock(mx.gluon.HybridBlock):
    def __init__(self, conv_size, latent_size, prefix, use_first_conv=True, use_wscale=False,
                 in_channels=None, use_blur=True, use_fused_upscale=False, fix_noise=True, **kwargs):
        super(StyleGeneratorBlock, self).__init__(**kwargs)

        in_channels = conv_size if in_channels is None else in_channels
        Conv2D_T = Conv2DW

        if use_fused_upscale:
            if use_first_conv:
                block0 = Conv2DTransposeW(conv_size, 4, strides=2, padding=1, use_bias=False, in_channels=in_channels,
                                          use_wscale=use_wscale, prefix='{}_deconv_1_'.format(prefix))
                self.block0 = block0
            else:
                self.block0 = None
            self.upsample = None
        else:
            if use_first_conv:
                block0 = Conv2D_T(conv_size, 3, strides=1, padding=1, use_bias=False, in_channels=in_channels,
                                  use_wscale=use_wscale, prefix='{}_conv_1_'.format(prefix))
                self.block0 = block0
                self.upsample = UpSample(scale=2, sample_type='nearest', prefix='{}_up_1_'.format(prefix))
            else:
                self.block0 = None
                self.upsample = None

        if use_blur:
            self.blur = Blur(conv_size, prefix='{}_blur_1_'.format(prefix))
        else:
            self.blur = None

        block1 = nn.HybridSequential()
        block1.add(AddNoise(conv_size, fix_noise=fix_noise, prefix='{}_noise_1_'.format(prefix)))
        block1.add(Bias(conv_size, prefix='{}_bias_1_'.format(prefix)))
        block1.add(nn.LeakyReLU(0.2, prefix='{}_a_1_'.format(prefix)))
        self.block1 = block1
        self.adain1 = AdaIN(conv_size, latent_size, use_wscale=use_wscale,
                            prefix='{}_adain_1_'.format(prefix))

        block2 = nn.HybridSequential()
        block2.add(Conv2D_T(conv_size, 3, strides=1, padding=1, use_bias=False, in_channels=conv_size,
                           use_wscale=use_wscale, prefix='{}_conv_2_'.format(prefix)))

        block2.add(AddNoise(conv_size, fix_noise=fix_noise, prefix='{}_noise_2_'.format(prefix)))
        block2.add(Bias(conv_size, prefix='{}_bias_2_'.format(prefix)))
        block2.add(nn.LeakyReLU(0.2, prefix='{}_a_2_'.format(prefix)))
        self.block2 = block2
        self.adain2 = AdaIN(conv_size, latent_size, use_wscale=use_wscale,
                            prefix='{}_adain_2_'.format(prefix))

    def hybrid_forward(self, F, x, w1, w2, *args, **kwargs):
        y = x

        if self.upsample is not None:
            y = self.upsample(y)

        if self.block0 is not None:
            y = self.block0(y)

        if self.blur is not None:
            y = self.blur(y)

        y = self.block1(y)
        y = self.adain1(y, w1)
        y = self.block2(y)
        y = self.adain2(y, w2)

        return y


class Generator(mx.gluon.HybridBlock):

    def __init__(self, config, **kwargs):
        super(Generator, self).__init__(prefix='', **kwargs)

        self.fmap_base = config['fmap_base']
        self.fmap_decay = config['fmap_decay']
        self.fmap_max = config['fmap_max']
        self.base_scale_x = config['base_scale_x']
        self.base_scale_y = config['base_scale_y']
        self.use_wscale = config['use_wscale']
        self.fix_noise = config['fix_noise']

        self.nc = config['channels']
        self.latent_size = config['latent_size']
        self.max_res_log2 = config['max_res_log2']

        with self.name_scope():
            tensor_shape = (1, self.num_features(2), self.base_scale_y, self.base_scale_x)
            self.constant_tensor = self.params.get('constant_tensor', shape=tensor_shape, init=mx.init.Normal(1),
                                                   allow_deferred_init=True)
            self.latent_avg = self.params.get('latent_avg', shape=(512,), init=mx.init.Constant(0),
                                              allow_deferred_init=True)
            self.truncation_psi = self.params.get('truncation_psi', shape=((self.max_res_log2-1)*2,), init=mx.init.Constant(1.0),
                                                  allow_deferred_init=True)

        setattr(self, 'mapping', self.build_mapping())

        for res_log2 in range(2, self.max_res_log2+1):
            setattr(self, 'net{}'.format(res_log2), self.build_block(res_log2))
        setattr(self, 'to_rgb{}'.format(self.max_res_log2), self.build_to_rgb(self.max_res_log2))

        self.upscale2x = self.build_upscale2x()

    def build_upscale2x(self):
        upscale2x = UpSample(scale=2, sample_type='nearest')
        return upscale2x

    def num_features(self, res_log2):
        fmaps = int(self.fmap_base / (2.0 ** ((res_log2 - 1) * self.fmap_decay)))
        return min(fmaps, self.fmap_max)

    def build_to_rgb(self, res_log2, use_bias=True):
        scale = 2 ** res_log2
        conv_size = self.num_features(res_log2)
        net_block = nn.HybridSequential()
        Conv2D_T = Conv2DW
        net_block.add(Conv2D_T(self.nc, 1, strides=1, padding=0, in_channels=conv_size, use_bias=use_bias,
                               use_wscale=self.use_wscale,
                               gain=1, prefix='{}_conv_to_rgb_'.format(scale)))
        return net_block

    def build_mapping(self):

        layers = nn.HybridSequential()
        layers.add(PixelNorm())

        for i in range(8):
            layers.add(DenseW(self.latent_size, in_units=self.latent_size, use_wscale=self.use_wscale, gain=np.sqrt(2),
                              lr_mult=0.01,
                              prefix='mp_dense_{}_'.format(i)))
            layers.add(nn.LeakyReLU(0.2, prefix='mp_a_{}_'.format(i)))

        return layers

    def build_block(self, res_log2):
        conv_size = self.num_features(res_log2)
        scale = 2 ** res_log2
        prefix = '{}'.format(scale)
        in_channels = self.num_features(res_log2 - 1) if res_log2 > 2 else conv_size

        if res_log2 == 2:
            net_block = StyleGeneratorBlock(conv_size, self.latent_size, prefix, use_first_conv=False,
                                            use_wscale=self.use_wscale, in_channels=in_channels,
                                            use_blur=False, fix_noise=self.fix_noise)
        else:
            net_block = StyleGeneratorBlock(conv_size, self.latent_size, prefix, use_first_conv=True,
                                            use_wscale=self.use_wscale, in_channels=in_channels,
                                            use_fused_upscale=res_log2 >= 7, use_blur=True, fix_noise=self.fix_noise)

        return net_block

    def lerp(self, F, coeff, latent_avg, w):
        coeff_ = F.reshape(coeff, (1, 1))
        coeff_rev = F.reshape(1 - coeff, (1, 1))
        # latent_avg * (1 - coeff) + w
        w = F.broadcast_add(F.broadcast_mul(latent_avg, coeff_rev), F.broadcast_mul(w, coeff_))
        return w

    def hybrid_forward(self, F, x, constant_tensor=None, latent_avg=None, truncation_psi=None):

        # block forward
        w = getattr(self, 'mapping')(x)

        truncation_psi = F.split(truncation_psi, num_outputs=self.max_res_log2*2-2, axis=0)
        latent_avg = F.reshape(latent_avg, (1, -1))

        xrsh = F.expand_dims(x, axis=2)
        xrsh = F.expand_dims(xrsh, axis=3)
        xrsh = F.repeat(xrsh, self.base_scale_y, axis=2)
        xrsh = F.repeat(xrsh, self.base_scale_x, axis=3)

        constant_tensor = F.broadcast_like(constant_tensor, xrsh, lhs_axes=(0,), rhs_axes=(0,))

        w1 = self.lerp(F, truncation_psi[0], latent_avg, w)
        w2 = self.lerp(F, truncation_psi[1], latent_avg, w)

        features = []
        y = getattr(self, 'net2')(constant_tensor, w1, w2)
        features.append(y)
        for res in range(3, self.max_res_log2 + 1):

            w1 = self.lerp(F, truncation_psi[2 * (res - 2)], latent_avg, w)
            w2 = self.lerp(F, truncation_psi[2 * (res - 2) + 1], latent_avg, w)

            y = getattr(self, 'net{}'.format(res))(y, w1, w2)
            features.append(y)

        to_rgb_y = getattr(self, 'to_rgb{}'.format(self.max_res_log2))
        y = to_rgb_y(y)

        return y, features


class Blur(mx.gluon.HybridBlock):
    def __init__(self, channels, filter_kernel=(1,2,1), prefix='', **kwargs):
        super(Blur, self).__init__(prefix=prefix, **kwargs)
        self.channels = channels
        self.pad = int((len(filter_kernel) - 1) // 2)
        self.kernel_size = len(filter_kernel)

        w_kernel = self.get_kernel(filter_kernel, self.channels)
        with self.name_scope():
            self.w_kernel = self.params.get_constant('w_kernel', w_kernel)

    def get_kernel(self, filter_kernel, channels):
        filter_kernel = mx.nd.array(filter_kernel, dtype=np.float32)

        filter_kernel_x = mx.nd.reshape(filter_kernel, (1, -1))
        filter_kernel_y = mx.nd.reshape(filter_kernel, (-1, 1))

        filter_kernel_xy = mx.nd.broadcast_mul(filter_kernel_x, filter_kernel_y)

        # make sure sum of values in gaussian kernel equals 1.
        filter_kernel_xy = filter_kernel_xy / mx.nd.sum(filter_kernel_xy)

        # reshape to 2d depthwise convolutional weight
        filter_kernel_xy = filter_kernel_xy.reshape((1, 1, len(filter_kernel), len(filter_kernel)))
        filter_kernel_xy = filter_kernel_xy.repeat(channels, axis=0)

        return filter_kernel_xy

    def hybrid_forward(self, F, x, w_kernel, *args, **kwargs):

        kernel_size = (self.kernel_size,)*2
        stride = (1, 1)
        pad = (self.pad,)*2
        x_mean = F.Convolution(x, weight=w_kernel, kernel=kernel_size, stride=stride, pad=pad,
                               num_filter=self.channels, num_group=self.channels, no_bias=True)

        return x_mean


class AdaIN(mx.gluon.HybridBlock):
    def __init__(self, channels, latent_size, use_wscale=False, prefix='', **kwargs):
        super(AdaIN, self).__init__(prefix=prefix, **kwargs)
        self.channels = channels
        self.latent_size = latent_size
        self.affine = DenseW(channels*2, in_units=latent_size, use_bias=True, gain=1,
                             use_wscale=use_wscale, flatten=False, prefix=prefix + 'dense_affine_')
        self.instance = nn.InstanceNorm(axis=1, center=False, scale=False, in_channels=channels,
                                        prefix=prefix + 'norm_')


    def hybrid_forward(self, F, x, w, *args, **kwargs):

        y = self.affine(w) # NZ --> N(2C)

        y = F.reshape(y, (0, 2, -1)) # N(2C) --> N2C

        ys, yb = F.split(y, num_outputs=2, axis=1, squeeze_axis=True)

        ys = F.reshape(ys, (0, 0, 1, 1))
        yb = F.reshape(yb, (0, 0, 1, 1))

        x_norm = self.instance(x)
        x_scaled = F.broadcast_plus(F.broadcast_mul(x_norm, ys + 1), yb)

        return x_scaled


class AddNoise(mx.gluon.HybridBlock):
    def __init__(self, channels, fix_noise=True, **kwargs):
        super(AddNoise, self).__init__(**kwargs)
        self.h, self.w = 0, 0
        self.batch_size = 0
        self.dtype = np.float32
        self.fix_noise = fix_noise

        if fix_noise:
            self._noise = None

        with self.name_scope():
            self.scale_factors = self.params.get('scale_factors', shape=(1, channels, 1, 1),
                                                 init=mx.init.Constant(0),
                                                 allow_deferred_init=True)

    def hybrid_forward(self, F, x, scale_factors, *args, **kwargs):

        if isinstance(x, mx.nd.NDArray):
            batch_size, _, h, w = x.shape
            self.h, self.w = h, w
            dtype = x.dtype
            self.dtype = dtype
            self.batch_size = batch_size
            if self.fix_noise and self._noise is None:
                self._noise = mx.nd.random.normal(0, 1, (batch_size, 1) + (h, w), dtype=dtype)
        else:
            batch_size, h, w = self.batch_size, self.h, self.w
            dtype = self.dtype

        if self.fix_noise:
            noise = self._noise
        else:
            noise = F.random.normal(0, 1, (batch_size, 1) + (h, w), dtype=dtype)

        noise_scaled = F.broadcast_mul(scale_factors, noise)

        y = F.broadcast_plus(x, noise_scaled)
        return y


class UpSample(mx.gluon.HybridBlock):
    def __init__(self, scale, sample_type, **kwargs):
        super(UpSample, self).__init__(**kwargs)
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x):
        return F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)


class Reshape(mx.gluon.HybridBlock):
    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.shape = shape

    def hybrid_forward(self, F, x):
        return F.reshape(x, self.shape)


class MinibatchStdLayer(mx.gluon.HybridBlock):
    def __init__(self, group_size, width, height, **kwargs):
        super(MinibatchStdLayer, self).__init__(**kwargs)
        self.group_size = group_size
        self.h = height
        self.w = width

    def hybrid_forward(self, F, x, *args, **kwargs):

        y = F.expand_dims(x, axis=0)                                # [1NCHW]   input shape
        y = F.reshape(y, shape=(self.group_size, -1, 0, 0, 0))      # [GMCHW]   split minibatch into M groups of size G.
        y = F.broadcast_sub(y, F.mean(y, axis=0, keepdims=True))    # [GMCHW]   subtract mean over group.
        y = F.mean(F.square(y), axis=0)                             # [MCHW]    calc variance over group.
        y = F.sqrt(y + 1e-8)                                        # [MCHW]    calc stddev over group.
        y = F.mean(y, axis=(1, 2, 3), keepdims=True)                # [M111]    take average over fmaps and pixels.

        y = F.tile(y, (self.group_size, 1, self.h, self.w))         # [N1HW]    replicate over group.

        return F.concat(x, y, dim=1)


def _infer_weight_shape(op_name, data_shape, kwargs):
    op = getattr(mx.symbol, op_name)
    sym = op(mx.symbol.var('data', shape=data_shape), **kwargs)
    return sym.infer_shape_partial()[0]


class _ConvW(nn.HybridBlock):
    def __init__(self, channels, kernel_size, strides, padding, dilation,
                 groups, layout, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 op_name='Convolution', adj=None, prefix=None, params=None,
                 use_wscale=False, gain=np.sqrt(2), lr_mult=1, fan_in=None):
        super(_ConvW, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            if isinstance(strides, mx.base.numeric_types):
                strides = (strides,)*len(kernel_size)
            if isinstance(padding, mx.base.numeric_types):
                padding = (padding,)*len(kernel_size)
            if isinstance(dilation, mx.base.numeric_types):
                dilation = (dilation,)*len(kernel_size)
            self._op_name = op_name
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout}
            if adj is not None:
                self._kwargs['adj'] = adj

            dshape = [0]*(len(kernel_size) + 2)
            dshape[layout.find('N')] = 1
            dshape[layout.find('C')] = in_channels
            wshapes = _infer_weight_shape(op_name, dshape, self._kwargs)
            self.weight = self.params.get('weight', shape=wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=wshapes[2],
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation+'_')
            else:
                self.act = None

            self.lr_mult = lr_mult

            if use_wscale:
                if fan_in is None:
                    fan_in = kernel_size[0] * kernel_size[1] * in_channels
                std = gain / np.sqrt(fan_in)  # He init
                self.std = self.params.get_constant('std', [std])
            else:
                self.std = None

    def hybrid_forward(self, F, x, weight, bias=None, std=None):
        if std is not None:
            weight = F.broadcast_mul(weight, std)
        weight = weight * self.lr_mult
        if bias is not None:
            bias = bias * self.lr_mult
        if bias is None:
            act = getattr(F, self._op_name)(x, weight, name='fwd', **self._kwargs)
        else:
            act = getattr(F, self._op_name)(x, weight, bias, name='fwd', **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def _alias(self):
        return 'conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)


class Conv2DW(_ConvW):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, mx.base.numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(Conv2DW, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class Conv2DTransposeW(_ConvW):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 output_padding=(0, 0), dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, mx.base.numeric_types):
            kernel_size = (kernel_size,)*2
        if isinstance(output_padding, mx.base.numeric_types):
            output_padding = (output_padding,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        assert len(output_padding) == 2, "output_padding must be a number or a list of 2 ints"
        super(Conv2DTransposeW, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer,
            bias_initializer, op_name='Deconvolution', adj=output_padding, **kwargs)
        self.outpad = output_padding


class DenseW(nn.HybridBlock):
    def __init__(self, units, activation=None, use_bias=True, flatten=True,
                 dtype='float32', weight_initializer=None, bias_initializer='zeros',
                 in_units=0, use_wscale=False, gain=np.sqrt(2), lr_mult=1., fan_in=None, **kwargs):
        super(DenseW, self).__init__(**kwargs)
        self._flatten = flatten
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)

            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            else:
                self.bias = None
            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation+'_')
            else:
                self.act = None

            self.lr_mult = lr_mult

            if use_wscale:
                if fan_in is None:
                    fan_in = in_units
                std = gain / np.sqrt(fan_in)  # He init
                self.std = self.params.get_constant('std', [std])
            else:
                self.std = None

    def hybrid_forward(self, F, x, weight, bias=None, std=None):
        if std is not None:
            weight = F.broadcast_mul(weight, std)
        weight = weight * self.lr_mult
        if bias is not None:
            bias = bias * self.lr_mult

        act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                               flatten=self._flatten, name='fwd')
        if self.act is not None:
            act = self.act(act)
        return act

    def __repr__(self):
        s = '{name}({layout}, {act})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        act=self.act if self.act else 'linear',
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


class Bias(nn.HybridBlock):
    def __init__(self, units, dtype='float32', bias_initializer='zeros', **kwargs):
        super(Bias, self).__init__(**kwargs)
        with self.name_scope():
            self._units = units
            self.bias = self.params.get('bias', shape=(1, units, 1, 1),
                                        init=bias_initializer, dtype=dtype,
                                        allow_deferred_init=True)

    def hybrid_forward(self, F, x, bias):
        y = F.broadcast_plus(x, bias)
        return y


class NormalWithL2Norm(mx.init.Initializer):
    def __init__(self, sigma=0.01):
        super(NormalWithL2Norm, self).__init__(sigma=sigma)
        self.sigma = sigma

    def _init_weight(self, _, arr):
        mx.random.normal(0, self.sigma, out=arr)
        arr[:] = arr / (arr.norm() + 1e-12)


class PixelNorm(mx.gluon.HybridBlock):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)
        self.eps = epsilon

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = F.broadcast_mul(x, F.rsqrt(F.mean(F.square(x), axis=1, keepdims=True) + self.eps))
        return y