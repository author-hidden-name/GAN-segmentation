import numpy as np
import mxnet as mx
from networks_stylegan import Generator


class ImageGenerator():

    def __init__(self, gpu_ids, gan_dir, gan='ffhq', batch_size=4, return_latents=False):
        super().__init__()

        max_res_log2_dict = {'ffhq': 10, 'cars': 9, 'bedrooms': 8}
        self.max_res_log2 = max_res_log2_dict[gan]

        self.latent_size = 512
        self.return_latents = return_latents
        self.batch_size = batch_size
        self.ctx = [mx.gpu(gpu_id) for gpu_id in gpu_ids] if len(gpu_ids) > 0 else [mx.cpu()]
        self.cfg = self._get_config(max_res_log2=self.max_res_log2)

        self.netG = self._get_G(self.cfg, self.ctx, initialize=False)
        stylegan_name = f'stylegan-{gan}.params'
        self.netG.load_parameters(f'{gan_dir}/{stylegan_name}', ignore_extra=True, ctx=self.ctx)


    def _get_G(self, config, ctx, initialize=True):
        # build the generator
        netG = Generator(config)

        if initialize:
            if config['init'] == 'normal':
                std = config['init_normal_std']
                netG.initialize(mx.init.Normal(std), ctx=ctx)
            elif config['init'] == 'xavier':
                magnitude = config['init_xavier_magnitude']
                netG.initialize(mx.init.Xavier(factor_type='in', magnitude=magnitude), ctx=ctx)
            elif config['init'] == 'orthogonal':
                netG.initialize(mx.init.Orthogonal(), ctx=ctx)
            else:
                print('unknown initialization: {}'.format(config['init']))
                raise NotImplementedError

        netG.collect_params('mp_dense_*').setattr('lr_mult', 0.01)

        return netG

    def _get_config(self, max_res_log2=9):
        cfg = {}

        # network parameters
        cfg['use_wscale'] = True

        cfg['fmap_base'] = 8192
        cfg['fmap_decay'] = 1.0
        cfg['fmap_max'] = 512
        cfg['max_res_log2'] = max_res_log2
        cfg['fix_noise'] = False

        cfg['base_scale_x'] = 4
        cfg['base_scale_y'] = 4

        # initialization
        cfg['init'] = 'normal'  # 'xavier', 'normal', 'orthogonal'
        cfg['init_normal_std'] = 1.0
        cfg['init_xavier_magnitude'] = 1.0

        # input format
        cfg['latent_size'] = 512
        cfg['latent_prior'] = 'normal'  # 'normal' or 'bernoulli'

        cfg['channels'] = 3
        cfg['imrange'] = (-1, 1)
        cfg['dtype'] = 'fp32'

        return cfg

    def _transform_gan_back(self, img, cfg):
        imrange = cfg['imrange']
        channel_swap = (0, 2, 3, 1)
        img = np.transpose(img, axes=channel_swap)
        img = (img - imrange[0]) / (imrange[1] - imrange[0])
        img = np.clip(img, 0.0, 1.0)
        img = 255. * img
        img = img.astype(np.uint8)
        return img

    def get_images(self, n):

        n_batches = n // self.batch_size
        n_batches += 1 if n % self.batch_size > 0 else 0

        n_generated = 0
        for n_batch in range(n_batches):
            batch_size_s = min(self.batch_size, n - n_generated)
            latent_z = mx.nd.random_normal(0, 1, shape=(batch_size_s, self.latent_size))
            latent_z_ctx = mx.gluon.utils.split_and_load(latent_z, self.ctx, even_split=False)
            data_ctx = []
            features_ctx = []
            for latent_z_s in latent_z_ctx:
                data_s, features_s = self.netG(latent_z_s)
                data_ctx.append(data_s)
                features_ctx.append(features_s)
            mx.nd.waitall()
            data = [data_s.asnumpy() for data_s in data_ctx]
            data = np.concatenate(data, axis=0)
            latent_z_np = latent_z.asnumpy()

            features = []
            for feature_i in range(len(features_ctx[0])):
                feat_all_ctx = []
                for features_s in features_ctx:
                    feat = features_s[feature_i].asnumpy()
                    feat_all_ctx.append(feat)
                feat_all_ctx = np.concatenate(feat_all_ctx, axis=0)
                features.append(feat_all_ctx)

            imgs = self._transform_gan_back(data, self.cfg)
            n_generated += imgs.shape[0]
            for i in range(imgs.shape[0]):
                img = imgs[i]
                feats = [feat[i] for feat in features]
                if self.return_latents:
                    yield img, feats, latent_z_np
                else:
                    yield img, feats