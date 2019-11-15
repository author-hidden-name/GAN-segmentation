import time
import numpy as np
import mxnet as mx
from os.path import join, isdir, isfile, splitext
from utils import list_files_with_ext
import logging
logging.getLogger().setLevel(logging.INFO)
from networks_seg import Decoder
from seg_datasets import CollectionDataset
from metrics import SegmentationMetric
import cv2
from utils import get_draw_mask
from mxnet.gluon.data import DataLoader


class SegSolver():
    def __init__(self, max_res_log2, path_to_data, checkpoints_dir, gpu_ids, keep_weights=True):

        self.path_to_data = path_to_data
        self.checkpoints_dir = checkpoints_dir
        self.keep_weights = keep_weights

        if len(gpu_ids) == 0:
            # use CPU
            ctx = [mx.cpu()]
        else:
            ctx = [mx.gpu(i) for i in gpu_ids]

        self.is_trained = False
        self.params_file = None
        self.ctx = ctx
        self.cfg = self.get_config(max_res_log2=max_res_log2)
        self.net = self.init_net()
        self.is_trained = self.load()

    def init_net(self):

        initw = mx.init.Xavier(factor_type='in', magnitude=2.34)

        net = Decoder(self.cfg, num_devices=len(self.ctx))
        net.hybridize()
        if isinstance(initw, list):
            for p, initw_p in initw:
                net.collect_params(p).initialize(initw_p, ctx=self.ctx)
        else:
            net.collect_params().initialize(initw, ctx=self.ctx)
        self.print_params(net, 'decoder')

        return net

    def init_trainer(self):

        optimizer, optimizer_params = self.get_optimizer_params()
        loss = mx.gluon.loss.SoftmaxCELoss(axis=1)
        kv = mx.kv.create(self.cfg['kvstore'])
        trainer = mx.gluon.Trainer(self.net.collect_params(), optimizer, optimizer_params, kvstore=kv)

        return trainer, loss

    def print_params(self, net, title):
        net_params = net.collect_params()
        print('{:<36}{:<16}{:<24}{:<16}'.format(title, 'params', 'weight shape', 'dtype'))
        print('{:<36}{:<16}{:<24}{:<16}'.format('---', '---', '---', '---'))
        total = 0
        for p in net_params:
            param_data = net_params[p].data(ctx=self.ctx[0])
            pshape = param_data.shape
            if param_data.dtype == np.float32:
                pdtype = 'np.float32'
            elif param_data.dtype == np.float16:
                pdtype = 'np.float16'
            else:
                print('unknown dtype: {}'.format(pdtype))
                raise ValueError
            n_params = np.prod(np.array(pshape))
            pshape_str = str(pshape)
            total += n_params
            print('{:<36}{:<16}{:<24}{:<16}'.format(p, n_params, pshape_str, pdtype))
        print('{:<36}{:<16}{:<24}{:<16}'.format('---', '---', '---', '---'))
        print('{:<36}{:<16}'.format('total', total))
        print('{:<36}{:<16}'.format('---', '---'))

    def get_config(self, max_res_log2=9):
        cfg = {}

        cfg['seed'] = 1
        cfg['kvstore'] = 'nccl'
        cfg['cache_max_size'] = 4  # in GB
        cfg['plot_graph'] = True

        cfg['num_classes'] = 2
        cfg['not_ignore_classes'] = None
        cfg['cls_type'] = 'hair'

        cfg['train_epochs'] = 24

        cfg['base_lr'] = 1e-4
        cfg['factor_d'] = 0.1
        cfg['wd'] = 0.0
        cfg['optimizer'] = 'adam'
        cfg['momentum'] = None
        cfg['scheduler'] = None

        cfg['preprocess_mask'] = True

        cfg['train_display_iters'] = 4
        cfg['train_batch_size'] = 1
        cfg['val_batch_size'] = 1

        cfg['val_loader_workers'] = 0
        cfg['train_loader_workers'] = 0

        cfg['train_show_images'] = 1
        cfg['val_show_images'] = 1

        cfg['val_report_intermediate'] = False
        cfg['val_report_interval'] = 0.34

        cfg['use_bn'] = True
        cfg['use_sync_bn'] = False
        cfg['use_dropout'] = True
        cfg['start_res'] = 0

        cfg['features'] = [32, 32, 32, 32, 32, 32, 32, 32, 16]
        cfg['in_channels'] = [512, 512, 512, 512, 256, 128, 64, 32, 16]

        cfg['features'] = cfg['features'][:max_res_log2-1] + [cfg['num_classes']]
        cfg['in_channels'] = cfg['in_channels'][:max_res_log2-1]

        cfg['dtype'] = 'fp32'

        return cfg

    def init_data(self):

        cfg = self.cfg
        train_dataset = CollectionDataset(self.path_to_data, cfg, max_samples=None, load_to_memory=False)
        train_n_samples = len(train_dataset)
        if train_n_samples <= 0:
            print('number of training samples should be > 0')
            exit(-1)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], thread_pool=True,
                                      shuffle=True, num_workers=cfg['train_loader_workers'],
                                      last_batch='discard')
        iters_per_epoch = int(len(train_dataset) / cfg['train_batch_size'])


        print('total train samples: {}'.format(train_n_samples))
        print('batch size: {}'.format(cfg['train_batch_size']))
        print('epoch size: {}'.format(iters_per_epoch))

        return train_dataset, train_dataloader, iters_per_epoch

    def init_eval_data(self, input_dir):

        cfg = self.cfg
        eval_dataset = CollectionDataset(input_dir, cfg, max_samples=None, load_to_memory=False, output_idx=True)
        eval_n_samples = len(eval_dataset)
        if eval_n_samples <= 0:
            print('number of training samples should be > 0')
            raise ValueError

        eval_dataloader = DataLoader(eval_dataset, batch_size=cfg['val_batch_size'], thread_pool=True,
                                      shuffle=True, num_workers=cfg['val_loader_workers'],
                                      last_batch='discard')

        print('total eval samples: {}'.format(eval_n_samples))
        print('batch size: {}'.format(cfg['val_batch_size']))

        return eval_dataset, eval_dataloader

    def init_metric(self):
        train_metric = mx.metric.Accuracy()
        return train_metric

    def get_optimizer_params(self):

        cfg = self.cfg
        iters_per_epoch = self.iters_per_epoch

        factor_d = None
        optimizer = None
        wd = None
        momentum = None

        base_lr = cfg['base_lr']
        optimizer = cfg['optimizer']
        wd = cfg['wd']
        factor_d = cfg['factor_d']
        momentum = cfg['momentum']
        scheduler = cfg['scheduler']

        lr_sch = None
        if scheduler is not None:
            if scheduler == 'steps':
                epochs_steps = cfg['epochs_steps']
                iter_steps = [int(s * iters_per_epoch) for s in epochs_steps]
                lr_sch = mx.lr_scheduler.MultiFactorScheduler(iter_steps, factor=factor_d)
            elif scheduler == 'cos':
                lr_sch = mx.lr_scheduler.CosineScheduler(max_update=cfg['train_epochs'] * iters_per_epoch,
                                                         base_lr=base_lr,
                                                         final_lr=base_lr / 1000, warmup_begin_lr=base_lr / 10,
                                                         warmup_steps=iters_per_epoch * 1)
            else:
                raise ValueError

        optimizer_params = []
        if base_lr:
            optimizer_params.append(('learning_rate', base_lr))
        if lr_sch:
            optimizer_params.append(('lr_scheduler', lr_sch))
        if momentum:
            optimizer_params.append(('momentum', momentum))
        if wd:
            optimizer_params.append(('wd', wd))

        optimizer_params = {name: value for name, value in optimizer_params}

        return optimizer, optimizer_params

    def evaluate(self, input_dir, output_dir=None):
        eval_dataset, eval_dataloader = self.init_eval_data(input_dir)
        eval_metric = SegmentationMetric(self.cfg['num_classes'], skip_bg=True)
        loss_f = mx.gluon.loss.SoftmaxCELoss(axis=1)
        return self.evaluate_for_data(eval_dataset, eval_dataloader, loss_f, eval_metric, output_dir=output_dir)

    def evaluate_for_data(self, eval_dataset, val_dataloader, loss_f, eval_metric, output_dir=None):
        val_iter = iter(val_dataloader)
        total_loss = 0
        total_cnt = 0
        for item in val_iter:
            mask = item[2]
            features = item[3:]
            features = [feat.as_in_context(self.ctx[0]) for feat in features]

            mask = mask.as_in_context(self.ctx[0])
            pred_mask = self.net(*features)

            l_ones = mx.nd.ones(mask.shape, ctx=mask.context, dtype=np.float32)
            l_zeros = mx.nd.zeros(mask.shape, ctx=mask.context, dtype=np.float32)
            l_w = 1.0 * mx.nd.ones(mask.shape, ctx=mask.context, dtype=np.float32)

            sample_weight = mx.nd.where(mask > -1, l_ones, l_zeros)
            sample_weight = mx.nd.where(mask >= 0.5, l_w, sample_weight)

            err = loss_f(pred_mask, mask, sample_weight)

            loss_v = mx.nd.mean(err).asscalar()
            total_loss += loss_v
            total_cnt += 1

            mask = mx.nd.squeeze(mask, axis=1)
            eval_metric.update([mask], [pred_mask])

            if output_dir is not None:
                idx = item[0].asnumpy()
                imgs = item[1]
                imgs = mx.nd.transpose(imgs, (0, 2, 3, 1)).asnumpy()
                pred_mask_np = mx.nd.argmax(pred_mask, axis=1).asnumpy()
                mask_np = mask.asnumpy()

                for i in range(imgs.shape[0]):

                    metric = SegmentationMetric(self.cfg['num_classes'], skip_bg=True)
                    metric.update([mask[i:i+1]], [pred_mask[i:i+1]])
                    result_i = metric.get_name_value()

                    imname = eval_dataset.get_imname(idx[i])
                    img_i = imgs[i]
                    pred_mask_i = pred_mask_np[i].astype(np.int32)
                    mask_i = mask_np[i].astype(np.int32)

                    metric_str = ', '.join([f'{name} {v:.3f}' for name, v in result_i])

                    cv2.imwrite(join(output_dir, imname), img_i[:,:,::-1])

                    pred_mask_i[pred_mask_i == 1] = 255
                    pred_mask_i[pred_mask_i == 0] = 128
                    mask_i[mask_i == 1] = 255
                    mask_i[mask_i == 0] = 128
                    mask_i[mask_i == -1] = 0

                    pred_name = imname.replace('img', 'mask').replace('.jpg', '.png')
                    gt_name = imname.replace('img', 'gt_mask').replace('.jpg', '.png')
                    txt_name = imname.replace('img', 'metrics').replace('.jpg', '.txt')

                    cv2.imwrite(join(output_dir, imname), img_i[:,:,::-1])
                    cv2.imwrite(join(output_dir, pred_name), pred_mask_i)
                    cv2.imwrite(join(output_dir, gt_name), mask_i)

                    with open(join(output_dir, txt_name), 'w') as fp:
                        write_list = [imname, img_i.shape, pred_mask_i.shape, mask_i.shape, metric_str]
                        write_str = ', '.join([str(w) for w in write_list])
                        fp.write(write_str + '\n')

        if total_cnt > 0:
            total_loss = total_loss / total_cnt
        else:
            total_loss = 0.0

        result = eval_metric.get_name_value()
        result.append(('total-loss', total_loss))

        return result

    def predict(self, features):

        features_n = []
        for f in features:
            if not isinstance(f, mx.nd.NDArray):
                f = mx.nd.array(f, dtype=np.float32)
            if len(f.shape) == 3:
                f = mx.nd.expand_dims(f, axis=0)
            features_n.append(f)

        features_ctx = [mx.gluon.utils.split_and_load(feat, self.ctx) for feat in features_n]
        pred_mask_ctx = []
        for ctx_i in range(len(self.ctx)):
            features_s = [feat[ctx_i].detach() for feat in features_ctx]
            pred_mask_s = self.net(*features_s)
            pred_mask_ctx.append(pred_mask_s)
        mx.nd.waitall()
        pred_mask_ctx = [pred_mask.as_in_context(mx.cpu()) for pred_mask in pred_mask_ctx]
        pred_masks = mx.nd.concatenate(pred_mask_ctx, axis=0)
        pred_masks = mx.nd.argmax(pred_masks, axis=1, keepdims=True)
        pred_masks = mx.nd.transpose(pred_masks, (0, 2, 3, 1)).asnumpy()

        return pred_masks

    def save(self, suffix=None):
        if suffix is None:
            param_name = 'checkpoint_last.params'
        else:
            param_name = f'checkpoint_{suffix}.params'
        self.params_file = param_name
        self.net.save_parameters(join(self.checkpoints_dir, param_name))

    def load(self):

        params_files = list_files_with_ext(self.checkpoints_dir, valid_exts=['.params'])
        if len(params_files) > 0:
            params_file = params_files[0]
            print(f'loading checkpoint: {params_file}')
            self.params_file = params_file
            self.net.load_parameters(join(self.checkpoints_dir, params_file))
            return True
        else:
            return False

    def fit(self, epoch_end_callback=None):

        if not self.keep_weights:
            self.net = self.init_net()

        self.train_dataset, self.train_dataloader, self.iters_per_epoch = self.init_data()
        self.trainer, self.loss = self.init_trainer()
        self.train_metric = self.init_metric()

        loss = self.loss
        trainer = self.trainer
        train_dataloader = self.train_dataloader
        iters_per_epoch = self.iters_per_epoch
        batch_size = self.cfg['train_batch_size']

        display = self.cfg['train_display_iters']
        train_metric = self.train_metric
        epochs_to_train = self.cfg['train_epochs']

        scores = []

        for epoch in range(epochs_to_train):
            # new epoch
            tic = time.time()

            train_metric.reset()

            nbatch = 0
            name_values = []
            speed_tic = time.time()
            data_iter = iter(train_dataloader)
            end_of_batch = False
            next_data_batch = next(data_iter)

            while not end_of_batch:
                mask = next_data_batch[1]
                features = next_data_batch[2:]

                mask_ctx = mx.gluon.utils.split_and_load(mask, self.ctx)
                features_ctx = [mx.gluon.utils.split_and_load(feat, self.ctx) for feat in features]

                err_ctx = []
                pred_mask_ctx = []
                with mx.autograd.record():
                    for ctx_i in range(len(self.ctx)):
                        mask_s = mask_ctx[ctx_i]
                        features_s = [feat[ctx_i].detach() for feat in features_ctx]
                        pred_mask_s = self.net(*features_s)

                        l_ones = mx.nd.ones(mask_s.shape, ctx=mask_s.context, dtype=np.float32)
                        l_zeros = mx.nd.zeros(mask_s.shape, ctx=mask_s.context, dtype=np.float32)
                        l_w = 1.0 * mx.nd.ones(mask_s.shape, ctx=mask_s.context, dtype=np.float32)

                        sample_weight = mx.nd.where(mask_s > -1, l_ones, l_zeros)
                        sample_weight = mx.nd.where(mask_s >= 0.5, l_w, sample_weight)

                        err_s = loss(pred_mask_s, mask_s, sample_weight)

                        err_ctx.append(err_s)
                        pred_mask_ctx.append(pred_mask_s)

                for err_s in err_ctx:
                    err_s.backward()

                for ctx_i in range(len(self.ctx)):
                    pred_mask_s = pred_mask_ctx[ctx_i]
                    mask_s = mask_ctx[ctx_i]
                    mask_s = mx.nd.squeeze(mask_s, axis=1)
                    train_metric.update([mask_s], [pred_mask_s])

                trainer.step(mask.shape[0])

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                except StopIteration:
                    end_of_batch = True

                if end_of_batch:
                    name_values = train_metric.get_name_value()

                nbatch += 1
                global_step = (epoch * iters_per_epoch + nbatch) * batch_size

                # speedometer
                if display is not None and nbatch % display == 0:

                    speed = 1.0 * display * batch_size / (time.time() - speed_tic)

                    name_value = train_metric.get_name_value()
                    total_loss = np.mean([mx.nd.mean(err_sum).asscalar() for err_sum in err_ctx])
                    name_value.append(('total-loss', total_loss))
                    train_metric.reset()

                    msg = 'Epoch[%03d] Batch[%04d] Speed: % 9.2f samples/sec'
                    msg += ' %s=%f' * len(name_value)
                    logging.info(msg, epoch, nbatch, speed, *sum(name_value, ()))
                    speed_tic = time.time()

            global_step = ((epoch + 1) * iters_per_epoch) * batch_size

            # one epoch of training is finished
            for name, val in name_values:
                logging.info('Epoch[%d] Train-%s=%f', epoch + 1, name, val)
            time_cost = (time.time() - tic)
            logging.info('Epoch[%d] Learning rate=%.5f', epoch + 1, trainer.learning_rate)
            logging.info('Epoch[%d] Time cost=%.3f', epoch + 1, time_cost)

            if epoch_end_callback is not None:
                epoch_end_callback()

        # save checkpoint
        self.is_trained = True
        self.save()

        return scores