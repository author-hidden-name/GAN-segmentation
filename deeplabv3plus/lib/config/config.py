import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.RT = edict()
cfg.RT.gpus = '0'
cfg.RT.kvstore = 'local'
cfg.RT.no_cuda = False
cfg.RT.dataloader_workers = 4

cfg.MODEL = edict()

cfg.TRAIN = edict()
cfg.TRAIN.METRICS = edict()
cfg.TRAIN.DATALOADER = edict()
cfg.TRAIN.DATALOADER.MODULE = 'data.sun_rgbd.rgb_segmentation'

cfg.TEST = edict()
cfg.TEST.METRICS = edict()
cfg.TEST.DATALOADER = edict()


def load_config(config_path):
    with open(config_path, 'r') as f:
        loaded_config = yaml.load(f)

        for k, v in loaded_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        cfg[k][vk] = vv
                else:
                        cfg[k] = v
            else:
                if isinstance(v, dict):
                    cfg[k] = edict(v)
                else:
                    cfg[k] = v
