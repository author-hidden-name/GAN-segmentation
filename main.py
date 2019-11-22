import time
import mxnet as mx
import numpy as np
from seg_annotator import SegmentationAnnotator
from utils import load_config_file
import tkinter as tk
import argparse
from os.path import join, isdir
from os import mkdir, makedirs
from seg_solver import SegSolver
from tqdm import tqdm
import cv2
from image_generator import ImageGenerator



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('action', nargs='?',
                        choices=('annotation', 'train', 'evaluate', 'generate'),
                        default='annotation')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    seed = 0
    np.random.seed(seed)
    mx.random.seed(seed)

    cfg = load_config_file('config.yml')
    root_dir = cfg['BASE_DIR']
    gan = cfg['GAN']
    gan_dir = cfg['GAN_DIR']
    gan_gpu_ids = cfg['GAN_GPU_IDS']
    gan_batch_size = cfg['GAN_BATCH_SIZE_PER_GPU']
    solver_gpu_ids = cfg['SOLVER_GPU_IDS']
    annotation = cfg['ANNOTATION']
    no_gan = cfg.get('NO_GAN', False)
    imgs_dir = cfg.get('IMGS_DIR', None)
    n_generate = cfg.get('GENERATE_NUM', 10000)

    if args.action == 'annotation':
        root = tk.Tk()
        if annotation == 'segmentation':
            SegmentationAnnotator(root, root_dir, gan_gpu_ids=gan_gpu_ids, solver_gpu_ids=solver_gpu_ids,
                                  gan_dir=gan_dir, gan=gan, n_generate=n_generate).pack(fill='both', expand=True)
        else:
            print(f'uknown annotation type: {annotation}')
        root.mainloop()
    elif args.action == 'train':
        max_res_log2_dict = {'ffhq': 10, 'cars': 9, 'bedrooms': 8}
        max_res_log2 = max_res_log2_dict[gan]
        solver = SegSolver(max_res_log2, join(root_dir, 'data'),
                           join(root_dir, 'checkpoints'), gpu_ids=solver_gpu_ids,
                           keep_weights=False)
        solver.fit()
    elif args.action == 'evaluate':
        max_res_log2_dict = {'ffhq': 10, 'cars': 9, 'bedrooms': 8}
        max_res_log2 = max_res_log2_dict[gan]
        solver = SegSolver(max_res_log2, join(root_dir, 'data'),
                           join(root_dir, 'checkpoints'), gpu_ids=solver_gpu_ids,
                           keep_weights=False)
        if not solver.is_trained:
            print('train Decoder first!')
            exit(-1)

        result = solver.evaluate(join(root_dir, 'eval'))
        result_str = ', '.join([f'{name}: {value:.4f}' for name, value in result])
        print(result_str)

    elif args.action == 'generate':

        max_res_log2_dict = {'ffhq': 10, 'cars': 9, 'bedrooms': 8}
        max_res_log2 = max_res_log2_dict[gan]
        solver = SegSolver(max_res_log2, join(root_dir, 'data'),
                           join(root_dir, 'checkpoints'), gpu_ids=solver_gpu_ids,
                           keep_weights=False)
        if not solver.is_trained:
            print('train Decoder first!')
            exit(-1)

        buffer_size = min(2, len(gan_gpu_ids))
        batch_size = gan_batch_size*len(gan_gpu_ids)
        netG = ImageGenerator(gpu_ids=gan_gpu_ids, gan_dir=gan_dir, gan=gan, batch_size=batch_size)
        dst_dir = join(root_dir, 'dataset', 'train_generated')
        if not isdir(dst_dir):
            makedirs(dst_dir)

        n_imgs = n_generate
        iter = netG.get_images(n_imgs)
        index = 0
        with tqdm(total=n_imgs) as pb:
            for index in range(n_imgs):
                img, features = next(iter)
                mask = solver.predict(features)[0].astype(np.uint8)
                imname = f'img_{index:06d}.jpg'
                maskname = f'mask_{index:06d}.png'
                cv2.imwrite(join(dst_dir, imname), img[:, :, ::-1])
                cv2.imwrite(join(dst_dir, maskname), mask[:, :, 0])
                pb.update()
