from os.path import join, isdir, basename, splitext, isfile, islink
from os.path import sep as os_path_sep
from os import listdir, walk
import numpy as np
import cv2
import yaml


def list_subdirs(base_dir):
    subdirs = []
    for f in listdir(base_dir):
        if not isdir(join(base_dir, f)):
            continue
        subdirs.append(f)
    return subdirs


def list_files_with_ext(base_dir, valid_exts, recursive=False):
    images = []

    if recursive:
        list_files_with_ext_rec(base_dir, images, valid_exts)
    else:
        assert isdir(base_dir) or islink(base_dir), f'{base_dir} is not a valid directory'
        base_path_len = len(base_dir.split(os_path_sep))
        for root, dnames, fnames in sorted(walk(base_dir)):
            root_parts = root.split(os_path_sep)
            root_m = os_path_sep.join(root_parts[base_path_len:])
            for fname in fnames:
                if not isfile(join(root, fname)):
                    continue
                filext = splitext(fname.lower())[1]
                if filext not in valid_exts:
                    continue
                path = join(root_m, fname)
                images.append(path)
    return images


def list_files_with_ext_rec(base_dir, images, valid_exts):
    assert isdir(base_dir), f'{base_dir} is not a valid directory'
    base_path_len = len(base_dir.split(os_path_sep))
    for root, dnames, fnames in sorted(walk(base_dir, followlinks=True)):
        root_parts = root.split(os_path_sep)
        root_m = os_path_sep.join(root_parts[base_path_len:])

        for fname in fnames:
            if not isfile(join(root, fname)):
                continue
            filext = splitext(fname.lower())[1]
            if filext not in valid_exts:
                continue
            path = join(root_m, fname)
            images.append(path)


def list_images(base_dir, valid_exts=['.jpg', '.jpeg', '.png', '.bmp', '.ppm']):
    images_list = []
    for f in listdir(base_dir):
        if not isfile(join(base_dir, f)):
            continue
        filext = splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        images_list.append(f)
    return images_list


def get_seg_color_map():
    color_bg = np.array([0,0,0],dtype=np.uint8)
    color_fg = np.array([13, 198, 20],dtype=np.uint8)
    color_neg = np.array([54, 30, 211],dtype=np.uint8)
    color_map = []
    color_map.append([0, color_bg])
    color_map.append([1, color_fg])
    color_map.append([2, color_neg])
    return color_map


def get_draw_mask(img, mask, alpha=0.5, color_map=None, skip_background=True):
    if color_map is None:
        color_map = get_seg_color_map()

    im_cpy = np.array(img)

    im_cpy_b = im_cpy[:, :, 0]
    im_cpy_g = im_cpy[:, :, 1]
    im_cpy_r = im_cpy[:, :, 2]

    for idx, color in color_map:
        if idx == 0 and skip_background:
            continue
        mask_cur = mask == idx
        im_cpy_b[mask_cur] = alpha * color[0] + (1 - alpha) * im_cpy_b[mask_cur]
        im_cpy_g[mask_cur] = alpha * color[1] + (1 - alpha) * im_cpy_g[mask_cur]
        im_cpy_r[mask_cur] = alpha * color[2] + (1 - alpha) * im_cpy_r[mask_cur]

    im_cpy[:, :, 0] = im_cpy_b
    im_cpy[:, :, 1] = im_cpy_g
    im_cpy[:, :, 2] = im_cpy_r

    return im_cpy


def morph_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def load_config_file(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def crop_image(img, bbox):
    x_st = bbox[0]
    y_st = bbox[1]

    x_en = bbox[0] + bbox[2] - 1
    y_en = bbox[1] + bbox[3] - 1

    x_st_pad = int(max(0, -x_st))
    y_st_pad = int(max(0, -y_st))
    x_en_pad = int(max(0, x_en - img.shape[1] + 1))
    y_en_pad = int(max(0, y_en - img.shape[0] + 1))

    x_en = x_en + max(0, -x_st)
    y_en = y_en + max(0, -y_st)
    x_st = max(0, x_st)
    y_st = max(0, y_st)

    if y_st_pad != 0 or y_en_pad != 0 or x_st_pad != 0 or x_en_pad != 0:
        assert len(img.shape) in (2, 3)
        if len(img.shape) == 3:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad, img.shape[2]), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1], :] = img
        else:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1]] = img
    else:
        img_pad = img
    img_cropped = img_pad[y_st:y_en+1, x_st:x_en+1]
    return img_cropped


def prepare_crop(im, prepare_sz, fit_whole=False, use_nn_interpolation=False):
    if im.shape[0] != prepare_sz[1] or im.shape[1] != prepare_sz[0]:
        prepare_r = float(prepare_sz[0]) / prepare_sz[1]
        orig_r = float(im.shape[1]) / im.shape[0]

        if fit_whole:
            do_fit_width = orig_r > prepare_r
        else:
            do_fit_width = orig_r < prepare_r

        if do_fit_width:
            # fit width
            crop_w = im.shape[1]
            crop_h = crop_w / prepare_r
        else:
            # fit height
            crop_h = im.shape[0]
            crop_w = crop_h * prepare_r

        crop_x = int((im.shape[1] - crop_w) / 2.)
        crop_y = int((im.shape[0] - crop_h) / 2.)
        crop_w = int(crop_w)
        crop_h = int(crop_h)

        crop_rect = [crop_x, crop_y, crop_w, crop_h]
        im = crop_image(im, crop_rect)

        interp = cv2.INTER_NEAREST if use_nn_interpolation else cv2.INTER_LINEAR
        im = cv2.resize(im, prepare_sz, interpolation=interp)
    return im