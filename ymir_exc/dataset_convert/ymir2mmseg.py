"""
convert ymir dataset to cityscapes segmentation dataset
"""
import logging
import os
import os.path as osp
from typing import Any, Dict, Tuple, Union

import numpy as np
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm


def convert_rgb_to_label_id(rgb_img: Union[Image.Image, np.ndarray, str],
                            palatte_dict: Dict[Tuple, int],
                            dtype: Any = np.uint8) -> Union[Image.Image, np.ndarray, str]:
    """
    map rgb color to label id, start from 1
    for the output mask, note to ignore label 0.
    """
    if isinstance(rgb_img, Image.Image):
        np_rgb_img = np.array(rgb_img)

    if isinstance(rgb_img, str):
        np_rgb_img = np.array(Image.open(rgb_img))

    height, width = np_rgb_img.shape[0:2]
    np_label_id = np.ones(shape=(height, width), dtype=dtype) * 255

    # rgb = (0,0,0), idx = 0 can skip.
    for rgb, idx in palatte_dict.items():
        r = np_rgb_img[:, :, 0] == rgb[0]
        g = np_rgb_img[:, :, 1] == rgb[1]
        b = np_rgb_img[:, :, 2] == rgb[2]

        np_label_id[r & g & b] = idx

    if isinstance(rgb_img, (Image.Image, str)):
        # mode = L: 8-bit unsigned integer pixels
        # mode = I: 32-bit signed integer pixels
        pil_label_id_img = Image.fromarray(np_label_id, mode='L' if dtype == np.uint8 else 'I')
        return pil_label_id_img
    elif isinstance(rgb_img, np.ndarray):
        return np_label_id
    else:
        assert False, f'unknown rgb_img format {type(rgb_img)}'


def save_rgb_to_label_id(rgb_img: str, label_id_img: str, palatte_dict: Dict[Tuple, int], dtype: Any = np.uint8):
    """
    map rgb color to label id, start from 1
    for the output mask, note to ignore label 0.
    """
    pil_rgb_img = Image.open(rgb_img)
    np_rgb_img = np.array(pil_rgb_img)
    height, width = np_rgb_img.shape[0:2]
    np_label_id = np.ones(shape=(height, width), dtype=dtype) * 255

    # rgb = (0,0,0), idx = 0 can skip.
    for rgb, idx in palatte_dict.items():
        r = np_rgb_img[:, :, 0] == rgb[0]
        g = np_rgb_img[:, :, 1] == rgb[1]
        b = np_rgb_img[:, :, 2] == rgb[2]

        np_label_id[r & g & b] = idx

    # mode = L: 8-bit unsigned integer pixels
    # mode = I: 32-bit signed integer pixels
    pil_label_id_img = Image.fromarray(np_label_id, mode='L' if dtype == np.uint8 else 'I')
    pil_label_id_img.save(label_id_img)


def convert_ymir_to_mmseg(ymir_cfg: edict) -> Dict[str, str]:
    """
    convert annotation images from RGB mode to label id mode
    return new index files
    note: call before ddp, avoid multi-process problem
    """
    ymir_ann_files = dict(train=ymir_cfg.ymir.input.training_index_file,
                          val=ymir_cfg.ymir.input.val_index_file,
                          test=ymir_cfg.ymir.input.candidate_index_file)

    in_dir = ymir_cfg.ymir.input.root_dir
    out_dir = ymir_cfg.ymir.output.root_dir
    new_ann_files = dict()
    for split in ['train', 'val']:
        new_ann_files[split] = osp.join(out_dir, osp.relpath(ymir_ann_files[split], in_dir))

    new_ann_files['test'] = ymir_ann_files['test']
    # call before ddp, avoid multi-process problem, just to return new_ann_files
    if osp.exists(new_ann_files['train']):
        return new_ann_files

    label_map_txt = osp.join(ymir_cfg.ymir.input.annotations_dir, 'labelmap.txt')
    with open(label_map_txt, 'r') as fp:
        lines = fp.readlines()

    # note: class_names maybe the subset of label_map
    class_names = ymir_cfg.param.class_names
    palatte_dict: Dict[Tuple, int] = {}
    for idx, line in enumerate(lines):
        label, rgb = line.split(':')[0:2]
        r, g, b = [int(x) for x in rgb.split(',')]
        if label in class_names:
            class_id = class_names.index(label)
            palatte_dict[(r, g, b)] = class_id
            logging.info(f'label map: {class_id}={label} ({r}, {g}, {b})')
        else:
            logging.info(f'ignored label in labelmap.txt: {label} {rgb}')
    # palatte_dict[(0, 0, 0)] = 255

    for split in ['train', 'val']:
        with open(ymir_ann_files[split], 'r') as fp:
            lines = fp.readlines()

        fw = open(new_ann_files[split], 'w')
        for line in tqdm(lines, desc=f'convert {split} dataset'):
            img_path, ann_path = line.strip().split()

            new_ann_path = osp.join(out_dir, osp.relpath(ann_path, in_dir))
            os.makedirs(osp.dirname(new_ann_path), exist_ok=True)
            save_rgb_to_label_id(ann_path, new_ann_path, palatte_dict)
            fw.write(f'{img_path}\t{new_ann_path}\n')
        fw.close()

    return new_ann_files
