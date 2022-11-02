"""
convert ymir dataset to cityscapes segmentation dataset
"""
from PIL import Image
import numpy as np
from typing import Any, Dict, Tuple


def convert_rgb_to_label_id(rgb_img: str, label_id_img: str, palatte_dict: Dict[Tuple, int], dtype: Any = np.uint8):
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
