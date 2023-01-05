"""
convert ymir dataset to semantic segmentation dataset
"""
import logging
import os
import os.path as osp
import random
from typing import Any, Dict, Tuple, Union

import numpy as np
from easydict import EasyDict as edict
from PIL import Image
from pycocotools import coco
from pycocotools import mask as maskUtils
from tqdm import tqdm


def find_blank_area_in_dataset(ymir_cfg: edict, max_sample_num: int = 100) -> bool:
    """
    check the coco annotation file has blank area or not
    """
    with open(ymir_cfg.ymir.input.training_index_file, 'r') as fp:
        lines = fp.readlines()
    coco_ann_file = lines[0].split()[1]

    coco_ann = coco.COCO(coco_ann_file)
    img_ids = coco_ann.getImgIds()
    sample_img_ids = random.sample(img_ids, min(max_sample_num, len(img_ids)))
    for img_id in sample_img_ids:
        ann_ids = coco_ann.getAnnIds(imgIds=[img_id])
        width = coco_ann.imgs[img_id]['width']
        height = coco_ann.imgs[img_id]['height']

        total_mask_area = 0
        for ann_id in ann_ids:
            ann = coco_ann.anns[ann_id]
            mask_area = maskUtils.area(ann['segmentation'])
            total_mask_area += mask_area

        # total_mask_area < width * height means exist background
        if total_mask_area < width * height:
            return True

    return False
