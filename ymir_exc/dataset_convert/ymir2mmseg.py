"""
convert ymir dataset to semantic segmentation dataset
"""
import os
import random

from easydict import EasyDict as edict
from pycocotools import coco
from pycocotools import mask as maskUtils
from ymir_exc.util import get_bool


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
            rle = coco_ann.annToRLE(ann)
            mask_area = maskUtils.area(rle)
            total_mask_area += mask_area

        # total_mask_area < width * height means exist background
        if total_mask_area < width * height:
            return True

    return False


def train_with_black_area_or_not(ymir_cfg: edict, max_sample_num: int = 100) -> bool:
    if get_bool(ymir_cfg, key='ignore_blank_area', default_value=False):
        return False

    env_with_blank_area = os.environ.get('WITH_BLANK_AREA', None)
    if env_with_blank_area:
        if env_with_blank_area.lower() == 'true':
            return True
        elif env_with_blank_area.lower() == 'false':
            return False
        else:
            raise Exception(f'unknown value for WITH_BLANK_AREA = {env_with_blank_area}')

    # set environment variable to keep consistency
    with_blank_area = find_blank_area_in_dataset(ymir_cfg, max_sample_num)
    if with_blank_area:
        os.environ.setdefault('WITH_BLANK_AREA', 'TRUE')
    else:
        os.environ.setdefault('WITH_BLANK_AREA', 'FALSE')

    return with_blank_area
