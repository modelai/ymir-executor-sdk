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


def convert_ymir_to_mmseg(ymir_cfg: edict) -> Dict[str, str]:
    """
    convert annotation images from RGB mode to label id mode
    return new index files
    note: call before ddp, avoid multi-process problem
    """
    ymir_ann_files = dict(
        train=ymir_cfg.ymir.input.training_index_file,
        val=ymir_cfg.ymir.input.val_index_file,
        test=ymir_cfg.ymir.input.candidate_index_file,
    )

    in_dir = ymir_cfg.ymir.input.root_dir
    out_dir = ymir_cfg.ymir.output.root_dir
    new_ann_files = dict()
    for split in ["train", "val"]:
        new_ann_files[split] = osp.join(
            out_dir, osp.relpath(ymir_ann_files[split], in_dir)
        )

    new_ann_files["test"] = ymir_ann_files["test"]
    # call before ddp, avoid multi-process problem, just to return new_ann_files
    if osp.exists(new_ann_files["train"]):
        return new_ann_files

    label_map_txt = osp.join(ymir_cfg.ymir.input.annotations_dir, "labelmap.txt")
    with open(label_map_txt, "r") as fp:
        lines = fp.readlines()

    # note: class_names maybe the subset of label_map
    class_names = ymir_cfg.param.class_names
    palette_dict: Dict[Tuple, int] = {}
    for idx, line in enumerate(lines):
        label, rgb = line.split(":")[0:2]
        r, g, b = [int(x) for x in rgb.split(",")]
        if label in class_names:
            class_id = class_names.index(label)
            palette_dict[(r, g, b)] = class_id
            logging.info(f"label map: {class_id}={label} ({r}, {g}, {b})")
        else:
            logging.info(f"ignored label in labelmap.txt: {label} {rgb}")
    # palette_dict[(0, 0, 0)] = 255

    for split in ["train", "val"]:
        with open(ymir_ann_files[split], "r") as fp:
            lines = fp.readlines()

        fw = open(new_ann_files[split], "w")
        for line in tqdm(lines, desc=f"convert {split} dataset"):
            img_path, ann_path = line.strip().split()

            new_ann_path = osp.join(out_dir, osp.relpath(ann_path, in_dir))
            os.makedirs(osp.dirname(new_ann_path), exist_ok=True)
            save_rgb_to_label_id(ann_path, new_ann_path, palette_dict)
            fw.write(f"{img_path}\t{new_ann_path}\n")
        fw.close()

    return new_ann_files
