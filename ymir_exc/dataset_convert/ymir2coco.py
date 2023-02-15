import json
import os
import os.path as osp
from typing import Dict, List

import imagesize

from ymir_exc.util import get_merged_config


def convert_ymir_to_coco(cat_id_from_zero: bool = False) -> Dict[str, Dict[str, str]]:
    """
    convert ymir dataset to coco format for training task
    for the input index file:
        ymir_index_file: for each line it likes: {img_path} \t {ann_path}

    for the outout json file:
        cat_id_from_zero: category id start from zero or not

    output the coco dataset information:
        dict(train=dict(img_dir=xxx, ann_file=xxx),
            val=dict(img_dir=xxx, ann_file=xxx))
    """
    cfg = get_merged_config()
    out_dir = cfg.ymir.output.root_dir
    ymir_dataset_dir = osp.join(out_dir, "ymir_dataset")
    os.makedirs(ymir_dataset_dir, exist_ok=True)

    output_info = {}
    for split, prefix in zip(["train", "val"], ["training", "val"]):
        ymir_index_file = getattr(cfg.ymir.input, f"{prefix}_index_file")
        with open(ymir_index_file) as fp:
            lines = fp.readlines()

        img_id = 0
        ann_id = 0
        data: Dict[str, List] = dict(images=[], annotations=[], categories=[], licenses=[])

        cat_id_start = 0 if cat_id_from_zero else 1
        for id, name in enumerate(cfg.param.class_names):
            data["categories"].append(dict(id=id + cat_id_start, name=name, supercategory="none"))

        for line in lines:
            img_file, ann_file = line.strip().split()
            width, height = imagesize.get(img_file)
            img_info = dict(file_name=img_file, height=height, width=width, id=img_id)

            data["images"].append(img_info)

            if osp.exists(ann_file):
                for ann_line in open(ann_file, "r").readlines():
                    ann_strlist = ann_line.strip().split(",")
                    class_id, x1, y1, x2, y2 = [int(s) for s in ann_strlist[0:5]]
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    bbox_quality = (float(ann_strlist[5]) if len(ann_strlist) > 5 and ann_strlist[5].isnumeric() else 1)
                    ann_info = dict(
                        bbox=[x1, y1, bbox_width, bbox_height],  # x,y,width,height
                        area=bbox_area,
                        score=1.0,
                        bbox_quality=bbox_quality,
                        iscrowd=0,
                        segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
                        category_id=class_id + cat_id_start,  # start from cat_id_start
                        id=ann_id,
                        image_id=img_id,
                    )
                    data["annotations"].append(ann_info)
                    ann_id += 1

            img_id += 1

        split_json_file = osp.join(ymir_dataset_dir, f"ymir_{split}.json")
        with open(split_json_file, "w") as fw:
            json.dump(data, fw)

        output_info[split] = dict(img_dir=cfg.ymir.input.assets_dir, ann_file=split_json_file)

    return output_info
