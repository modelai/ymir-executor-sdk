"""
convert cityscapes dataset to ymir import format

eg:
python3 cityscapes_to_ymir.py --root_dir xxx --split train --out_dir xxx --num 500

suppose we have cityscapes dataset as follow:

root_dir
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val

output ymir dataset:

root_dir
├── train
│   ├── gt
│   │   └── coco-annotations.json
│   ├── images
└── val
    ├── gt
    │   └── coco-annotations.json
    ├── images

images changes:
# cityscapes format
{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}

# cityscapes image
leftImg8bit/val/munster/munster_000001_000019_leftImg8bit.png

# cityscapes annotation
gtFine/val/munster/munster_000000_000019_gtFine_color.png
gtFine/val/munster/munster_000000_000019_gtFine_instanceIds.png
gtFine/val/munster/munster_000000_000019_gtFine_labelIds.png
gtFine/val/munster/munster_000000_000019_gtFine_polygons.json

# ymir image
root_dir/train/images/munster_000001_000019_leftImg8bit.png

# ymir annotation file
root_dir/train/gt/coco-annotations.json
"""
import argparse
import glob
import json
import os
import os.path as osp
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
# from cityscapesscripts.helpers.labels import labels  # type: ignore
from cityscapesscripts.preparation.json2labelImg import \
    json2labelImg  # type: ignore
from tools.seg.pycococreatortools import (create_annotation_info, create_image_info)
from tqdm import tqdm  # type: ignore


def convert_json_to_label(json_file) -> str:
    """
    from https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py
    """
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    if not osp.exists(label_file):
        json2labelImg(json_file, label_file, 'trainIds')

    return label_file


def get_args():
    parser = argparse.ArgumentParser("convert cityscapes dataset to ymir")
    parser.add_argument("--root_dir", help="root dir for cityscapes dataset")
    parser.add_argument("--split", default="train", help="split for dataset")
    parser.add_argument("--out_dir", help="the output directory", default="./out")
    parser.add_argument("--num", help="sample number for dataset", default=0, type=int)

    return parser.parse_args()


class COCOAnn(object):

    def __init__(self, ann_file):
        self.ann_file = ann_file
        self.init()

    def init(self):
        self.anns = {}
        self.anns['images'] = []
        self.anns['annotations'] = []
        self.anns['categories'] = []
        self.img_id, self.ann_id = 0, 0

        self.class_names = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle')

        for class_id, class_name in enumerate(self.class_names):
            self.anns['categories'].append(dict(id=class_id, name=class_name, supercategory='none'))

    def add_ann(self, img_file, ann_file):
        ann = cv2.imread(ann_file, cv2.IMREAD_GRAYSCALE)
        height, width = ann.shape[0:2]
        image_info = create_image_info(image_id=self.img_id,
                                       file_name=osp.basename(img_file),
                                       image_size=(width, height))
        self.anns['images'].append(image_info)

        exist_class_ids = np.unique(ann)
        for class_id, class_name in enumerate(self.class_names):
            if class_id in exist_class_ids:
                binary_mask = ann == class_id
                # use is_crowd = True to create rle mask for semantic segmentation
                # use is-crowd = False to create polygon
                category_info = {'id': class_id, 'is_crowd': True}
                ann_info = create_annotation_info(self.ann_id, self.img_id, category_info, binary_mask, tolerance=2)

                if ann_info is not None:
                    self.anns['annotations'].append(ann_info)
                    self.ann_id += 1
                else:
                    print(f'empty mask for {img_file} {ann_file} {class_name}')
        self.img_id += 1

    def dump(self):
        with open(self.ann_file, 'w') as fw:
            json.dump(self.anns, fw)


def main():
    args = get_args()
    assert osp.isdir(args.root_dir)
    assert osp.exists(osp.join(args.root_dir, "gtFine"))
    assert osp.exists(osp.join(args.root_dir, "leftImg8bit"))

    ann_folder_name = "gt"
    img_folder_name = "images"
    ann_file = osp.join(args.out_dir, args.split, ann_folder_name, "coco-annotations.json")
    index_file_name = "index.txt"

    os.makedirs(osp.join(args.out_dir, args.split, img_folder_name), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, args.split, ann_folder_name), exist_ok=True)
    src_img_files = glob.glob(osp.join(args.root_dir, "leftImg8bit", args.split, "*", "*.png"))

    coco_ann = COCOAnn(ann_file)
    random.seed(25)
    if args.num > 0:
        random.shuffle(src_img_files)
        src_img_files = src_img_files[0:args.num]

    des_img_files = []
    for src_img in tqdm(src_img_files):
        # munster_000001_000019_leftImg8bit.png --> munster_000000_000019_gtFine_polygons.json
        name_splits = Path(src_img).name.split("_")
        name_splits[-1] = "gtFine_polygons.json"
        ann_name = "_".join(name_splits)

        # gtFine/val/munster/munster_000000_000019_gtFine_polygons.json
        ann_json = osp.join(
            args.root_dir,
            "gtFine",
            args.split,
            name_splits[0],  # city
            ann_name,
        )

        label_file = convert_json_to_label(ann_json)

        # make ymir image and annotation with the same filename
        des_img = osp.join(args.out_dir, args.split, img_folder_name, Path(src_img).name)
        shutil.copy(src_img, des_img)
        des_img_files.append(des_img)
        coco_ann.add_ann(des_img, label_file)

    coco_ann.dump()
    with open(osp.join(args.out_dir, args.split, index_file_name), "w") as fp:
        fp.writelines(des_img_files)


if __name__ == "__main__":
    main()
