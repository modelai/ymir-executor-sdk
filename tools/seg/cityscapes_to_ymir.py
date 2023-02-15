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
│   │   ├── labelmap.txt
│   │   └── SegmentationClass [ymir annotation images]
│   ├── images
└── val
    ├── gt
    │   ├── labelmap.txt
    │   └── SegmentationClass [ymir annotation images]
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

# ymir annotation image
root_dir/train/gt/SegmentationClass/munster_000001_000019_leftImg8bit.png
"""
import argparse
import glob
import os
import os.path as osp
import random
import shutil
from pathlib import Path

from cityscapesscripts.helpers.labels import labels  # type: ignore
from cityscapesscripts.preparation.json2labelImg import json2labelImg  # type: ignore
from tqdm import tqdm  # type: ignore


def convert_json_to_label(json_file) -> str:
    """
    from https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py
    """
    # label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    label_file = json_file.replace("_polygons.json", "_color.png")
    if not osp.exists(label_file):
        # json2labelImg(json_file, label_file, 'trainIds')
        json2labelImg(json_file, label_file, "colors")

    return label_file


def get_args():
    parser = argparse.ArgumentParser("convert cityscapes dataset to ymir")
    parser.add_argument("--root_dir", help="root dir for cityscapes dataset")
    parser.add_argument("--split", default="train", help="split for dataset")
    parser.add_argument("--out_dir", help="the output directory", default="./out")
    parser.add_argument("--num", help="sample number for dataset", default=0, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    assert osp.isdir(args.root_dir)
    assert osp.exists(osp.join(args.root_dir, "gtFine"))
    assert osp.exists(osp.join(args.root_dir, "leftImg8bit"))

    ann_folder_name = "gt/SegmentationClass"
    img_folder_name = "images"
    label_map_file = "gt/labelmap.txt"
    index_file_name = "index.txt"

    os.makedirs(osp.join(args.out_dir, args.split, img_folder_name), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, args.split, ann_folder_name), exist_ok=True)
    src_img_files = glob.glob(osp.join(args.root_dir, "leftImg8bit", args.split, "*", "*.png"))

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
        des_ann = osp.join(args.out_dir, args.split, ann_folder_name, Path(src_img).name)
        shutil.copy(src_img, des_img)
        shutil.copy(label_file, des_ann)
        des_img_files.append(des_img)

    with open(osp.join(args.out_dir, args.split, index_file_name), "w") as fp:
        fp.writelines(des_img_files)

    with open(osp.join(args.out_dir, args.split, label_map_file), "w") as fp:
        # labels is ordered
        for x in labels:
            if x.trainId not in [255, -1]:
                r, g, b = x.color
                fp.write(f"{x.name}:{r},{g},{b}::\n")


if __name__ == "__main__":
    main()
