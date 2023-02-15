import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import imagesize
import yaml
from easydict import EasyDict as edict
from packaging.version import Version
from tqdm import tqdm


def img2label_paths(img_paths):
    """
    copy from https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
    """
    # Define label paths as a function of image paths
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def convert(cfg: edict, out_dir: str, asset: str, ann: str) -> Tuple[str, str]:
    """
    convert ymir annotations to yolov5 annotations
    input:
        out_dir: /out
        asset: /in/assets/xx/xxx.jpg
        ann: /in/annotations/xx/xxx.txt
    output:
        img: /out/images/xx/xxx.jpg
        txt: /out/labels/xx/xxx.txt
    """
    protocol_version = cfg.ymir.protocol_version
    if Version(protocol_version) >= Version("1.0.0"):
        ymir_dataset_format = cfg.param.get("export_format", "det-voc:raw")
    else:
        ymir_dataset_format = cfg.param.get("export_format", "det-ark:raw")

    ann_dir = cfg.ymir.input.annotations_dir
    asset_dir = cfg.ymir.input.assets_dir
    img_dir = osp.join(out_dir, "images")
    txt_dir = osp.join(out_dir, "labels")

    img = asset.replace(asset_dir, img_dir)
    txt = ann.replace(ann_dir, txt_dir)

    os.makedirs(osp.dirname(txt), exist_ok=True)

    width, height = imagesize.get(asset)
    if ymir_dataset_format in ["ark:raw", "det-ark:raw"]:
        # ymir: class_id: int, xmin: int, ymin: int, xmax: int, ymax: int, bbox_quality: float
        # yolov5: class_id: int, x_center: float, y_center: float, width: float, height: float
        txt_lines = []
        with open(ann, "r") as fp:
            lines = fp.readlines()

        for line in lines:
            class_id, xmin, ymin, xmax, ymax = [int(x) for x in line.split(",")[0:5]]
            xc = (xmax + xmin) / 2 / width
            yc = (ymax + ymin) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            txt_lines.append(f"{class_id} {xc} {yc} {w} {h}")

        with open(txt, "w") as fw:
            fw.write("\n".join(txt_lines))
    elif ymir_dataset_format in ["det-voc:raw", "voc:raw"]:
        raise Exception(f"unknown dataset format {ymir_dataset_format}")
    else:
        raise Exception(f"unknown dataset format {ymir_dataset_format}")

    return img, txt


def convert_ymir_to_yolov5(cfg: edict, out_dir: str = None) -> str:
    """
    convert ymir format dataset to yolov5 format
    generate data.yaml for training/mining/infer
    return data.yaml file path

    / # ymir format
    ├── in
    │   ├── annotations
    │   ├── assets
    │   ├── config.yaml
    │   ├── env.yaml
    │   ├── train-index.tsv
    │   └── val-index.tsv

    / # yolov5 format
    ├── out
    │   ├── images # ln -s /in/assets /out/images
    │   ├── labels
    │   ├── train-index.txt
    │   └── val-index.txt
    """

    out_dir = out_dir or cfg.ymir.output.root_dir
    if cfg.ymir.run_training:
        Path(osp.join(out_dir, "images")).symlink_to(cfg.ymir.input.assets_dir)
    data = dict(path=out_dir, nc=len(cfg.param.class_names), names=cfg.param.class_names)
    for split, prefix in zip(["train", "val", "test"], ["training", "val", "candidate"]):
        src_file = getattr(cfg.ymir.input, f"{prefix}_index_file")
        if osp.exists(src_file) and split in ["train", "val"]:
            with open(src_file, "r") as fp:
                lines = fp.readlines()

            img_files: List[str] = []
            for line in tqdm(lines):
                asset, ann = line.split()
                img, _ = convert(cfg, out_dir, asset, ann)
                img_files.append(img)

            split_txt = osp.join(out_dir, f"{split}-index.txt")
            with open(split_txt, "w") as fw:
                fw.write("\n".join(img_files))

        data[split] = f"{split}-index.txt"

    data_yaml = osp.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as fw:
        fw.write(yaml.safe_dump(data))

    return data_yaml
