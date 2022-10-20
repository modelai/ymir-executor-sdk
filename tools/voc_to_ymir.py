"""
you can import voc dataset to ymir directly
this script help you to import train/val alone to ymir

suppose we have voc dataset as follow:
VOCdevkit
├── VOC2007
│   ├── Annotations
│   ├── ImageSets
│   │   ├── Layout
│   │   ├── Main
│   │   └── Segmentation
│   ├── JPEGImages
│   ├── SegmentationClass
│   └── SegmentationObject
└── VOC2012
    ├── Annotations
    ├── ImageSets
    │   ├── Action
    │   ├── Layout
    │   ├── Main
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject
"""
import argparse
import os
import os.path as osp
import shutil

from tqdm import tqdm  # type: ignore


def get_args():
    parser = argparse.ArgumentParser('split voc dataset to ymir')
    parser.add_argument('--root_dir', help='root dir for voc devkit')
    parser.add_argument('--index_file',
                        default='VOC2012/ImageSets/Main/val.txt',
                        help='index file for import image and labels')
    parser.add_argument('--out_dir', help='the output directory', default='./out')

    return parser.parse_args()


def main():
    args = get_args()
    assert osp.isdir(args.root_dir)
    assert osp.exists(osp.join(args.root_dir, args.index_file))

    if args.index_file.find('VOC2012') > -1:
        voc_year = 'VOC2012'
    elif args.index_file.find('VOC2007') > -1:
        voc_year = 'VOC2007'
    else:
        assert False, f'unknown format {args.index_file}'

    assert osp.exists(osp.join(args.root_dir, voc_year))
    des_dataset_name = osp.splitext(osp.basename(args.index_file))[0]
    out_root_dir = osp.join(args.out_dir, voc_year, des_dataset_name)
    os.makedirs(out_root_dir, exist_ok=True)
    os.makedirs(osp.join(out_root_dir, 'images'), exist_ok=True)

    ann_folder_name = 'gt'  # for ymir>=1.2.2, use gt instead of annotations
    os.makedirs(osp.join(out_root_dir, ann_folder_name), exist_ok=True)

    with open(osp.join(args.root_dir, args.index_file), 'r') as fp:
        lines = fp.readlines()
    for line in tqdm(lines):
        line_splits = line.strip().split()
        if len(line_splits) == 1:
            pass
        elif len(line_splits) == 2:
            if line_splits[1] == '-1':
                continue
            elif line_splits[1] == '1':
                pass
            else:
                print(f'unknown format {line}')
                continue
        else:
            print(f'unknown length of line: {line}')
            continue

        basename = line_splits[0]

        src_img_file = osp.join(args.root_dir, voc_year, 'JPEGImages', basename + '.jpg')
        des_img_file = osp.join(args.out_dir, voc_year, des_dataset_name, 'images', basename + '.jpg')
        if osp.exists(src_img_file) and not osp.exists(des_img_file):
            shutil.copy(src_img_file, des_img_file)
        else:
            print(f'not found {src_img_file} or exist {des_img_file}')

        src_xml_file = osp.join(args.root_dir, voc_year, 'Annotations', basename + '.xml')
        des_xml_file = osp.join(args.out_dir, voc_year, des_dataset_name, ann_folder_name, basename + '.xml')
        if osp.exists(src_xml_file) and not osp.exists(des_xml_file):
            shutil.copy(src_xml_file, des_xml_file)
        else:
            print(f'not found {src_xml_file} or exist {des_xml_file}')


if __name__ == '__main__':
    main()
