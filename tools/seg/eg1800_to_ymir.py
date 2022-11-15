"""
thanks for
https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/paddleseg/datasets/eg1800.py
"""
import argparse
import os
import os.path as osp
import random
import shutil

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('convert eg1800 dataset to ymir')
    parser.add_argument('--root_dir', help='root dir for eg1800 dataset')
    parser.add_argument('--split', default='train', help='split for dataset', choices=['train', 'test'])
    parser.add_argument('--out_dir', help='the output directory', default='./out')
    parser.add_argument('--num', help='sample number for dataset', default=0, type=int)

    return parser.parse_args()


def main():
    args = get_args()

    assert osp.isdir(args.root_dir)
    assert osp.exists(osp.join(args.root_dir, 'Images'))
    assert osp.exists(osp.join(args.root_dir, 'Labels'))

    ann_folder_name = 'gt/SegmentationClass'
    img_folder_name = 'images'
    label_map_file = 'gt/labelmap.txt'
    index_file_name = f'eg1800_{args.split}.txt'

    os.makedirs(osp.join(args.out_dir, args.split, img_folder_name), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, args.split, ann_folder_name), exist_ok=True)
    with open(osp.join(args.root_dir, index_file_name), 'r') as fp:
        lines = fp.readlines()

    if args.num > 0:
        lines = lines[0:args.num]

    random.seed(25)
    for line in tqdm(lines):
        filename = line.strip()
        src_img = osp.join(args.root_dir, 'Images', filename)
        src_ann = osp.join(args.root_dir, 'Labels', filename)

        des_img = osp.join(args.out_dir, args.split, img_folder_name, filename)
        des_ann = osp.join(args.out_dir, args.split, ann_folder_name, filename)

        shutil.copy(src_img, des_img)
        shutil.copy(src_ann, des_ann)

    # write labelmap
    with open(osp.join(args.out_dir, args.split, label_map_file), 'w') as fw:
        fw.write('bg:0,0,0::\n')
        fw.write('fg:1,1,1::\n')


if __name__ == '__main__':
    main()
