from typing import Iterator, List, Tuple

from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm

from ymir_exc import env


def _index_file_for_dataset_type(env_config: env.EnvConfig, dataset_type: env.DatasetType) -> str:
    mapping = {
        env.DatasetType.TRAINING: env_config.input.training_index_file,
        env.DatasetType.VALIDATION: env_config.input.val_index_file,
        env.DatasetType.CANDIDATE: env_config.input.candidate_index_file,
    }
    return mapping[dataset_type]


def item_paths(dataset_type: env.DatasetType) -> Iterator[Tuple[str, str]]:
    file_path = _index_file_for_dataset_type(env.get_current_env(), dataset_type)
    if not file_path:
        raise ValueError(f"index file not set for dataset: {dataset_type}")

    with open(file_path, "r") as f:
        for line in f:
            # note: last char of line is \n
            components = line.strip().split("\t")
            if len(components) >= 2:
                yield (components[0], components[1])
            elif len(components) == 1:
                yield (components[0], "")
            else:
                # ignore empty lines
                continue


def items_count(dataset_type: env.DatasetType) -> int:
    file_path = _index_file_for_dataset_type(env.get_current_env(), dataset_type)
    if not file_path:
        raise ValueError(f"index file not set for dataset: {dataset_type}")

    with open(file_path, "r") as f:
        return len(f.readlines())


def images_count(cfg: edict, split: str) -> int:
    """return the image number in dataset

    Parameters
    ----------
    cfg : edict
        ymir merged config
    split : str
        for training task: support [train/training, val/validation]
        for infer/mining task: support [test/infer/candidate]
    Returns
    -------
    int
        the image number in dataset

    Raises
    ------
    Exception
        unknown split name
    """
    if split in ['training', 'train']:
        index_file = cfg.ymir.input.training_index_file
    elif split in ['val', 'validation']:
        index_file = cfg.ymir.input.val_index_file
    elif split in ['candidate', 'test', 'infer']:
        index_file = cfg.ymir.input.candidate_index_file
    else:
        raise Exception(f'unknown split {split}, not in [train/training, val/validation, test/infer/candidate]')

    with open(index_file, 'r') as fp:
        lines = fp.readlines()

    return len(lines)


def bboxes_count(cfg: edict, split: str) -> int:
    """return the bounding box number in dataset annotations

    Parameters
    ----------
    cfg : edict
        ymir merged config
    split : str
        for training task: support [train/training, val/validation]
        not support mining/infer task

    Returns
    -------
    int
        the bounding box number in dataset annotations

    Raises
    ------
    Exception
        unknown split
    Exception
        unknown export_format
    """
    if split in ['training', 'train']:
        index_file = cfg.ymir.input.training_index_file
    elif split in ['val', 'validation']:
        index_file = cfg.ymir.input.val_index_file
    else:
        raise Exception(f'unknown split {split}, not in [train/training, val/validation]')

    with open(index_file, 'r') as fp:
        lines = fp.readlines()

    total_bbox = 0
    if cfg.param.export_format in ['ark:raw', 'det-ark:raw']:
        for line in tqdm(lines):
            img_path, txt_path = line.split()
            with open(txt_path, 'r') as fq:
                num = len(fq.readlines())

            total_bbox += num
    else:
        raise Exception(f'unsupport export_format {cfg.param.export_format}')
    return total_bbox


def filter_broken_images(image_files: List[str]) -> List[str]:
    """
    filter out the broken image files
    return readable image
    """
    normal_image_files = []
    for img_f in image_files:
        try:
            img = Image.open(img_f)
            img.verify()
            normal_image_files.append(img_f)
        except Exception as e:
            print(f"bad img file {img_f}: {e}")

    return normal_image_files
