import glob
import json
import os
import os.path as osp
import socket
import warnings
from enum import IntEnum
from typing import Dict, List, Tuple

import imagesize
import yaml
from easydict import EasyDict as edict
from packaging.version import Version

from ymir_exc import env
from ymir_exc import result_writer as rw


def find_free_port():
    """
    code from detectron2: https://github.com/facebookresearch/detectron2
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


class YmirStage(IntEnum):
    PREPROCESS = 1  # convert dataset
    TASK = 2    # training/mining/infer
    POSTPROCESS = 3  # export model


def get_ymir_process(stage: YmirStage, p: float, task_idx: int = 0, task_num: int = 1) -> float:
    """
    stage: pre-process/task/post-process
    p: percent for stage
    task_idx: index for multiple tasks like mining (task_idx=0) and infer (task_idx=1)
    task_num: the total number of multiple tasks.
    """
    # const value for ymir process
    PREPROCESS_PERCENT = int(os.getenv('PREPROCESS_PERCENT',0.1))
    TASK_PERCENT = int(os.getenv('TASK_PERCENT',0.8))
    POSTPROCESS_PERCENT = int(os.getenv('POSTPROCESS_PERCENT',0.1))

    if p < 0 or p > 1.0:
        raise Exception(f'p not in [0,1], p={p}')

    ratio = 1.0 / task_num
    init = task_idx * ratio
    if stage == YmirStage.PREPROCESS:
        return init + PREPROCESS_PERCENT * p * ratio
    elif stage == YmirStage.TASK:
        return init + (PREPROCESS_PERCENT + TASK_PERCENT * p) * ratio
    elif stage == YmirStage.POSTPROCESS:
        return init + (PREPROCESS_PERCENT + TASK_PERCENT + POSTPROCESS_PERCENT * p) * ratio
    else:
        raise NotImplementedError(f'unknown stage {stage}')


def get_merged_config() -> edict:
    """
    merge ymir_config and executor_config
    """
    merged_cfg = edict()
    # the hyperparameter information
    merged_cfg.param = env.get_executor_config()

    # the ymir path information
    merged_cfg.ymir = env.get_current_env()
    return merged_cfg


def convert_ymir_to_coco(cat_id_from_zero=False):
    """
    convert ymir dataset to coco format for training task
    cat_id_from_zero: category id start from zero or not
    ymir_index_file: for each line it likes: {img_path} \t {ann_path}
    output the coco dataset information
    """
    cfg = get_merged_config()
    out_dir = cfg.ymir.output.root_dir
    # os.environ.setdefault('DETECTRON2_DATASETS', out_dir)
    ymir_dataset_dir = osp.join(out_dir, 'ymir_dataset')
    os.makedirs(ymir_dataset_dir, exist_ok=True)

    output_info = {}
    for split, prefix in zip(['train', 'val'], ['training', 'val']):
        src_file = getattr(cfg.ymir.input, f'{prefix}_index_file')
        with open(src_file) as fp:
            lines = fp.readlines()

        img_id = 0
        ann_id = 0
        data: Dict[str, List] = dict(images=[],
                                     annotations=[],
                                     categories=[],
                                     licenses=[]
                                     )

        cat_id_start = 0 if cat_id_from_zero else 1
        for id, name in enumerate(cfg.param.class_names):
            data['categories'].append(dict(id=id + cat_id_start,
                                           name=name,
                                           supercategory="none"))

        for line in lines:
            img_file, ann_file = line.strip().split()
            width, height = imagesize.get(img_file)
            img_info = dict(file_name=img_file,
                            height=height,
                            width=width,
                            id=img_id)

            data['images'].append(img_info)

            if osp.exists(ann_file):
                for ann_line in open(ann_file, 'r').readlines():
                    ann_strlist = ann_line.strip().split(',')
                    class_id, x1, y1, x2, y2 = [
                        int(s) for s in ann_strlist[0:5]]
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    bbox_quality = float(ann_strlist[5]) if len(
                        ann_strlist) > 5 and ann_strlist[5].isnumeric() else 1
                    ann_info = dict(bbox=[x1, y1, bbox_width, bbox_height],   # x,y,width,height
                                    area=bbox_area,
                                    score=1.0,
                                    bbox_quality=bbox_quality,
                                    iscrowd=0,
                                    segmentation=[
                                        [x1, y1, x1, y2, x2, y2, x2, y1]],
                                    category_id=class_id + cat_id_start,   # start from cat_id_start
                                    id=ann_id,
                                    image_id=img_id)
                    data['annotations'].append(ann_info)
                    ann_id += 1

            img_id += 1

        split_json_file = osp.join(ymir_dataset_dir, f'ymir_{split}.json')
        with open(split_json_file, 'w') as fw:
            json.dump(data, fw)

        output_info[split] = dict(img_dir=cfg.ymir.input.assets_dir,
                                  ann_file=split_json_file)

    return output_info

def get_weight_files(cfg: edict, suffix: Tuple[str, ...]=('.pt','.pth')) -> List[str]:
    """
    return the weight file path by priority
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path
    """
    if cfg.ymir.run_training:
        model_params_path = cfg.param.get('pretrained_model_params', [])
    else:
        model_params_path = cfg.param.model_params_path

    model_dir = cfg.ymir.input.models_dir
    model_params_path = [osp.join(model_dir, p)
                         for p in model_params_path if osp.exists(osp.join(model_dir, p)) and p.endswith(suffix)]

    return model_params_path


def write_ymir_training_result(cfg: edict, map50: float, files: List[str], id: str) -> None:
    """
    cfg: ymir merged config, view get_merged_config()
    map50: evaluation result
    files: weight and related files to save, [] means save all files in /out/models
    id: weight name to distinguish models from different epoch/step
    """
    if not files and map50 > 0:
        warnings.warn(f'map50 = {map50} > 0 when save all files')

    YMIR_VERSION = os.getenv('YMIR_VERSION', '1.1.0')
    if Version(YMIR_VERSION) >= Version('1.2.0'):
        _write_latest_ymir_training_result(cfg, float(map50), id, files)
    else:
        _write_ancient_ymir_training_result(cfg, float(map50), id, files)

def _write_latest_ymir_training_result(cfg: edict,
                                       map50: float,
                                       id: str,
                                       files: List[str]) -> None:
    """
    for ymir>=1.2.0
    """
    # use `rw.write_training_result` to save training result
    if files:
        rw.write_model_stage(stage_name=id,
                             files=[osp.basename(f) for f in files],
                             mAP=map50)
    else:
        # save other files with best map50, use relative path, filter out directory.
        root_dir = cfg.ymir.output.models_dir
        files = [osp.relpath(f, start=root_dir) for f in glob.glob(osp.join(root_dir, '**','*'), recursive=True) if osp.isfile(f)]

        training_result_file = cfg.ymir.output.training_result_file
        if osp.exists(training_result_file):
            with open(training_result_file, 'r') as f:
                training_result = yaml.safe_load(stream=f)

            map50 = max(training_result.get('map', 0.0), map50)
        rw.write_model_stage(stage_name=id,
                             files=files,
                             mAP=map50)


def _write_ancient_ymir_training_result(cfg: edict,
        map50: float,
        id: str,
        files: List[str]) -> None:
    """
    for 1.0.0 <= ymir <=1.1.0
    """

    if not files:
        # save other files with best map50, use relative path, filter out directory.
        root_dir = cfg.ymir.output.models_dir
        files = [osp.relpath(f, start=root_dir) for f in glob.glob(osp.join(root_dir, '**','*'), recursive=True) if osp.isfile(f)]
    training_result_file = cfg.ymir.output.training_result_file
    if osp.exists(training_result_file):
        with open(training_result_file, 'r') as f:
            training_result = yaml.safe_load(stream=f)

        if training_result is None:
            training_result = {}
        training_result['model'] = files
        max_map50 = max(training_result.get('map', 0), map50)
        training_result['map'] = max_map50

        # when save other files like onnx model, we cannot obtain map50, set map50=0 to use the max_map50
        training_result[id] = map50 if map50 > 0 else max_map50
    else:
        training_result = {
            'model': files,
            'map': map50,
            id: map50
        }

    with open(training_result_file, 'w') as f:
        yaml.safe_dump(training_result, f)
