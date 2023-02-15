import glob
import json
import math
import os
import os.path as osp
import re
import socket
import warnings
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import imagesize
import yaml
from deprecated.sphinx import deprecated, versionadded, versionchanged
from easydict import EasyDict as edict

from ymir_exc import env
from ymir_exc import result_writer as rw
from ymir_exc.monitor import YmirTask, write_monitor_logger_for_multiple_tasks


def find_free_port():
    """find free port for DDP
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
    TASK = 2  # training/mining/infer
    POSTPROCESS = 3  # export model


class YmirStageWeight(object):

    def __init__(self, weights: Optional[List[float]] = None):
        """
        weights: weight for each ymir stage
        if weights is None:
            self.weights = [0, 0, 0]
            self.weights[0] = float(os.getenv('PREPROCESS_WEIGHT', 0.00001))
            self.weights[1] = float(os.getenv('TASK_WEIGHT', 0.99998))
            self.weights[2] = float(os.getenv('POSTPROCESS_WEIGHT', 0.00001))
        """
        if weights:
            self.weights = weights
        else:
            self.weights = [0, 0, 0]
            self.weights[0] = float(os.getenv("PREPROCESS_WEIGHT", 0.00001))
            self.weights[1] = float(os.getenv("TASK_WEIGHT", 0.99998))
            self.weights[2] = float(os.getenv("POSTPROCESS_WEIGHT", 0.00001))

        assert math.isclose(sum(self.weights), 1), f"sum of weights {weights} != 1"
        assert len(self.weights) == 3, f"len of weights {weights} != 3"

    def get_stage_process(self, stage: Union[YmirStage, str], p: float) -> float:
        """return the stage process for a task, range in [0, 1]
        for preprocess stage:
            return process range in [0, self.weight[0]]
        for task stage:
            return process range in [self.weights[0], self.weight[0]+self.weight[1]]
        for postprocess stage:
            return process range in [self.weight[0]+self.weight[1], 1]
        """
        if stage in [YmirStage.PREPROCESS, "preprocess"]:
            return self.weights[0] * p
        elif stage in [YmirStage.TASK, "task"]:
            return self.weights[0] + self.weights[1] * p
        elif stage in [YmirStage.POSTPROCESS, "postprocess"]:
            return self.weights[0] + self.weights[1] + self.weights[2] * p
        else:
            raise NotImplementedError(f"unknown stage {stage}")


@deprecated(
    version="1.3.1",
    reason="This method is deprecated, recommand use all-in-on function write_ymir_monitor_process() instead",
)
def get_ymir_process(
    stage: Union[YmirStage, str],
    p: float,
    task_idx: int = 0,
    task_num: int = 1,
    weights: Optional[YmirStageWeight] = None,
) -> float:
    """return the process for ymir, range in [0,1]
    stage: pre-process/task/post-process
    p: naive percent for stage, range in [0,1]
    task_idx: index for multiple tasks like mining (task_idx=0) and infer (task_idx=1)
    task_num: the total number of multiple tasks.

    for single task:
        task_idx = 0, task_num = 1
        return process range in [0, 1]

    for multiple tasks:
        the first task: task_idx = 0, task_num = 2
        return the first process range in [0, 0.5]
        the second task: task_idx = 1, task_num = 2
        return the second process range in [0.5, 1]
    """
    if weights is None:
        weights = YmirStageWeight()

    if p < 0 or p > 1.0:
        raise Exception(f"p not in [0,1], p={p}")

    task_ratio = 1.0 / task_num
    task_init = task_idx * task_ratio
    return task_init + task_ratio * weights.get_stage_process(stage, p)


def get_merged_config() -> edict:
    """return all config for ymir
    view https://github.com/modelai/ymir-executor-fork/wiki/input-(-in)-and-output-(-out)-for-docker-image for detail
    merged_cfg.param: read from /in/config.yaml and code_config, code_config will be overwritten by /in/config.yaml.
    merged_cfg.ymir: read from /in/env.yaml
    """

    def get_code_config(code_config_file: str) -> dict:
        if code_config_file:
            with open(code_config_file, "r") as f:
                return yaml.safe_load(f)
        else:
            return dict()

    merged_cfg = edict()

    # the hyperparameter information
    exe_cfg = env.get_executor_config()
    if exe_cfg.get("git_url", ""):
        # live code mode
        code_config_file = exe_cfg.get("code_config", "")
        code_cfg = get_code_config(code_config_file)
        code_cfg.update(exe_cfg)

        merged_cfg.param = code_cfg
    else:
        # normal mode
        merged_cfg.param = exe_cfg

    # the ymir path/env information
    merged_cfg.ymir = env.get_current_env()
    return merged_cfg


def convert_ymir_to_coco(cat_id_from_zero: bool = False) -> Dict[str, Dict[str, str]]:
    """
    convert ymir detection dataset to coco format for training task
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


def get_weight_files(cfg: edict, suffix: Tuple[str, ...] = (".pt", ".pth")) -> List[str]:
    """
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path with `suffix`
    return the weight file list
    """
    if cfg.ymir.run_training:
        model_params_path = cfg.param.get("pretrained_model_params", [])
    else:
        model_params_path = cfg.param.model_params_path

    model_dir = cfg.ymir.input.models_dir
    model_params_path = [
        osp.join(model_dir, p) for p in model_params_path if osp.exists(osp.join(model_dir, p)) and p.endswith(suffix)
    ]

    return model_params_path


def get_bool(cfg: edict, key: str, default_value: bool = True) -> bool:
    """get bool hyper-parameter from ymir merged config, return default_value if not defined.
    the value for key in cfg may str, int or bool
        str will ignore case
        str: f, F, false, False, 0 will return False
        str: t, T, true, True, 1 will return True
        int: 0 will return 0
        int: 1 will return 1
        bool: True will return True
        bool: False will return False
        other str or int will raise Exception
    return bool
    """
    v = cfg.param.get(key, default_value)

    if isinstance(v, str):
        if v.lower() in ["f", "false", "0"]:
            return False
        elif v.lower() in ["t", "true", "1"]:
            return True
        else:
            raise Exception(f"unknown bool str {key} = {v}")
    elif isinstance(v, int):
        if v in [0, 1]:
            return bool(v)
        else:
            raise Exception(f"unknown bool int {key} = {v}")
    elif isinstance(v, bool):
        return v
    else:
        raise Exception(f"unknown bool type {key} = {v} ({type(v)})")


@versionadded(
    version="1.3.1",
    reason="support user custom by hyper-parameter: ymir_saved_file_patterns",
)
def filter_saved_files(cfg: edict, files: List[str]):
    """
    if files == []:
        if ymir_saved_file_patterns:
            return "filterd files in cfg.ymir.output.models_dir"
        else:
            return "all file in cfg.ymir.output.models_dir"
    else:
        if ymir_saved_file_patterns:
            return "filterd files"
        else:
            return files

    return filtered relpath
    """
    ymir_saved_file_patterns: str = cfg.param.get("ymir_saved_file_patterns", "")

    root_dir = cfg.ymir.output.models_dir
    if not files:
        root_dir = cfg.ymir.output.models_dir
        files = [osp.relpath(f, start=root_dir) for f in glob.glob(osp.join(root_dir, "*")) if osp.isfile(f)]
    else:
        files = [osp.relpath(f, start=root_dir) if osp.isabs(f) else f for f in files]

    if ymir_saved_file_patterns:
        patterns: List[str] = ymir_saved_file_patterns.split(",")
        custom_saved_files = []

        for f in files:
            for pattern in patterns:
                try:
                    if re.search(pattern=pattern.strip(), string=f) is not None:
                        custom_saved_files.append(f)
                        break
                except Exception as e:
                    warnings.warn(f"bad python regular expression pattern {pattern} with {e}")
                    patterns.remove(pattern)

        return custom_saved_files
    else:
        # ymir not support absolute path
        return files


@versionchanged(
    version="2.0.0",
    reason="add support for segmentation",
)
def write_ymir_training_result(
    cfg: edict,
    files: List[str],
    id: str,
    map50: float,
    attachments: Optional[Dict[str, List[str]]] = None,
) -> None:
    """write training result to disk for ymir
    cfg: ymir merged config, view get_merged_config()
    map50: evaluation result
    files: weight and related files to save, [] means save all files in /out/models
    id: weight name to distinguish models from different epoch/step
    attachments: attachment files, All files should under
        directory: `/out/models`
    """
    files = filter_saved_files(cfg, files)

    if files:
        if rw.multiple_model_stages_supportable():
            if id.isnumeric():
                warnings.warn(f"use stage_{id} instead {id} for stage name")
                id = f"stage_{id}"
            _write_latest_ymir_training_result(cfg=cfg, map50=map50, id=id, files=files, attachments=attachments)
        else:
            if map50:
                _write_earliest_ymir_training_result(cfg, float(map50), id, files)
            else:
                raise Exception('old ymir not support evaluation result')


def _write_latest_ymir_training_result(
    cfg: edict,
    id: str,
    files: List[str],
    map50: float,
    attachments: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    for ymir>=1.2.0
    """
    rw.write_model_stage(stage_name=id, files=files, mAP=map50, attachments=attachments)


def _write_earliest_ymir_training_result(cfg: edict, map50: float, id: str, files: List[str]) -> None:
    """
    for 1.0.0 <= ymir <=1.1.0
    """

    training_result_file = cfg.ymir.output.training_result_file
    if osp.exists(training_result_file):
        with open(training_result_file, "r") as f:
            training_result = yaml.safe_load(stream=f)

        if training_result is None:
            training_result = {}
        training_result["model"] = files
        max_map50 = max(training_result.get("map", 0), map50)
        training_result["map"] = max_map50

        if 0 < map50 < max_map50:
            warnings.warn(f"map50 = {map50} < max_map50 = {max_map50} when save all files, ignore map50")
        # when save other files like onnx model, we cannot obtain map50, set map50=0 to use the max_map50
        training_result[id] = map50 if map50 > 0 else max_map50
    else:
        training_result = {"model": files, "map": map50, id: map50}

    with open(training_result_file, "w") as f:
        yaml.safe_dump(training_result, f)


def write_ymir_monitor_process(
    cfg: edict,
    task: Union[YmirTask, str],
    naive_stage_percent: float,
    stage: Union[YmirStage, str],
    stage_weights: Optional[Union[YmirStageWeight, List[float]]] = None,
    task_order: str = "tmi",
) -> None:
    """all in one process monitor function
    task: training, infer or mining
    stage: pre-process, task or post-process
    naive_stage_percent: [0, 1] percent for a specific stage
    weights: weights for each stage
    """

    if stage_weights is None or isinstance(stage_weights, List):
        stage_weights = YmirStageWeight(weights=stage_weights)

    if naive_stage_percent < 0 or naive_stage_percent > 1.0:
        raise Exception(f"p not in [0,1], naive stage percent={naive_stage_percent}")

    naive_task_percent = stage_weights.get_stage_process(stage, naive_stage_percent)
    write_monitor_logger_for_multiple_tasks(cfg, task, naive_task_percent, task_order)
