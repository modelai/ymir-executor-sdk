import json
import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import yaml
from deprecated.sphinx import versionadded, versionchanged
from packaging.version import Version
from pydantic import BaseModel

from ymir_exc import env

_MAX_MODEL_STAGES_COUNT_ = 11  # 10 latest stages, 1 best stage


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Annotation(BaseModel):
    class_name: str
    score: float
    box: Box


def multiple_model_stages_supportable() -> bool:
    """
    for ymir>=1.3.0, add new keywords for protocol_version>=1.0.0
    """
    protocol_version = env.get_current_env().protocol_version
    if Version(protocol_version) >= Version("1.0.0"):
        return True
    else:
        ymir_version = os.getenv("YMIR_VERSION", "1.1.0")
        try:
            if Version(ymir_version) >= Version("1.2.0"):
                return True
            else:
                return False
        except Exception as e:
            warnings.warn(f"{e}, unknown YMIR_VERSION {ymir_version}, use 1.1.0 instead")
            return False


@versionchanged(
    version="2.0.0",
    reason="support segmentation metrics and custom metrics",
)
def write_model_stage(stage_name: str,
                      files: List[str],
                      evaluation_result: Dict[str, Union[float, int]] = {},
                      mAP: Optional[float] = None,
                      timestamp: Optional[int] = None,
                      attachments: Optional[Dict[str, List[str]]] = None,
                      evaluate_config: Optional[dict] = None) -> None:
    """
    Write model stage and model attachments
    Args:
        stage_name (str): name to this model stage
        files (List[str]): model file basename names for this stage
            All files should under directory: `/out/models` or `/out/models/stage_name`
        mAP (float): mean average precision of this stage, depracated
        evaluation_result (Dict[str, Union[float, int]]):
            detection example: `{'mAP': 0.65, 'mAR': 0.8, tp: 10, fp: 1, fn: 1}`
            semantic segmentation example: {'mIoU': 0.78, 'mAcc': 0.8, tp: 10, fp: 1, fn: 1}
            instance segmentation example: {'maskAP': 0.6, 'boxAP': 0.8, tp: 10, fp: 1, fn: 1}
            mAP (float, required for object detection): mean average precision
            mIoU (float, required for semantic segmentation): mean intersection over union
            maskAP (float, required for instance segmentation): mask average precision
            mAR (float, optional for object detection): mean average recall
            mAcc (float, optional for semantic segmentation): mean accuracy
            boxAP (float, optional for instance segmentation): box average precision
            tp (int, optional): true positive box count
            fp (int, optional): false positive box count
            fn (int, optional): false negative box count
        timestamp (int): timestamp (in seconds)
        evaluate_config (dict): configurations used to evaluate this model, which contains:
            iou_thr (float): iou threshold
            conf_thr (float): confidence threshold
        attachments: attachment files, All files should under
            directory: `/out/models`
    """
    if not stage_name or not files:
        raise ValueError("empty stage_name or files")

    stage_name = stage_name.replace("-", "_")
    if not stage_name.isidentifier():
        raise ValueError(
            f"invalid stage_name: {stage_name}, need alphabets, numbers and underlines, start with alphabets")

    if not evaluation_result:
        if mAP is None:
            raise Exception('please specify evaluation_result')
        else:
            evaluation_result = {'mAP': mAP}
            warnings.warn('please use evaluation_result instead of mAP')

    if 'maskAP' in evaluation_result:
        top1_metric = 'maskAP'
    elif 'mIoU' in evaluation_result:
        top1_metric = 'mIoU'
    elif 'mAP' in evaluation_result:
        top1_metric = 'mAP'
    else:
        raise Exception(f'unknown evaluation_result {evaluation_result}, without one of [maskAP, mIoU, mAP]')

    training_result: dict = ({})  # key: stage name, value: stage name, files, timestamp, mAP

    env_config = env.get_current_env()
    try:
        with open(env_config.output.training_result_file, "r") as f:
            training_result = yaml.safe_load(stream=f)
    except FileNotFoundError:
        pass  # will create new if not exists, so dont care this exception

    if multiple_model_stages_supportable():
        model_stages = training_result.get("model_stages", {})
        # stage_name --> intermediate
        model_stages[stage_name] = {
            "stage_name": stage_name,
            "files": files,
            "timestamp": timestamp or int(time.time()),
            **evaluation_result,
        }

        # best stage
        sorted_model_stages = sorted(
            model_stages.values(),
            key=lambda x: (x.get(top1_metric, 0), x.get("timestamp", 0)),
        )
        training_result["best_stage_name"] = sorted_model_stages[-1]["stage_name"]
        training_result[top1_metric] = sorted_model_stages[-1][top1_metric]

        # if too many stages, remove a earlest one
        if len(model_stages) > _MAX_MODEL_STAGES_COUNT_:
            sorted_model_stages = sorted(model_stages.values(), key=lambda x: x.get("timestamp", 0))
            del_stage_name = sorted_model_stages[0]["stage_name"]
            if del_stage_name == training_result["best_stage_name"]:
                del_stage_name = sorted_model_stages[1]["stage_name"]
            del model_stages[del_stage_name]
            logging.info(f"data_writer removed model stage: {del_stage_name}")
        training_result["model_stages"] = model_stages

        # attachments, replace old value if valid
        if attachments:
            training_result["attachments"] = attachments

        # evaluate config
        if evaluate_config:
            training_result['evaluate_config'] = evaluate_config
    else:
        warnings.warn("mutiple model stages is not supported, use write_training_result() instead")
        _files = training_result.get("model", [])

        training_result = {
            "model": list(set(files + _files)),
            "timestamp": timestamp or int(time.time()),
            "stage_name": stage_name,
            **evaluation_result
        }

    # save all
    with open(env_config.output.training_result_file, "w") as f:
        yaml.safe_dump(data=training_result, stream=f)


@versionadded(
    version="2.1.0",
    reason="support segmentation metrics and custom metrics",
)
def write_training_result(stage_name: str,
                          files: List[str],
                          evaluation_result: Dict[str, Union[float, int]] = {},
                          timestamp: Optional[int] = None,
                          attachments: Optional[Dict[str, List[str]]] = None,
                          evaluate_config: Optional[dict] = None) -> None:
    """write training result to training result file

    if exist result in training result file, new result will append to file
    """
    write_model_stage(stage_name=stage_name,
                      files=files,
                      evaluation_result=evaluation_result,
                      timestamp=timestamp,
                      attachments=attachments,
                      evaluate_config=evaluate_config)


def write_mining_result(mining_result: List[Tuple[str, float]]) -> None:
    """write mining result to mining_result_file

    Parameters
    ----------
    mining_result : List[Tuple[str, float]]
        - image_path: str
        - score: float
    """
    # sort desc by score
    sorted_mining_result = sorted(mining_result, reverse=True, key=(lambda v: v[1]))

    env_config = env.get_current_env()
    with open(env_config.output.mining_result_file, "w") as f:
        for img_path, score in sorted_mining_result:
            f.write(f"{img_path}\t{score}\n")


@versionchanged(
    version="2.0.0",
    reason="support detection and segmentation infer result",
)
def write_infer_result(infer_result: Union[Dict, Dict[str, List[Annotation]]],
                       algorithm: Optional[str] = 'detection') -> None:
    """
    supported_algorithms = ['classification', 'detection', 'segmentation']
    for detection infer result, use ymir format
    for segmentation infer result, write coco-format to infer_result_file directly
    """
    env_config = env.get_current_env()
    supported_algorithms = ['classification', 'detection', 'segmentation']
    assert algorithm in supported_algorithms, f'unknown {algorithm}, not in {supported_algorithms}'

    if algorithm == 'detection':
        protocol_version = env_config.protocol_version
        # from ymir1.3.0, keyword change from annotations to boxes
        if Version(protocol_version) >= Version("1.0.0"):
            keyword = "boxes"
        else:
            keyword = "annotations"

        detection_result = {}
        for asset_path, annotations in infer_result.items():
            asset_basename = os.path.basename(asset_path)
            detection_result[asset_basename] = {keyword: [annotation.dict() for annotation in annotations]}

        result = {"detection": detection_result}
        with open(env_config.output.infer_result_file, "w") as f:
            f.write(json.dumps(result))
    elif algorithm == 'segmentation':
        with open(env_config.output.infer_result_file, 'w') as f:
            f.write(json.dumps(infer_result))
    else:
        raise Exception(f'not implement {algorithm} infer result')
