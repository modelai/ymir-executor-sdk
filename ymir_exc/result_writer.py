import json
import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import yaml
from deprecated.sphinx import deprecated, versionchanged
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


def write_model_stage(stage_name: str,
                      files: List[str],
                      mAP: float,
                      timestamp: Optional[int] = None,
                      attachments: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Write model stage and model attachments
    Args:
        stage_name (str): name to this model stage
        files (List[str]): model file names for this stage
            All files should under directory: `/out/models`
        mAP (float): mean average precision of this stage
        timestamp (int): timestamp (in seconds)
        attachments: attachment files, All files should under
            directory: `/out/models`
    """
    if not stage_name or not files:
        raise ValueError('empty stage_name or files')

    stage_name = stage_name.replace('-', '_')
    if not stage_name.isidentifier():
        raise ValueError(
            f"invalid stage_name: {stage_name}, need alphabets, numbers and underlines, start with alphabets")

    training_result: dict = {}  # key: stage name, value: stage name, files, timestamp, mAP

    env_config = env.get_current_env()
    try:
        with open(env_config.output.training_result_file, 'r') as f:
            training_result = yaml.safe_load(stream=f)
    except FileNotFoundError:
        pass  # will create new if not exists, so dont care this exception

    if multiple_model_stages_supportable():
        model_stages = training_result.get('model_stages', {})
        # stage_name --> intermediate
        model_stages[stage_name] = {
            'stage_name': stage_name,
            'files': files,
            'timestamp': timestamp or int(time.time()),
            'mAP': mAP
        }

        # best stage
        sorted_model_stages = sorted(model_stages.values(), key=lambda x: (x.get('mAP', 0), x.get('timestamp', 0)))
        training_result['best_stage_name'] = sorted_model_stages[-1]['stage_name']
        training_result['map'] = sorted_model_stages[-1]['mAP']

        # if too many stages, remove a earlest one
        if len(model_stages) > _MAX_MODEL_STAGES_COUNT_:
            sorted_model_stages = sorted(model_stages.values(), key=lambda x: x.get('timestamp', 0))
            del_stage_name = sorted_model_stages[0]['stage_name']
            if del_stage_name == training_result['best_stage_name']:
                del_stage_name = sorted_model_stages[1]['stage_name']
            del model_stages[del_stage_name]
            logging.info(f"data_writer removed model stage: {del_stage_name}")
        training_result['model_stages'] = model_stages

        # attachments, replace old value if valid
        if attachments:
            training_result['attachments'] = attachments
    else:
        warnings.warn('mutiple model stages is not supported, use write_training_result() instead')
        _files = training_result.get('model', [])

        training_result = {
            'model': list(set(files + _files)),
            'map': mAP,
            'timestamp': timestamp or int(time.time()),
            'stage_name': stage_name
        }

    # save all
    with open(env_config.output.training_result_file, 'w') as f:
        yaml.safe_dump(data=training_result, stream=f)


@deprecated(
    version="1.3.0",
    reason="This method is deprecated, recommand use write_model_stage() instead",
)
def write_training_result(model_names: List[str], mAP: float, **kwargs: dict) -> None:
    if multiple_model_stages_supportable():
        warnings.warn("multiple model stages is supported, use write_model_stage() instead")
        write_model_stage(stage_name="default_best_stage", files=model_names, mAP=mAP)
    else:
        training_result = {"model": model_names, "map": mAP}
        training_result.update(kwargs)

        env_config = env.get_current_env()
        with open(env_config.output.training_result_file, "w") as f:
            yaml.safe_dump(training_result, f)


def write_mining_result(mining_result: List[Tuple[str, float]]) -> None:
    # sort desc by score
    sorted_mining_result = sorted(mining_result, reverse=True, key=(lambda v: v[1]))

    env_config = env.get_current_env()
    with open(env_config.output.mining_result_file, "w") as f:
        for asset_id, score in sorted_mining_result:
            f.write(f"{asset_id}\t{score}\n")


@versionchanged(
    version="2.0.0",
    reason="support detection and segmentation infer result",
)
def write_infer_result(infer_result: Dict, algorithm: Optional[str] = 'detection') -> None:
    """
    write detection infer result
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
