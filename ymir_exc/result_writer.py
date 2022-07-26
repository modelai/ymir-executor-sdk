import json
import logging
import os
import time
from typing import Dict, List, Tuple

from pydantic import BaseModel
import yaml

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


def write_model_stage(stage_name: str,
                      files: List[str],
                      mAP: float,
                      timestamp: int = None) -> None:
    if not stage_name or not files:
        raise ValueError('empty stage_name or files')
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
    
    _files = training_result.get('model', [])

    training_result = {
        'model': list(set(files + _files)),
        'map': mAP,
        'timestamp': timestamp,
        'stage_name': stage_name
    }

    # save all
    with open(env_config.output.training_result_file, 'w') as f:
        yaml.safe_dump(data=training_result, stream=f)


def write_training_result(model_names: List[str], mAP: float, **kwargs: dict) -> None:
    training_result = {
        'model': model_names,
        'map': mAP
    }
    training_result.update(kwargs)

    env_config = env.get_current_env()
    with open(env_config.output.training_result_file, 'w') as f:
        yaml.safe_dump(training_result, f)


def write_mining_result(mining_result: List[Tuple[str, float]]) -> None:
    # sort desc by score
    sorted_mining_result = sorted(mining_result, reverse=True, key=(lambda v: v[1]))

    env_config = env.get_current_env()
    with open(env_config.output.mining_result_file, 'w') as f:
        for asset_id, score in sorted_mining_result:
            f.write(f"{asset_id}\t{score}\n")


def write_infer_result(infer_result: Dict[str, List[Annotation]]) -> None:
    detection_result = {}
    for asset_path, annotations in infer_result.items():
        asset_basename = os.path.basename(asset_path)
        detection_result[asset_basename] = {'annotations': [annotation.dict() for annotation in annotations]}

    result = {'detection': detection_result}
    env_config = env.get_current_env()
    with open(env_config.output.infer_result_file, 'w') as f:
        f.write(json.dumps(result))
