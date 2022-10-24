"""

## contents of env.yaml
```
task_id: t000000100000166d7761660213748
protocol_version: 1.0.0
run_training: true
run_mining: false
run_infer: false
input:
  annotations_dir: /in/annotations
  assets_dir: /in/assets
  candidate_index_file: /in/candidate-index.tsv
  config_file: /in/config.yaml
  models_dir: /in/models
  root_dir: /in
  training_index_file: /in/train-index.tsv
  val_index_file: /in/val-index.tsv
output:
  infer_result_file: /out/infer-result.json
  mining_result_file: /out/result.tsv
  models_dir: /out/models
  monitor_file: /out/monitor.txt
  root_dir: /out
  tensorboard_dir: /out/tensorboard
  training_result_file: /out/models/result.yaml
```

## dir and file structure for training task
```
/
├── in
│   ├── annotations
│   ├── assets
│   ├── config.yaml
│   ├── env.yaml
│   ├── train-index.tsv
│   └── val-index.tsv
├── out
│   ├── models
│   ├── monitor.txt
│   ├── tensorboard
│   └── ymir-executor-out.log
```

"""

from enum import IntEnum, auto

import yaml
from pydantic import BaseModel

from ymir_exc import settings


class DatasetType(IntEnum):
    UNKNOWN = auto()
    TRAINING = auto()
    VALIDATION = auto()
    CANDIDATE = auto()


class EnvInputConfig(BaseModel):
    root_dir: str = '/in'
    assets_dir: str = '/in/assets'
    annotations_dir: str = '/in/annotations'
    models_dir: str = '/in/models'
    training_index_file: str = ''
    val_index_file: str = ''
    candidate_index_file: str = ''
    config_file: str = '/in/config.yaml'


class EnvOutputConfig(BaseModel):
    root_dir: str = '/out'
    models_dir: str = '/out/models'
    tensorboard_dir: str = '/out/tensorboard'
    training_result_file: str = '/out/models/result.yaml'
    mining_result_file: str = '/out/result.tsv'
    infer_result_file: str = '/out/infer-result.json'
    monitor_file: str = '/out/monitor.txt'
    executor_log_file: str = '/out/ymir-executor-out.log'


class EnvConfig(BaseModel):
    task_id: str = 'default-task'
    protocol_version: str = '0.0.1'  # input/output api version
    run_training: bool = False
    run_mining: bool = False
    run_infer: bool = False

    input: EnvInputConfig = EnvInputConfig()
    output: EnvOutputConfig = EnvOutputConfig()


def get_current_env() -> EnvConfig:
    with open(settings.DEFAULT_ENV_FILE_PATH, 'r') as f:
        return EnvConfig.parse_obj(yaml.safe_load(f.read()))


def get_executor_config() -> dict:
    with open(get_current_env().input.config_file, 'r') as f:
        executor_config = yaml.safe_load(f)
    return executor_config
