"""

## contents of env.yaml
```
task_id: task0
run_training: True
run_mining: True
run_infer: True
input:
    root_dir: /in
    assets_dir: assets
    annotations_dir: annotations
    models_dir: models
    training_index_file: train-index.tsv
    val_index_file: val-index.tsv
    candidate_index_file: candidate-index.tsv
    config_file: config.yaml
output:
    root_dir: /out
    models_dir: models
    tensorboard_dir: tensorboard
    training_result_file: result.yaml
    mining_result_file: result.txt
    infer_result_file: infer-result.yaml
    monitor_file: monitor.txt
```

## dir and file structure
```
/in/assets
/in/annotations
/in/train-index.tsv
/in/val-index.tsv
/in/candidate-index.tsv
/in/config.yaml
/in/env.yaml
/out/models
/out/tensorboard
/out/monitor.txt
/out/monitor-log.txt
/out/ymir-executor-out.log
```

"""

from enum import IntEnum, auto

from easydict import EasyDict as edict
import yaml

from ymir_exc import settings


class DatasetType(IntEnum):
    UNKNOWN = auto()
    TRAINING = auto()
    VALIDATION = auto()
    CANDIDATE = auto()


def get_current_env() -> edict:
    with open(settings.DEFAULT_ENV_FILE_PATH, 'r') as f:
        return edict(yaml.safe_load(f.read()))


def get_executor_config() -> edict:
    with open(get_current_env().input.config_file, 'r') as f:
        return edict(yaml.safe_load(f))
