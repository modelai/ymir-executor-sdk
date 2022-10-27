# read function

update 2022/10/27

## read config

### read ymir config

read `/in/env.yaml`

```
from ymir_exc import env

ymir_env_config = env.get_current_env()
```

### read hyper-parameters config

read `/in/config.yaml`

```
from ymir_exc import env

param_config = env.get_executor_config()
```

### read ymir + hyper-parameters config

read `/in/config.yaml` and `/in/env.yaml`

```
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
```

## read dataset

```
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
```

### read training dataset

read `/in/train-index.tsv`

```
with open(cfg.ymir.input.training_index_file, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        img_abs_path, ann_abs_path = line.split()
```

### read validation dataset

the same with training dataset, read `/in/val-index.tsv`

```
with open(cfg.ymir.input.val_index_file, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        img_abs_path, ann_abs_path = line.split()
```

### read mining or infer dataset

read `/in/condidate-index.tsv`

```
with open(cfg.ymir.input.candidate_index_file, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        img_abs_path = line.strip()
```
