# 读取ymir平台的配置，超参数与数据集

2023/03/01 更新

## 配置与超参数

解析 `/in/env.yaml`，可获得任务id, 任务类型等配置信息。
解析 `/in/config.yaml`, 可获得可用gpu，任务超参数，数据集标签，预训练模型等超参数信息。

### 读取配置与超参数

同时解析 `/in/env.yaml` 与 `/in/config.yaml`， 其中 `cfg.param` 包含超参数信息， `cfg.ymir` 包含配置信息。

```
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
```

### 单独读取配置

单独解析 `/in/env.yaml`

```
from ymir_exc import env

ymir_env_config = env.get_current_env()
```

### 单独读取超参数

单独解析 `/in/config.yaml`

```
from ymir_exc import env

param_config = env.get_executor_config()
```


## 读取数据集

### 读取配置与超参数

数据集索引文件的路径可以通过 `cfg` 获得。

```
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
```

### 读取训练集

训练集索引文件 `/in/train-index.tsv` 的每一行包含一个图像的绝对路径及对应标注文件的绝对路径，两者以制作符进行分隔。

```
with open(cfg.ymir.input.training_index_file, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        img_abs_path, ann_abs_path = line.split()
```

### 读取验证集

验证集索引文件 `/in/val-index.tsv` 与训练集索引文件的结构一样。

```
with open(cfg.ymir.input.val_index_file, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        img_abs_path, ann_abs_path = line.split()
```

### 读取推理或挖掘集

推理与挖掘集共享同一个索引文件 `/in/condidate-index.tsv`，它的每一行均为图像的绝对路径。

```
with open(cfg.ymir.input.candidate_index_file, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        img_abs_path = line.strip()
```
