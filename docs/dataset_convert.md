# 数据集转换

2023/03/01 更新

## ymir 数据集

```
/
├── in
│   ├── annotations
│   ├── assets
│   ├── config.yaml
│   ├── env.yaml
│   ├── train-index.tsv
│   └── val-index.tsv
```

# 目录检测

## 转换为yolov5格式

```
from ymir_exc.dataset_convert import convert_ymir_to_yolov5
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
data_yaml_path = convert_ymir_to_yolov5(cfg)
```

### 转换结果

```
/
├── out
│   ├── images # ln -s /in/assets /out/images
│   ├── labels
│   ├── train-index.txt
│   └── val-index.txt
```

## 转换为coco格式

适应于mmdetection等支持coco格式的框架

```
from ymir_exc.dataset_convert import convert_ymir_to_coco
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
coco_dict = convert_ymir_to_coco(cat_id_from_zero=True)

for split in ['train', 'val']:
    print(coco_dict[split]['img_dir'])
    print(coco_dict[split]['ann_file'])
```

### 转换结果

转换后，会在/out/ymir_dataset目录下生成训练集标注文件与测试集标注文件

```
/out
└── ymir_dataset
    ├── ymir_train.json
    └── ymir_val.json
```

