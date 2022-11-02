# dataset convert

## ymir origin dataset

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

## ymir to yolov5

```
from ymir_exc.dataset_convert import convert_ymir_to_yolov5
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
data_yaml_path = convert_ymir_to_yolov5(cfg)
```

### yolov5 format

```
/
├── out
│   ├── images # ln -s /in/assets /out/images
│   ├── labels
│   ├── train-index.txt
│   └── val-index.txt
```

## ymir to coco

```
from ymir_exc.dataset_convert import convert_ymir_to_coco
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
coco_dict = convert_ymir_to_coco(cat_id_from_zero=True)

for split in ['train', 'val']:
    print(coco_dict[split]['img_dir'])
    print(coco_dict[split]['ann_file'])
```

