# ymir镜像开发sdk

## 依赖

python >= 3.7

## 安装

```
pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.1.0"
```

## 使用

- 获取ymir平台配置，超参数与数据集信息

```
from ymir_exc.util import get_merged_config

cfg = get_merged_config()
```

- 保存进度结果

```
from ymir_exc import monitor

monitor.write_monitor_logger(percent)
```

- 保存训练结果

```
from ymir_exc import result_writer

## 目标检测结果
result_writer.write_training_result(stage_name='best', files=['best.pt', 'best.onnx', 'config.yaml'], evaluation_result=dict(mAP=0.8))

## 语义分割结果
result_writer.write_training_result(stage_name='best', files=['best.pt', 'best.onnx', 'config.yaml'], evaluation_result=dict(mIoU=0.8))

## 实例分割结果
result_writer.write_training_result(stage_name='best', files=['best.pt', 'best.onnx', 'config.yaml'], evaluation_result=dict(maskAP=0.8))
```

- 保存推理结果

```
from ymir_exc import result_writer
from ymir_exc.result_writer import Annotation, Box

## 目标检测结果
ann1 = Annotation(class_name = 'dog', score = 0.8, box = Box(x=10, y=20, w=10, h=10))
ann2 = Annotation(class_name = 'cat', score = 0.6, box = Box(x=10, y=20, w=10, h=10))
result_writer.write_infer_result(infer_result=dict(img_path1=[ann1, ann2], img_path2=[]))

## 语义分割与实例分割结果
coco_result = dict()
...
result_writer.write_infer_result(infer_result=coco_result, algorithm='segmentation')
```

- 保存挖掘结果

```
from ymir_exc import result_writer

result_writer.write_mining_result(mining_result=[(img_path1, 0.8), (img_path2, 0.6)])
```
