# 将结果写回ymir平台

2023/03/01 更新

## 写进度

::: ymir_exc.monitor.write_monitor_logger

将任务进度百分比 `percent` 写到监控文件 `cfg.ymir.output.monitor_file` (/out/monitor.txt) 中

```
from ymir_exc import monitor

monitor.write_monitor_logger(percent)
```

## 保存训练日志(tensorboard)

需要将tensorboard 日志文件保存到指定目录 `cfg.ymir.output.tensorboard_dir` (/out/tensorboard) 中

## 保存训练结果

::: ymir_exc.result_writer.write_training_result

将验证集评测指标及相关文件保存到结果文件 `cfg.ymir.output.training_result_file` (/out/models/result.yaml) 中

```
from ymir_exc import result_writer

## 目标检测结果
result_writer.write_training_result(stage_name='best', files=['best.pt', 'best.onnx', 'config.yaml'], evaluation_result=dict(mAP=0.8))

## 语义分割结果
result_writer.write_training_result(stage_name='best', files=['best.pt', 'best.onnx', 'config.yaml'], evaluation_result=dict(mIoU=0.8))

## 实例分割结果
result_writer.write_training_result(stage_name='best', files=['best.pt', 'best.onnx', 'config.yaml'], evaluation_result=dict(maskAP=0.8))
```

### 保存多组训练结果
```
from ymir_exc import result_writer

result_writer.write_training_result(stage_name='epoch_10', files=['epoch_10.pt', 'config.yaml'], evaluation_result=dict(mAP=0.82))

result_writer.write_training_result(stage_name='epoch_20', files=['epoch_20.pt', 'config.yaml'], evaluation_result=dict(mAP=0.84))
```

## 写挖掘结果

::: ymir_exc.result_writer.write_mining_result

将图像路径与对应分数写到结果文件 `cfg.ymir.output.mining_result_file` (/out/result.tsv) 中

```
from ymir_exc import result_writer

result_writer.write_mining_result(mining_result=[(img_path1, 0.8), (img_path2, 0.6)])
```

## 写推理结果

::: ymir_exc.result_writer.write_infer_result

将图像路径与对应推理结果写到 `cfg.ymir.output.infer_result_file` (/out/infer-result.json)

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

## 进阶功能

### 写进度，考虑多个过程

每个过程有对应的进度，对每个过程进行加权，即为整体进度。

mining and infer can run in the same task.
we support mining process percent in [0, 0.5], infer in [0.5, 1.0]

then we can:
```
# for mining, mining_percent in [0, 1]
monitor.write_monitor_logger(mining_percent * 0.5)

# for infer, infer_percent in [0, 1]
monitor.write_monitor_logger(0.5 + infer_percent * 0.5)
```

or we can
```
from ymir_exc import monitor

monitor.write_monitor_logger_for_mutiple_tasks(cfg, task='infer', percent=infer_percent, order='tmi')
```

if we consider more complicated case, mining task can divide into preprocess, task and postprocess stage, then we can use:

```
from ymir_exc.util import write_ymir_monitor_process, YmirStage

write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=0.2, stage=YmirStage.PREPROCESS, stage_weights = [0.1, 0.8, 0.1], task_order='tmi')
```
