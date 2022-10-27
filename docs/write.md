# write function

## monitor process

write a float `percent` to `cfg.ymir.output.monitor_file` (/out/monitor.txt)

```
from ymir_exc import monitor

monitor.write_monitor_logger(percent)
```

## write monitor for multiple tasks

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

## write training result

write `map` (map@0.5) and files to `cfg.ymir.output.training_result_file` (/out/models/result.yaml)

```
from ymir_exc import result_writer

result_writer.write_training_result(model_names=['best.pt', 'best.onnx', 'config.yaml'], mAP=0.8)
```

for complicated case, you want to save multiple stand-alone weight files
```
from ymir_exc.util import write_ymir_training_result

write_ymir_training_result(cfg, map50=0.8, files=['epoch10.pt', 'config.yaml'], id='epoch_10')

write_ymir_training_result(cfg, map50=0.9, files=['epoch20.pt', 'config.yaml'], id='epoch_20')
```

## write mining result

write image with mining score to `cfg.ymir.output.mining_result_file` (/out/result.tsv)

```
from ymir_exc import result_writer

result_writer.write_mining_result(mining_result=[(img_path1, 0.8), (img_path2, 0.6)])
```

## write infer result

write image filename with prediction result to `cfg.ymir.output.infer_result_file` (/out/infer-result.json)

```
from ymir_exc import result_writer
from ymir_exc.result_writer import Annotation, Box

ann1 = Annotation(class_name = 'dog', score = 0.8, box = Box(x=10, y=20, w=10, h=10))
result_writer.write_infer_result(infer_result=dict(img_path1=[ann1, ann1, ann1], img_path2=[]))
```
