from typing import Tuple

import torch


def get_gpu_memory(device: str = 'cuda:0') -> Tuple[float, float]:
    """
    code from https://github.com/ultralytics/yolov5
    return total memory, free memory
    """
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free

    return t, f
