import time
from enum import IntEnum
from easydict import EasyDict as edict
from typing import Union
from tensorboardX import SummaryWriter

from ymir_exc import env

TASK_STATE_RUNNING = 2


class YmirTask(IntEnum):
    TRAINING = 1
    MINING = 2
    INFER = 3


def write_monitor_logger(percent: float) -> None:
    env_config = env.get_current_env()
    with open(env_config.output.monitor_file, 'w') as f:
        f.write(f"{env_config.task_id}\t{time.time()}\t{percent:.2f}\t{TASK_STATE_RUNNING}\n")


def write_monitor_logger_for_multiple_tasks(cfg: edict,
                                            task: Union[YmirTask, str],
                                            percent: float,
                                            order='tmi') -> None:
    """write monitor for multiple class
    current support follow case:
    1. training
    2. mining
    3. infer
    4. mining and infer
       * percent in [0, 1], will map to [0, 0,5] or [0,5, 1] according to order and task.
    """
    if isinstance(task, str):
        assert task in ['training', 'mining', 'infer'], f'unsupported task {task}'

    assert 0 <= percent <= 1, f'percent {percent} not in [0, 1]'
    assert order in ['tmi', 'tim'], f'unsupported order {order}'

    if cfg.ymir.run_infer and cfg.ymir.run_mining:
        if (order == 'tmi' and task in ['mining', YmirTask.MINING]) or (order == 'tim'
                                                                        and task in ['infer', YmirTask.INFER]):  # noqa
            write_monitor_logger(percent=0.5 * percent)
        else:
            write_monitor_logger(percent=0.5 + 0.5 * percent)
    else:
        write_monitor_logger(percent=percent)


def write_tensorboard_text(text: str, tag: str = None) -> None:
    """
    donot call this function too often, tensorboard may
    overwrite history log text with the same `tag` and `global_step`
    """
    env_config = env.get_current_env()
    tag = tag if tag else "default"

    # show the raw text format instead of markdown
    text = f"```\n {text} \n```"
    with SummaryWriter(env_config.output.tensorboard_dir) as f:
        f.add_text(tag=tag, text_string=text, global_step=round(time.time() * 1000))


def write_final_executor_log(tag: str = None) -> None:
    env_config = env.get_current_env()
    exe_log_file = env_config.output.executor_log_file
    with open(exe_log_file) as f:
        write_tensorboard_text(f.read(), tag=tag)
