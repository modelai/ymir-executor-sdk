import logging
import subprocess
import sys

from ymir_exc import monitor
from ymir_exc.util import YmirStage, get_bool, get_merged_config, get_ymir_process


class Executor(object):

    def __init__(self, apps: dict = None):
        self.apps = apps or dict(training='python3 ymir/ymir_training.py',
                                 mining='python3 ymir/ymir_mining.py',
                                 infer='python3 ymir/ymir_infer.py')

    def start(self) -> int:
        logging.basicConfig(stream=sys.stdout,
                            format='%(levelname)-8s: [%(asctime)s] %(message)s',
                            datefmt='%Y%m%d-%H:%M:%S',
                            level=logging.INFO)

        cfg = get_merged_config()

        logging.info(f'merged config: {cfg}')

        if cfg.ymir.run_training:
            self._run_training()
        elif cfg.ymir.run_mining or cfg.ymir.run_infer:
            # for multiple tasks, run mining first, infer later.
            if cfg.ymir.run_mining and cfg.ymir.run_infer:
                task_num = 2
                mining_task_idx = 0
                infer_task_idx = 1
            else:
                task_num = 1
                mining_task_idx = 0
                infer_task_idx = 0

            if cfg.ymir.run_mining:
                self._run_mining(task_idx=mining_task_idx, task_num=task_num)
            if cfg.ymir.run_infer:
                self._run_infer(task_idx=infer_task_idx, task_num=task_num)
        else:
            logging.warning('no task running')

        ymir_debug = get_bool(cfg, 'ymir_debug', False)
        if ymir_debug:
            raise Exception('set ymir_debug to True, just for debug')
        return 0

    def _run_training(self) -> None:
        command = self.apps['training']
        logging.info(f'training: {command}')
        subprocess.run(command.split(), check=True)
        monitor.write_monitor_logger(percent=1.0)

    def _run_mining(self, task_idx: int = 0, task_num: int = 1) -> None:
        command = self.apps['mining']
        logging.info(f'mining: {command}')
        subprocess.run(command.split(), check=True)
        monitor.write_monitor_logger(
            percent=get_ymir_process(stage=YmirStage.POSTPROCESS, p=1, task_idx=task_idx, task_num=task_num))

    def _run_infer(self, task_idx: int = 0, task_num: int = 1) -> None:
        command = self.apps['infer']
        logging.info(f'infer: {command}')
        subprocess.run(command.split(), check=True)
        monitor.write_monitor_logger(
            percent=get_ymir_process(stage=YmirStage.POSTPROCESS, p=1, task_idx=task_idx, task_num=task_num))
