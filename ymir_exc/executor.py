import logging
import subprocess
import sys
from typing import Optional

from easydict import EasyDict as edict

from ymir_exc import monitor
from ymir_exc.util import (YmirStage, get_bool, get_merged_config, write_ymir_monitor_process)


class Executor(object):

    def __init__(self, apps: Optional[dict] = None):
        self.apps = apps or dict(
            training="python3 ymir/ymir_training.py",
            mining="python3 ymir/ymir_mining.py",
            infer="python3 ymir/ymir_infer.py",
        )

    def start(self) -> int:
        logging.basicConfig(
            stream=sys.stdout,
            format="%(levelname)-8s: [%(asctime)s] %(message)s",
            datefmt="%Y%m%d-%H:%M:%S",
            level=logging.INFO,
        )

        cfg = get_merged_config()

        logging.info(f"merged config: {cfg}")

        if cfg.ymir.run_training:
            self._run_training()
        elif cfg.ymir.run_mining or cfg.ymir.run_infer:
            # for multiple tasks, run mining first, infer later.

            if cfg.ymir.run_mining:
                self._run_mining(cfg)
            if cfg.ymir.run_infer:
                self._run_infer(cfg)
        else:
            logging.warning("no task running")

        ymir_debug = get_bool(cfg, "ymir_debug", False)
        if ymir_debug:
            raise Exception("set ymir_debug to True, just for debug")
        return 0

    def _run_training(self) -> None:
        command = self.apps["training"]
        logging.info(f"start training: {command}")
        subprocess.run(command.split(), check=True)
        monitor.write_monitor_logger(percent=1.0)
        logging.info("training finished")

    def _run_mining(self, cfg: edict) -> None:
        command = self.apps["mining"]
        logging.info(f"start mining: {command}")
        subprocess.run(command.split(), check=True)
        write_ymir_monitor_process(cfg, task="mining", naive_stage_percent=1, stage=YmirStage.POSTPROCESS)
        logging.info("mining finished")

    def _run_infer(self, cfg: edict) -> None:
        command = self.apps["infer"]
        logging.info(f"start infer: {command}")
        subprocess.run(command.split(), check=True)
        write_ymir_monitor_process(cfg, task="infer", naive_stage_percent=1, stage=YmirStage.POSTPROCESS)
        logging.info("infer finished")
